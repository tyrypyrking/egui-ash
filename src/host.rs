use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use ash::vk;
use egui_winit::winit::window::WindowId;

use crate::engine::EngineRenderer;
use crate::engine_thread::{
    self, EngineHealthState, HEALTH_CRASHED, HEALTH_RUNNING, HEALTH_STARTING, HEALTH_STOPPED,
};
use crate::image_registry::ImageRegistry;
use crate::mailbox::{self, MailboxReceiver};
use crate::render_targets::RenderTargetPool;
use crate::state_exchange::{self, StateReader, StateWriter};
use crate::storage::Storage;
use crate::types::{
    EngineContext, EngineHealth, EngineRestartError, EngineStatus, RunOption, VulkanContext,
};
use crate::viewport::Viewport;

// ─────────────────────────────────────────────────────────────────────────────
// EngineHandle — public API for restarting the engine from the UI closure
// ─────────────────────────────────────────────────────────────────────────────

/// Handle for restarting the engine from the UI closure.
///
/// Passed to the UI closure each frame. Call [`EngineHandle::restart`] to
/// replace the crashed engine with a fresh instance. The actual restart is
/// deferred until after the UI closure returns.
pub struct EngineHandle<E: EngineRenderer> {
    restart_request: std::cell::RefCell<Option<E>>,
    exit_request: std::cell::RefCell<Option<std::process::ExitCode>>,
    health: Arc<EngineHealthState>,
    image_registry: ImageRegistry,
}

impl<E: EngineRenderer> EngineHandle<E> {
    fn new(health: Arc<EngineHealthState>, image_registry: ImageRegistry) -> Self {
        Self {
            restart_request: std::cell::RefCell::new(None),
            exit_request: std::cell::RefCell::new(None),
            health,
            image_registry,
        }
    }

    /// Borrow the image registry for registering user-owned Vulkan
    /// textures to display in egui panels. See [`ImageRegistry`] for the
    /// full API. Backed by the same channel as
    /// [`EngineContext::image_registry`], so registrations from either
    /// thread are honoured.
    pub fn image_registry(&self) -> &ImageRegistry {
        &self.image_registry
    }

    /// Request that the engine be restarted with a fresh instance.
    ///
    /// Returns `Err(EngineRestartError::StillRunning)` if the engine has not
    /// yet stopped or crashed. The restart is deferred until after the current
    /// UI frame completes.
    pub fn restart(&self, engine: E) -> Result<(), EngineRestartError> {
        let h = self.health.health.load(Ordering::Acquire);
        if h == HEALTH_STARTING || h == HEALTH_RUNNING {
            return Err(EngineRestartError::StillRunning);
        }
        *self.restart_request.borrow_mut() = Some(engine);
        Ok(())
    }

    /// Request a clean app shutdown with the given exit code.
    ///
    /// The shutdown is deferred until after the current UI frame completes;
    /// the event loop will then run the normal destroy path (save persisted
    /// state, wait for engine thread, destroy Vulkan resources) before
    /// exiting with `code`. Safe to call multiple times — only the most
    /// recent code is retained.
    pub fn exit(&self, code: std::process::ExitCode) {
        *self.exit_request.borrow_mut() = Some(code);
    }

    fn take_restart(&self) -> Option<E> {
        self.restart_request.borrow_mut().take()
    }

    fn take_exit(&self) -> Option<std::process::ExitCode> {
        self.exit_request.borrow_mut().take()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HostPtr — raw-pointer wrapper for egui's immediate-viewport renderer
// ─────────────────────────────────────────────────────────────────────────────

/// Raw-pointer wrapper used to pass `*mut Host<E>` through egui's
/// `Context::set_immediate_viewport_renderer`, which requires an owned
/// `'static` closure.
///
/// The renderer callback is a `Fn + 'static`, but egui's
/// `IMMEDIATE_VIEWPORT_RENDERER` thread-local does NOT require the
/// callback to be `Send + Sync` (it's strictly main-thread-accessed).
/// So this wrapper stays safe-to-construct; the unsafety lives entirely
/// at the single call site in [`Host::install_immediate_viewport_renderer`].
///
/// See that method's safety invariant for the lifetime / pinning rules
/// that make dereferencing `HostPtr::0` sound inside the closure.
struct HostPtr<E: EngineRenderer>(*mut Host<E>);

impl<E: EngineRenderer> Clone for HostPtr<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: EngineRenderer> Copy for HostPtr<E> {}

// ─────────────────────────────────────────────────────────────────────────────
// Host — internal orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Host-side orchestrator. Owns all host resources and implements the
/// per-frame protocol: poll engine frames, run egui, composite, present.
pub(crate) struct Host<E: EngineRenderer> {
    // Vulkan context (owned)
    vulkan: VulkanContext,

    // Shared egui context — one per Host, shared across all viewports.
    context: egui::Context,

    // Swapchain present mode — captured from RunOption for child
    // viewport creation (immediate and deferred viewports inherit the
    // ROOT's present mode for the alpha; per-viewport overrides are a
    // post-alpha feature).
    present_mode: vk::PresentModeKHR,

    // Pointer to the `&ActiveEventLoop` currently driving a
    // `context.run(...)` call. Set by [`Host::frame`] at the start of
    // every frame, cleared at the end. Non-null only during egui's
    // run phase on the main thread. The immediate-viewport renderer
    // reads this to create child windows — winit only allows window
    // creation from inside `ApplicationHandler` methods, so we must
    // smuggle the reference across `context.run()`.
    active_event_loop: std::cell::Cell<*const egui_winit::winit::event_loop::ActiveEventLoop>,

    // Per-egui-viewport state: window + egui-winit adapter + compositor +
    // surface. ROOT is always present. Non-root entries appear/disappear
    // as egui spawns immediate or deferred viewports (§B9; multi-viewport
    // steps still to come — at this step the map size is always 1).
    viewports: HashMap<egui::ViewportId, Viewport>,

    // Reverse index for routing winit WindowEvents by WindowId.
    // Populated when a viewport is created; pruned on close. Kept in
    // sync with `viewports` as a strict invariant.
    window_id_to_viewport: HashMap<WindowId, egui::ViewportId>,

    // Focused viewport — tracked so host-global operations can target
    // the right window. Currently always ROOT; populated for real when
    // focus events are routed per-viewport in later steps.
    #[allow(dead_code)]
    focused_viewport: Option<egui::ViewportId>,

    // Render target pool (double-buffered images shared with engine)
    render_target_pool: RenderTargetPool,

    // Engine thread communication
    target_tx: mpsc::SyncSender<crate::types::RenderTarget>,
    frame_rx: MailboxReceiver,
    event_tx: mpsc::Sender<crate::event::EngineEvent>,
    engine_join: Option<std::thread::JoinHandle<()>>,

    // State exchange
    ui_state_writer: StateWriter<E::UiState>,
    engine_state_reader: StateReader<E::EngineState>,

    // Engine health + handle
    health: Arc<EngineHealthState>,
    engine_handle: EngineHandle<E>,

    // Render target tracking:
    //   engine_target_index:    index the engine is currently rendering into (or None)
    //   composited_target_index: index currently sampled by the compositor (or None)
    engine_target_index: Option<usize>,
    composited_target_index: Option<usize>,

    // Current engine-viewport image view broadcast to every
    // compositor's `engine_viewport_descriptor_set`. `None` means the
    // compositors are showing their own black placeholder (engine has
    // crashed, stopped, or not yet produced a frame). `Some(view)`
    // means every compositor samples this view. Kept on Host so newly-
    // created child viewports can be sync'd to the current state.
    current_engine_view: Option<vk::ImageView>,

    // Persistent UI state (survives across frames)
    ui_state: E::UiState,

    // Options
    clear_color: [f32; 4],

    // Exit channel
    _exit_tx: mpsc::Sender<std::process::ExitCode>,

    // Persistence — Storage is always present (feature-off = stub).
    // The two flags control whether automatic window/egui-memory save
    // happens on shutdown; user-level `set_value` / `get_value` persist
    // independently of the flags when the feature is enabled.
    storage: Storage,
    #[cfg(feature = "persistence")]
    persistent_windows: bool,
    #[cfg(feature = "persistence")]
    persistent_egui_memory: bool,
    /// `None` disables periodic flush (only save on destroy).
    #[cfg(feature = "persistence")]
    auto_save_interval: Option<Duration>,
    /// Instant of the most recent auto-save flush. Used to gate the next
    /// periodic save in `frame`.
    #[cfg(feature = "persistence")]
    last_auto_save: std::time::Instant,
}

impl<E: EngineRenderer> Host<E> {
    /// Borrow the ROOT viewport.
    ///
    /// # Panics
    /// Panics if ROOT has been removed. ROOT is inserted in `new` and
    /// removed last in `destroy`; no other code path may remove it.
    fn root(&self) -> &Viewport {
        self.viewports
            .get(&egui::ViewportId::ROOT)
            .expect("ROOT viewport invariant violated")
    }

    /// Borrow the ROOT viewport mutably. Same invariant as [`Self::root`].
    fn root_mut(&mut self) -> &mut Viewport {
        self.viewports
            .get_mut(&egui::ViewportId::ROOT)
            .expect("ROOT viewport invariant violated")
    }

    /// Broadcast a new engine-viewport image view (or "show black") to
    /// every compositor's `engine_viewport_descriptor_set`. Implements
    /// B9 decision #2: all viewports sample the same engine image; one
    /// `vkUpdateDescriptorSets` per compositor per new engine frame.
    ///
    /// Also updates `self.current_engine_view` so newly-created child
    /// viewports (after this call) can be sync'd to the current state
    /// via [`Self::sync_child_engine_viewport`].
    ///
    /// # Safety
    /// `view`, when `Some`, must remain valid until the next call.
    /// Caller is responsible for the engine-side fence handshake that
    /// makes this sound (see `render_target_pool` / `engine_timeline`).
    unsafe fn broadcast_engine_viewport(&mut self, view: Option<vk::ImageView>) {
        self.current_engine_view = view;
        for vp in self.viewports.values_mut() {
            match view {
                Some(v) => vp.compositor.set_engine_viewport(v),
                None => vp.compositor.set_engine_viewport_black(),
            }
        }
    }

    /// Apply `self.current_engine_view` to a single viewport's compositor.
    /// Called after creating a new child Viewport so it samples the
    /// same engine image ROOT is seeing rather than its own black
    /// placeholder.
    ///
    /// # Safety
    /// Same as [`Self::broadcast_engine_viewport`] plus: `id` must be
    /// present in `self.viewports`.
    unsafe fn sync_child_engine_viewport(&mut self, id: egui::ViewportId) {
        let current = self.current_engine_view;
        let Some(vp) = self.viewports.get_mut(&id) else {
            return;
        };
        match current {
            Some(view) => vp.compositor.set_engine_viewport(view),
            None => {
                // Child compositor's `Compositor::new` already leaves
                // the engine viewport pointed at its own black image;
                // nothing to do.
            }
        }
    }

    /// Create a new Host, spawning the engine thread.
    ///
    /// # Safety
    /// Caller must ensure all Vulkan handles in `vulkan` are valid and that
    /// the window/event_loop are on the main thread.
    pub(crate) unsafe fn new(
        vulkan: VulkanContext,
        engine: E,
        options: &RunOption,
        window: egui_winit::winit::window::Window,
        event_loop: &egui_winit::winit::event_loop::ActiveEventLoop,
        exit_tx: mpsc::Sender<std::process::ExitCode>,
        storage: Storage,
    ) -> Self {
        // 1.0.0-alpha restriction: cross-queue-family ownership transfer for
        // the engine viewport image is not yet implemented on the host side
        // (see docs/known-limitations.md). Reject mismatched families now
        // rather than silently produce UB when sampling the texture.
        assert_eq!(
            vulkan.host_queue_family_index, vulkan.engine_queue_family_index,
            "egui-ash 1.0.0-alpha requires host_queue_family_index == engine_queue_family_index; \
             cross-family ownership transfer is not yet supported"
        );

        // User-texture registry channel — the ROOT compositor drains
        // commands at the top of each frame. Per B9 decision #8, user
        // textures render in ROOT only for the alpha; non-root compositors
        // (added in later steps) receive a pre-closed dummy channel.
        let (image_registry, registry_rx) = crate::image_registry::new_pair();

        // Shared egui context — created BEFORE the ROOT viewport so the
        // viewport's egui-winit adapter can clone it.
        let context = egui::Context::default();
        let theme = if options.follow_system_theme {
            window.theme().or(Some(options.default_theme))
        } else {
            Some(options.default_theme)
        };

        // Derive the initial ViewportBuilder from user options so the ROOT
        // viewport carries it for later persistence/update passes.
        let builder = options
            .viewport_builder
            .clone()
            .unwrap_or_else(|| egui::ViewportBuilder::default().with_title("egui-ash"));

        let root_viewport = Viewport::new_root(
            &vulkan,
            event_loop,
            window,
            builder,
            &context,
            theme,
            options.present_mode,
            registry_rx,
        );
        let root_window_id = root_viewport.window.id();

        // Render target pool uses the compositor's actual swapchain extent
        // (may differ from `window.inner_size()` after surface-caps clamp).
        let extent = root_viewport.compositor.swapchain_extent();
        let format = vk::Format::B8G8R8A8_UNORM;
        let render_target_pool = RenderTargetPool::new(
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            extent,
            format,
        );

        // Channels
        let (target_tx, target_rx) = mailbox::target_channel();
        let (frame_tx, frame_rx) = mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::channel();

        // State exchange
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // Health
        let health = EngineHealthState::new();
        let engine_handle = EngineHandle::new(Arc::clone(&health), image_registry.clone());

        // Engine context
        let engine_ctx = EngineContext {
            device: vulkan.device.clone(),
            queue: vulkan.engine_queue,
            queue_family_index: vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
            queue_mutex: vulkan.queue_mutex.clone(),
            image_registry: image_registry.clone(),
        };

        // Spawn engine thread
        let engine_join = engine_thread::spawn_engine_thread(
            engine,
            engine_ctx,
            target_rx,
            frame_tx,
            event_rx,
            ui_state_reader,
            engine_state_writer,
            Arc::clone(&health),
        );

        // Send first render target (index 0)
        let mut render_target_pool = render_target_pool;
        let first_target = render_target_pool.make_target(0);
        target_tx
            .send(first_target)
            .expect("failed to send initial render target");

        // Restore persisted egui memory (collapsed panels, scroll state, etc.)
        // before the first frame runs. Safe to do after the viewport is
        // constructed — memory lives on the Context, not the winit state.
        #[cfg(feature = "persistence")]
        if options.persistent_egui_memory {
            if let Some(mem) = storage.get_egui_memory() {
                context.memory_mut(|m| *m = mem);
            }
        }

        // Assemble the viewport map + WindowId reverse index.
        let mut viewports = HashMap::new();
        viewports.insert(egui::ViewportId::ROOT, root_viewport);
        let mut window_id_to_viewport = HashMap::new();
        window_id_to_viewport.insert(root_window_id, egui::ViewportId::ROOT);

        Self {
            vulkan,
            context,
            present_mode: options.present_mode,
            active_event_loop: std::cell::Cell::new(std::ptr::null()),
            viewports,
            window_id_to_viewport,
            focused_viewport: Some(egui::ViewportId::ROOT),
            render_target_pool,
            target_tx,
            frame_rx,
            event_tx,
            engine_join: Some(engine_join),
            ui_state_writer,
            engine_state_reader,
            health,
            engine_handle,
            engine_target_index: Some(0), // index 0 was sent to engine
            composited_target_index: None,
            current_engine_view: None,
            ui_state: E::UiState::default(),
            clear_color: options.clear_color,
            _exit_tx: exit_tx,
            storage,
            #[cfg(feature = "persistence")]
            persistent_windows: options.persistent_windows,
            #[cfg(feature = "persistence")]
            persistent_egui_memory: options.persistent_egui_memory,
            #[cfg(feature = "persistence")]
            auto_save_interval: options.auto_save_interval,
            #[cfg(feature = "persistence")]
            last_auto_save: std::time::Instant::now(),
        }
    }

    /// Run a single host frame: poll engine, run egui UI, composite, present.
    ///
    /// `event_loop` is smuggled to the immediate-viewport renderer for
    /// the duration of `context.run()` via [`Self::active_event_loop`] —
    /// child windows must be created from an `ActiveEventLoop` context,
    /// which is only reachable during `ApplicationHandler` callbacks.
    ///
    /// # Safety
    /// Caller must ensure the Vulkan device and all handles are valid.
    pub(crate) unsafe fn frame(
        &mut self,
        event_loop: &egui_winit::winit::event_loop::ActiveEventLoop,
        ui: &mut impl FnMut(
            &egui::Context,
            &EngineStatus,
            &mut E::UiState,
            &E::EngineState,
            &EngineHandle<E>,
            &mut Storage,
        ),
    ) {
        // ── 1. Check for new engine frame ──────────────────────────────────
        if let Some(completed) = self.frame_rx.try_recv() {
            // CPU-wait on the engine timeline for the engine's signal_value.
            // This ensures the GPU has finished rendering into this target.
            let semaphores = [self.render_target_pool.engine_timeline()];
            let values = [completed.signal_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            self.vulkan
                .device
                .wait_semaphores(&wait_info, u64::MAX)
                .expect("wait_semaphores failed");

            // Figure out which target index the engine just finished.
            let finished_index = if completed.image == self.render_target_pool.image(0) {
                0
            } else {
                1
            };

            // Update ROOT compositor to sample this image. Non-root
            // compositors get the same update in step 8 (decision #2).
            // Broadcast to all compositors so non-root viewports can
            // render the engine texture too (B9 decision #2).
            let image_view = self.render_target_pool.image_view(finished_index);
            self.broadcast_engine_viewport(Some(image_view));

            // Send the next render target to the engine.
            let send_index = if let Some(prev) = self.composited_target_index {
                prev
            } else if finished_index == 0 {
                // First engine frame — the other target is free.
                1
            } else {
                0
            };
            let target = self.render_target_pool.make_target(send_index);
            // If send fails, the engine thread has exited — that's fine.
            let _ = self.target_tx.send(target);

            self.composited_target_index = Some(finished_index);
            self.engine_target_index = None;
        }

        // ── 2. If engine crashed and nothing composited, show black ────────
        let health_val = self.health.health.load(Ordering::Acquire);
        if (health_val == HEALTH_CRASHED || health_val == HEALTH_STOPPED)
            && self.composited_target_index.is_none()
        {
            self.broadcast_engine_viewport(None);
        }

        // ── 3. Build EngineStatus ──────────────────────────────────────────
        let health = match health_val {
            HEALTH_STARTING => EngineHealth::Starting,
            HEALTH_RUNNING => EngineHealth::Running,
            HEALTH_STOPPED => EngineHealth::Stopped,
            HEALTH_CRASHED => {
                let msg = self
                    .health
                    .crash_message
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string());
                EngineHealth::Crashed { message: msg }
            }
            _ => EngineHealth::Stopped,
        };
        let frames_delivered = self.health.frames_delivered.load(Ordering::Relaxed);
        let last_ns = self.health.last_frame_time_ns.load(Ordering::Relaxed);
        let last_frame_time = if last_ns > 0 {
            Some(Duration::from_nanos(last_ns))
        } else {
            None
        };
        let engine_status = EngineStatus {
            health,
            viewport_texture_id: self.root().compositor.engine_viewport_texture_id(),
            frames_delivered,
            last_frame_time,
        };

        // ── 4. Read latest engine state ────────────────────────────────────
        let engine_state = self.engine_state_reader.read();

        // ── 5. Run egui frame ──────────────────────────────────────────────
        let raw_input = {
            let root = self.root_mut();
            root.state.take_egui_input(&root.window)
        };

        // Smuggle the live `&ActiveEventLoop` to the immediate-viewport
        // renderer for the duration of `context.run()`. See
        // `Host::active_event_loop` docs for the invariant.
        self.active_event_loop.set(event_loop as *const _);
        let full_output = self.context.run(raw_input, |ctx| {
            ui(
                ctx,
                &engine_status,
                &mut self.ui_state,
                &engine_state,
                &self.engine_handle,
                &mut self.storage,
            );
        });
        self.active_event_loop.set(std::ptr::null());

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = full_output;

        {
            let root = self.root_mut();
            root.state
                .handle_platform_output(&root.window, platform_output);
        }

        // ── 5b. Periodic auto-save flush (crash safety) ────────────────────
        // Captures current window geometry and egui memory into the Storage
        // in-memory store and flushes the RON file if the interval elapsed.
        // `flush()` is a no-op when nothing was written since the last flush
        // (Storage tracks a dirty bit), so repeated calls are cheap.
        #[cfg(feature = "persistence")]
        if let Some(interval) = self.auto_save_interval {
            if self.last_auto_save.elapsed() >= interval {
                if self.persistent_windows {
                    let zoom = self.context.zoom_factor();
                    let settings =
                        egui_winit::WindowSettings::from_window(zoom, &self.root().window);
                    let mut map = std::collections::HashMap::new();
                    map.insert(egui::ViewportId::ROOT, settings);
                    self.storage.set_windows(&map);
                }
                if self.persistent_egui_memory {
                    let mem = self.context.memory(|m| m.clone());
                    self.storage.set_egui_memory(&mem);
                }
                self.storage.flush();
                self.last_auto_save = std::time::Instant::now();
            }
        }

        // ── 6. Publish UI state ────────────────────────────────────────────
        self.ui_state_writer.publish(self.ui_state.clone());

        // ── 7. Handle restart request ──────────────────────────────────────
        if let Some(new_engine) = self.engine_handle.take_restart() {
            self.restart_engine(new_engine);
        }

        // ── 8. Tessellate and composite ────────────────────────────────────
        //
        // `screen_size` must be the *actual* swapchain image dimensions, not
        // `window.inner_size()`. On Wayland (and some X11/driver combinations)
        // `VkSurfaceCapabilitiesKHR::currentExtent` comes back in logical
        // pixels, so the swapchain is created smaller than the window's
        // physical inner size. If we kept using `window.inner_size()` here
        // the Vulkan viewport would overshoot the framebuffer.
        //
        // `scale_factor` = physical swapchain pixels per egui point.
        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let sc_extent = self.root().compositor.swapchain_extent();
        let screen_size = [sc_extent.width, sc_extent.height];
        let egui_screen = self.context.viewport_rect();
        let w_points = egui_screen.width().max(1.0);
        let scale_factor = sc_extent.width as f32 / w_points;

        // Peek the compositor signal value; only commit it after the submit
        // succeeds so a failed frame does not strand any engine waiter.
        let signal_value = self.render_target_pool.next_compositor_signal_value();
        let compositor_timeline = self.render_target_pool.compositor_timeline();
        let clear_color = self.clear_color;

        let result = self.root_mut().compositor.render_frame(
            clipped_primitives,
            textures_delta,
            scale_factor,
            screen_size,
            clear_color,
            Some((compositor_timeline, signal_value)),
        );

        // ── 9. Commit signal value / handle swapchain rebuild ──────────────
        match result {
            Ok(crate::compositor::FrameOutcome::Rendered) => {
                self.render_target_pool
                    .commit_compositor_signal(signal_value);
            }
            Ok(crate::compositor::FrameOutcome::RenderedNeedsRebuild) => {
                self.render_target_pool
                    .commit_compositor_signal(signal_value);
                self.root_mut().compositor.recreate_swapchain();
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                // Acquire failed before submit — no signal was emitted. Do
                // not commit; the engine's next target will still wait on
                // the previously-committed value.
                self.root_mut().compositor.recreate_swapchain();
            }
            Err(e) => {
                // Generic Vulkan error from submit/present (device loss,
                // out-of-memory, etc). Compositor already logged it;
                // attempt to rebuild the swapchain on next frame.
                log::error!("render_frame failed: {:?}", e);
            }
        }

        // ── 10. Process deferred viewports ─────────────────────────────────
        // Create newly-seen deferred viewports, run their ui_cb, composite,
        // and prune viewports that have disappeared from egui's output.
        // Immediate viewports were already handled during the main
        // `context.run(...)` via the installed renderer.
        self.process_deferred_viewports(event_loop, &viewport_output);

        // ── 11. Request redraw ─────────────────────────────────────────────
        // Always request redraw — the engine is continuously producing frames
        // and the UI needs to stay responsive.
        self.root().window.request_redraw();
    }

    /// Create / render / prune deferred viewports based on egui's
    /// `FullOutput::viewport_output` map. Called once per frame, after
    /// the ROOT compositor's render submit.
    ///
    /// - A non-root entry with `viewport_ui_cb = Some(cb)` means egui
    ///   wants us to continue driving this deferred viewport: if it's
    ///   new we build the window, then we invoke `cb` via a nested
    ///   `context.run(...)` to collect that viewport's draw data and
    ///   composite it to its own swapchain.
    /// - A non-root entry with `viewport_ui_cb = None` means an
    ///   Immediate viewport that was handled during the main run —
    ///   nothing extra to do (it already presented).
    /// - A non-root Viewport we own that is NOT listed in
    ///   `viewport_output` has been abandoned by egui (user stopped
    ///   calling `show_viewport_deferred`, or immediate viewport
    ///   skipped this frame): destroy it inline. The B9 step-7 pass
    ///   converts this to a fence-poll deferred-destruction queue.
    ///
    /// Known limitation (alpha): egui's managed-texture delta (font
    /// atlas etc.) is only emitted to the compositor whose
    /// `context.run` produced it, so deferred viewports can have an
    /// empty managed-texture table and text may not render. Same class
    /// of issue as user-textures in non-root viewports (decision #8).
    /// Full broadcast-delta fix is tracked in docs/known-limitations.md
    /// and scheduled for a post-alpha release.
    ///
    /// # Safety
    /// Vulkan device must be valid; called from the main event-loop
    /// thread while no other host method is running.
    unsafe fn process_deferred_viewports(
        &mut self,
        event_loop: &egui_winit::winit::event_loop::ActiveEventLoop,
        viewport_output: &std::collections::BTreeMap<egui::ViewportId, egui::ViewportOutput>,
    ) {
        // ---- Prune viewports that disappeared from egui's output ------
        let to_remove: Vec<egui::ViewportId> = self
            .viewports
            .keys()
            .copied()
            .filter(|id| *id != egui::ViewportId::ROOT && !viewport_output.contains_key(id))
            .collect();
        if !to_remove.is_empty() {
            // Inline destruction for the alpha; step 7 replaces this
            // with the fence-poll deferred-destruction queue (decision
            // #3) so we don't stall the main thread on abandoned
            // viewports. `device_wait_idle` is the safe hammer.
            self.vulkan
                .device
                .device_wait_idle()
                .expect("device_wait_idle failed");
            for id in to_remove {
                if let Some(mut vp) = self.viewports.remove(&id) {
                    let wid = vp.window.id();
                    self.window_id_to_viewport.remove(&wid);
                    vp.destroy(&self.vulkan);
                }
            }
        }

        // ---- Render deferred viewports --------------------------------
        // Clone the id list up front so we can mutate `self.viewports`
        // inside the loop without holding a borrow.
        let deferred_ids: Vec<egui::ViewportId> = viewport_output
            .iter()
            .filter(|(id, out)| **id != egui::ViewportId::ROOT && out.viewport_ui_cb.is_some())
            .map(|(id, _)| *id)
            .collect();

        for id in deferred_ids {
            // Re-fetch the output entry inside the loop — cheap, and
            // avoids holding a borrow across mutating `self`.
            let (parent, builder, ui_cb) = {
                let out = viewport_output
                    .get(&id)
                    .expect("viewport_output invariant violated");
                (
                    out.parent,
                    out.builder.clone(),
                    out.viewport_ui_cb
                        .clone()
                        .expect("deferred_ids filter guarantees Some"),
                )
            };

            // Create on first encounter.
            if !self.viewports.contains_key(&id) {
                let child = crate::viewport::Viewport::new_child(
                    &self.vulkan,
                    event_loop,
                    id,
                    parent,
                    egui::ViewportClass::Deferred,
                    builder,
                    &self.context,
                    self.present_mode,
                    Some(ui_cb.clone()),
                );
                let window_id = child.window.id();
                self.viewports.insert(id, child);
                self.window_id_to_viewport.insert(window_id, id);
                // Sync to current engine image (B9 decision #2).
                self.sync_child_engine_viewport(id);
            }

            // Take this viewport's raw input.
            let raw_input = {
                let vp = self
                    .viewports
                    .get_mut(&id)
                    .expect("child viewport invariant violated");
                vp.state.take_egui_input(&vp.window)
            };

            // Run the stored UI callback through egui for this viewport.
            let full_output = self.context.run(raw_input, |ctx| ui_cb(ctx));
            let egui::FullOutput {
                platform_output,
                textures_delta,
                shapes,
                pixels_per_point,
                viewport_output: _,
            } = full_output;

            let clear_color = self.clear_color;
            let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
            let egui_screen = self.context.viewport_rect();

            let vp = self
                .viewports
                .get_mut(&id)
                .expect("child viewport invariant violated");
            vp.state.handle_platform_output(&vp.window, platform_output);

            let sc_extent = vp.compositor.swapchain_extent();
            let screen_size = [sc_extent.width, sc_extent.height];
            let w_points = egui_screen.width().max(1.0);
            let scale_factor = sc_extent.width as f32 / w_points;

            // Child viewports pass `None` for engine_signal (see
            // Compositor::render_frame docs).
            let result = vp.compositor.render_frame(
                clipped_primitives,
                textures_delta,
                scale_factor,
                screen_size,
                clear_color,
                None,
            );

            match result {
                Ok(crate::compositor::FrameOutcome::Rendered) => {}
                Ok(crate::compositor::FrameOutcome::RenderedNeedsRebuild) => {
                    vp.compositor.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    vp.compositor.recreate_swapchain();
                }
                Err(e) => {
                    log::error!("deferred viewport render_frame failed: {:?}", e);
                }
            }

            vp.window.request_redraw();
        }
    }

    /// Pending exit code requested by the UI closure via
    /// `EngineHandle::exit`. Returns `None` if no exit was requested this
    /// frame. Taking consumes the request.
    pub(crate) fn take_exit_request(&self) -> Option<std::process::ExitCode> {
        self.engine_handle.take_exit()
    }

    /// Forward a winit `DeviceEvent` to the engine thread. Best-effort —
    /// if the channel is closed (engine has exited) the event is dropped.
    pub(crate) fn handle_device_event(&self, event: egui_winit::winit::event::DeviceEvent) {
        let _ = self.event_tx.send(crate::event::EngineEvent::Device(event));
    }

    /// Forward an application lifecycle event to the engine thread.
    /// Best-effort — dropped if the channel is closed.
    pub(crate) fn handle_lifecycle_event(&self, event: crate::event::AppLifecycleEvent) {
        let _ = self
            .event_tx
            .send(crate::event::EngineEvent::Lifecycle(event));
    }

    /// Wire egui-winit's AccessKit adapter to the winit event-loop proxy.
    /// Must be called once after construction so screen readers see egui
    /// widgets. Without this, `egui-winit/accesskit` compiles but delivers
    /// nothing — the adapter defaults to `None`.
    ///
    /// Per B9 decision #6, AccessKit is wired on ROOT only for the alpha.
    #[cfg(feature = "accesskit")]
    pub(crate) fn init_accesskit(
        &mut self,
        event_loop: &egui_winit::winit::event_loop::ActiveEventLoop,
        proxy: egui_winit::winit::event_loop::EventLoopProxy<crate::run::UserEvent>,
    ) {
        let root = self.root_mut();
        root.state.init_accesskit(event_loop, &root.window, proxy);
    }

    /// Deliver an AccessKit action request (from winit's user_event channel)
    /// to the ROOT egui-winit state.
    #[cfg(feature = "accesskit")]
    pub(crate) fn handle_accesskit_event(&mut self, event: &egui_winit::accesskit_winit::Event) {
        use egui_winit::accesskit_winit::WindowEvent;
        if let WindowEvent::ActionRequested(request) = &event.window_event {
            self.root_mut()
                .state
                .on_accesskit_action_request(request.clone());
        }
    }

    /// The OS window id of the ROOT viewport. Used by [`crate::run`] to
    /// tell `CloseRequested` on ROOT (full-app exit) apart from
    /// `CloseRequested` on a pop-out viewport (close that viewport only).
    pub(crate) fn root_window_id(&self) -> WindowId {
        self.root().window.id()
    }

    /// Handle a winit window event, dispatched to the Viewport whose
    /// window matches `window_id`. Returns `true` if egui consumed the
    /// event. Events for unknown windows (e.g. stale events delivered
    /// after a viewport was destroyed) are silently ignored.
    pub(crate) fn handle_window_event(
        &mut self,
        window_id: WindowId,
        event: &egui_winit::winit::event::WindowEvent,
    ) -> bool {
        // Resolve target viewport. `copied()` releases the immutable
        // borrow of `window_id_to_viewport` so we can take `&mut self`
        // on `viewports` below.
        let Some(viewport_id) = self.window_id_to_viewport.get(&window_id).copied() else {
            return false;
        };

        // Track `Resized` and `ScaleFactorChanged` so the next swapchain
        // recreation picks up the real window size. Without this, a driver
        // that reports `currentExtent == u32::MAX` (seen on X11 + certain
        // Mesa configurations, and on Wayland when the client is responsible
        // for buffer sizing) would keep presenting at whatever fallback the
        // previous frame chose and any window resize would visibly rescale
        // a stale low-resolution framebuffer.
        use egui_winit::winit::event::WindowEvent as WE;
        match event {
            WE::Resized(new_size) => {
                let vp = self
                    .viewports
                    .get_mut(&viewport_id)
                    .expect("viewport map / window_id index out of sync");

                // Skip zero-sized resizes (e.g. minimize on X11/Windows).
                if new_size.width == 0 || new_size.height == 0 {
                    return vp.state.on_window_event(&vp.window, event).consumed;
                }

                let new_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                vp.compositor.set_fallback_extent(new_extent);

                // Only force a rebuild if the swapchain extent actually
                // changed. On drivers that report currentExtent == u32::MAX
                // the frame loop won't get ERROR_OUT_OF_DATE_KHR, so we
                // must rebuild here.
                let current = vp.compositor.swapchain_extent();
                if current.width != new_extent.width || current.height != new_extent.height {
                    // SAFETY: `recreate_swapchain` internally issues
                    // `device_wait_idle` before destroying any resources,
                    // so no in-flight command buffers reference the old
                    // framebuffers, image views, or swapchain at the point
                    // of destruction. Called from the single-threaded event
                    // loop, so no concurrent submits are in progress.
                    unsafe {
                        vp.compositor.recreate_swapchain();
                    }
                }
            }
            WE::ScaleFactorChanged { .. } => {
                // winit delivers a `Resized` right after this with the new
                // physical size, which is where we actually rebuild the
                // swapchain — nothing extra to do here.
            }
            _ => {}
        }

        let vp = self
            .viewports
            .get_mut(&viewport_id)
            .expect("viewport map / window_id index out of sync");
        let response = vp.state.on_window_event(&vp.window, event);
        if response.repaint {
            vp.window.request_redraw();
        }
        response.consumed
    }

    // ─────────────────────────────────────────────────────────────────────
    // Engine restart
    // ─────────────────────────────────────────────────────────────────────

    /// Restart the engine with a fresh instance.
    ///
    /// # Safety
    /// Caller must ensure the Vulkan device is valid.
    unsafe fn restart_engine(&mut self, engine: E) {
        // 1. Drop target sender to close the channel, signaling engine to stop.
        //    We replace it with a dummy that we'll overwrite below.
        let (dummy_tx, _dummy_rx) = mailbox::target_channel();
        let old_tx = std::mem::replace(&mut self.target_tx, dummy_tx);
        drop(old_tx);

        // 2. Wait for engine thread to exit.
        if let Some(join) = self.engine_join.take() {
            let _ = join.join();
        }

        // 3. device_wait_idle to ensure all GPU work is finished.
        self.vulkan
            .device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        // 4. Create fresh render targets, then destroy the old ones.
        // Creating first prevents the driver from reusing the old timeline semaphore handles,
        // which would cause the validation layer to report spurious signal-value errors on
        // the first engine submit after restart.
        let extent = self.root().compositor.swapchain_extent();
        let format = vk::Format::B8G8R8A8_UNORM;
        let new_pool = RenderTargetPool::new(
            &self.vulkan.instance,
            &self.vulkan.device,
            self.vulkan.physical_device,
            extent,
            format,
        );
        self.render_target_pool.destroy();
        self.render_target_pool = new_pool;

        // 5. Reset all compositors to black.
        self.broadcast_engine_viewport(None);

        // 6. Create new channels.
        let (target_tx, target_rx) = mailbox::target_channel();
        let (frame_tx, frame_rx) = mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::channel();

        // 7. New state exchange.
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // 8. New health state and engine handle. Reuse the existing
        // ImageRegistry clone so user-registered textures survive restart
        // — the compositor and the channel are unchanged.
        let health = EngineHealthState::new();
        let image_registry = self.engine_handle.image_registry.clone();
        let engine_handle = EngineHandle::new(Arc::clone(&health), image_registry.clone());

        // 9. Spawn new engine thread.
        let engine_ctx = EngineContext {
            device: self.vulkan.device.clone(),
            queue: self.vulkan.engine_queue,
            queue_family_index: self.vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
            queue_mutex: self.vulkan.queue_mutex.clone(),
            image_registry,
        };
        let engine_join = engine_thread::spawn_engine_thread(
            engine,
            engine_ctx,
            target_rx,
            frame_tx,
            event_rx,
            ui_state_reader,
            engine_state_writer,
            Arc::clone(&health),
        );

        // 10. Send first target (index 0).
        let first_target = self.render_target_pool.make_target(0);
        target_tx
            .send(first_target)
            .expect("failed to send initial render target");

        // 11. Update all Host fields.
        self.target_tx = target_tx;
        self.frame_rx = frame_rx;
        self.event_tx = event_tx;
        self.engine_join = Some(engine_join);
        self.ui_state_writer = ui_state_writer;
        self.engine_state_reader = engine_state_reader;
        self.health = health;
        self.engine_handle = engine_handle;
        self.engine_target_index = Some(0);
        self.composited_target_index = None;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Multi-viewport (B9) — immediate renderer install/uninstall + driver
    // ─────────────────────────────────────────────────────────────────────

    /// Wire up egui's multi-viewport support: install an
    /// immediate-viewport renderer pointing at this Host and disable
    /// egui's default `embed_viewports` short-circuit.
    ///
    /// Must be called exactly once per Host, AFTER it has been placed in
    /// a `Box` so the pointer we capture is stable, and BEFORE the first
    /// `frame()` call.
    ///
    /// # Safety
    /// - `ptr` must reference a `Host<E>` living inside a `Box` whose
    ///   heap address will not change while this renderer is installed.
    /// - The Host must outlive any call to the installed renderer.
    ///   [`Host::destroy`] explicitly uninstalls via
    ///   [`Host::uninstall_immediate_viewport_renderer`] before teardown.
    /// - Must run on the main (winit event-loop) thread, which is the
    ///   same thread that invokes `Context::show_viewport_immediate`.
    pub(crate) unsafe fn install_immediate_viewport_renderer(ptr: *mut Self) {
        // Tell egui to stop embedding — without this every
        // show_viewport_immediate / show_viewport_deferred is silently
        // coalesced into the calling context and our renderer is never
        // asked to produce a real window.
        (*ptr).context.set_embed_viewports(false);
        let host_ptr = HostPtr(ptr);
        // `set_immediate_viewport_renderer` is an associated function
        // (writes into a thread-local, not the Context instance) —
        // call via the type, not a method receiver.
        egui::Context::set_immediate_viewport_renderer(move |_ctx, iv| {
            // SAFETY: see `install_immediate_viewport_renderer`'s
            // contract. The closure is only invoked by
            // `Context::show_viewport_immediate`, which only fires
            // from inside `Context::run` on the main thread. Host
            // lives at the pinned heap address until `destroy()`
            // uninstalls this callback.
            unsafe {
                (*host_ptr.0).immediate_viewport_frame(iv);
            }
        });
    }

    /// Replace the installed immediate-viewport renderer with a no-op
    /// and re-enable `embed_viewports` so any stray invocation after
    /// this point does not dereference a to-be-dropped Host pointer.
    /// Called at the top of [`Host::destroy`].
    ///
    /// # Safety
    /// Must run on the main thread (the thread that owns the renderer
    /// thread-local). Safe to call multiple times.
    unsafe fn uninstall_immediate_viewport_renderer(&self) {
        self.context.set_embed_viewports(true);
        egui::Context::set_immediate_viewport_renderer(|_ctx, _iv| {});
    }

    /// Drive one immediate-viewport frame. Invoked from the closure
    /// installed by [`Self::install_immediate_viewport_renderer`] when
    /// user code calls `Context::show_viewport_immediate` inside
    /// `Context::run`. Creates the child viewport on first invocation
    /// for a given `ViewportId`, reuses it on subsequent calls, then
    /// drives the child's own egui cycle: take input → run UI →
    /// tessellate → composite → present.
    ///
    /// # Safety
    /// - Caller must be the thread that installed the renderer (main).
    /// - [`Host::active_event_loop`] must be non-null; it is set by
    ///   [`Host::frame`] for the duration of `context.run()` and
    ///   cleared afterwards.
    unsafe fn immediate_viewport_frame(&mut self, iv: egui::ImmediateViewport<'_>) {
        let egui::ImmediateViewport {
            ids,
            builder,
            mut viewport_ui_cb,
        } = iv;
        let id = ids.this;

        // On first encounter for this ViewportId, build the child
        // viewport (window + surface + compositor + state). Subsequent
        // calls reuse the existing entry. Builder deltas (size / title
        // changes) are left for a post-alpha pass.
        if !self.viewports.contains_key(&id) {
            let ev_ptr = self.active_event_loop.get();
            assert!(
                !ev_ptr.is_null(),
                "immediate_viewport_frame fired outside a Host::frame `context.run()` — \
                 Host::active_event_loop was not set by the frame driver",
            );
            // SAFETY: `active_event_loop` is set by `Host::frame` for
            // the duration of the enclosing `context.run(...)`, which is
            // the call we're in right now.
            let event_loop: &egui_winit::winit::event_loop::ActiveEventLoop = &*ev_ptr;

            let child = crate::viewport::Viewport::new_child(
                &self.vulkan,
                event_loop,
                id,
                ids.parent,
                egui::ViewportClass::Immediate,
                builder,
                &self.context,
                self.present_mode,
                None,
            );
            let window_id = child.window.id();
            self.viewports.insert(id, child);
            self.window_id_to_viewport.insert(window_id, id);
            // Point the new compositor at the current engine image so
            // the engine viewport renders in this window too (B9
            // decision #2).
            self.sync_child_engine_viewport(id);
        }

        // Take this viewport's pending winit input.
        let raw_input = {
            let vp = self
                .viewports
                .get_mut(&id)
                .expect("child viewport invariant violated");
            vp.state.take_egui_input(&vp.window)
        };

        // Run egui for this viewport. The caller's ui callback in
        // ImmediateViewport.viewport_ui_cb receives `&egui::Context` and
        // draws panels/windows against it — same shape as the root UI
        // closure.
        let full_output = self.context.run(raw_input, |ctx| viewport_ui_cb(ctx));

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output: _,
        } = full_output;

        // Extract cross-field reads before taking &mut vp so borrow-
        // checker is happy. `clear_color` and `context.tessellate` /
        // `viewport_rect` can all run with only immutable borrows of
        // `self` and standalone values; we then re-borrow viewports
        // mutably for the render call.
        let clear_color = self.clear_color;
        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let egui_screen = self.context.viewport_rect();

        let vp = self
            .viewports
            .get_mut(&id)
            .expect("child viewport invariant violated");
        vp.state.handle_platform_output(&vp.window, platform_output);

        let sc_extent = vp.compositor.swapchain_extent();
        let screen_size = [sc_extent.width, sc_extent.height];
        let w_points = egui_screen.width().max(1.0);
        let scale_factor = sc_extent.width as f32 / w_points;

        // Child viewports MUST pass `None` for engine_signal — only
        // ROOT signals the engine-handshake timeline (see Compositor
        // docs + decision #2 above).
        let result = vp.compositor.render_frame(
            clipped_primitives,
            textures_delta,
            scale_factor,
            screen_size,
            clear_color,
            None,
        );

        match result {
            Ok(crate::compositor::FrameOutcome::Rendered) => {}
            Ok(crate::compositor::FrameOutcome::RenderedNeedsRebuild) => {
                vp.compositor.recreate_swapchain();
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                vp.compositor.recreate_swapchain();
            }
            Err(e) => {
                log::error!("child viewport render_frame failed: {:?}", e);
            }
        }

        vp.window.request_redraw();
    }

    /// Destroy all host resources. Must be called before dropping.
    ///
    /// # Safety
    /// Caller must ensure the Vulkan device is valid and no GPU work is in flight.
    pub(crate) unsafe fn destroy(&mut self) {
        // Uninstall the immediate-viewport renderer FIRST. Any stray
        // invocation after this point (e.g. during `storage.flush` or
        // engine-thread teardown paths that might run egui code) hits
        // the no-op rather than dereferencing a to-be-invalid Host
        // pointer. See `install_immediate_viewport_renderer`'s safety
        // contract.
        self.uninstall_immediate_viewport_renderer();

        // Save persisted state BEFORE tearing down. Both window geometry
        // (from the live winit window) and egui memory (from the context)
        // need to be captured while their sources are still valid.
        #[cfg(feature = "persistence")]
        {
            if self.persistent_windows {
                let zoom = self.context.zoom_factor();
                let settings = egui_winit::WindowSettings::from_window(zoom, &self.root().window);
                let mut map = std::collections::HashMap::new();
                map.insert(egui::ViewportId::ROOT, settings);
                self.storage.set_windows(&map);
            }
            if self.persistent_egui_memory {
                let mem = self.context.memory(|m| m.clone());
                self.storage.set_egui_memory(&mem);
            }
            // Blocking flush — the Drop impl on InnerStorage waits on the
            // save thread, but being explicit here guarantees disk-visible
            // persistence before the event loop exits.
            self.storage.flush();
        }

        // Signal engine to shut down.
        let _ = self.event_tx.send(crate::event::EngineEvent::Shutdown);

        // Close the target channel so the engine thread exits its recv loop.
        let (dummy_tx, _dummy_rx) = mailbox::target_channel();
        let old_tx = std::mem::replace(&mut self.target_tx, dummy_tx);
        drop(old_tx);

        // Wait for engine thread.
        if let Some(join) = self.engine_join.take() {
            let _ = join.join();
        }

        // Decision #4: single device_wait_idle up front, then reverse-
        // insertion-order destroy across all viewports. At this step only
        // ROOT exists so the loop is trivial; later steps (§B9 step 7)
        // add non-root entries that must be destroyed before ROOT.
        self.vulkan
            .device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        self.render_target_pool.destroy();

        // Destroy non-root viewports first, root last.
        let root = self.viewports.remove(&egui::ViewportId::ROOT);
        for (_, mut vp) in self.viewports.drain() {
            vp.destroy(&self.vulkan);
        }
        if let Some(mut root) = root {
            root.destroy(&self.vulkan);
        }
        self.window_id_to_viewport.clear();
    }
}
