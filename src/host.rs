use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use ash::vk;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

use crate::compositor::Compositor;
use crate::engine::EngineRenderer;
use crate::engine_thread::{
    self, EngineHealthState, HEALTH_CRASHED, HEALTH_RUNNING, HEALTH_STARTING, HEALTH_STOPPED,
};
use crate::mailbox::{self, MailboxReceiver};
use crate::render_targets::RenderTargetPool;
use crate::state_exchange::{self, StateReader, StateWriter};
use crate::types::{
    EngineContext, EngineHealth, EngineRestartError, EngineStatus, RunOption, VulkanContext,
};

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
    health: Arc<EngineHealthState>,
}

impl<E: EngineRenderer> EngineHandle<E> {
    fn new(health: Arc<EngineHealthState>) -> Self {
        Self {
            restart_request: std::cell::RefCell::new(None),
            health,
        }
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

    fn take_restart(&self) -> Option<E> {
        self.restart_request.borrow_mut().take()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host — internal orchestrator
// ─────────────────────────────────────────────────────────────────────────────

/// Host-side orchestrator. Owns all host resources and implements the
/// per-frame protocol: poll engine frames, run egui, composite, present.
pub(crate) struct Host<E: EngineRenderer> {
    // Vulkan context (owned)
    vulkan: VulkanContext,

    // Window + egui
    window: egui_winit::winit::window::Window,
    context: egui::Context,
    egui_winit_state: egui_winit::State,

    // Compositor (swapchain + egui rendering)
    compositor: Compositor,

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

    // Options
    clear_color: [f32; 4],

    // Surface (for cleanup)
    surface: vk::SurfaceKHR,

    // Exit channel
    _exit_tx: mpsc::Sender<std::process::ExitCode>,
}

impl<E: EngineRenderer> Host<E> {
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
    ) -> Self {
        // Create surface
        let surface = ash_window::create_surface(
            &vulkan.entry,
            &vulkan.instance,
            window
                .display_handle()
                .expect("failed to get display handle")
                .as_raw(),
            window
                .window_handle()
                .expect("failed to get window handle")
                .as_raw(),
            None,
        )
        .expect("failed to create surface");

        // Create compositor
        let compositor = Compositor::new(
            &vulkan.entry,
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            vulkan.host_queue,
            vulkan.host_queue_family_index,
            surface,
            options.present_mode,
        );

        // Create render target pool
        let extent = compositor.swapchain_extent();
        let format = vk::Format::B8G8R8A8_UNORM;
        let render_target_pool = RenderTargetPool::new(
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            extent,
            format,
            vulkan.host_queue_family_index,
            vulkan.engine_queue_family_index,
        );

        // Create channels
        let (target_tx, target_rx) = mailbox::target_channel();
        let (frame_tx, frame_rx) = mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::channel();

        // State exchange
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // Health
        let health = EngineHealthState::new();
        let engine_handle = EngineHandle::new(Arc::clone(&health));

        // Engine context
        let engine_ctx = EngineContext {
            device: vulkan.device.clone(),
            queue: vulkan.engine_queue,
            queue_family_index: vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
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

        // Create egui context and winit state
        let context = egui::Context::default();
        let theme = if options.follow_system_theme {
            window.theme().or(Some(options.default_theme))
        } else {
            Some(options.default_theme)
        };

        let egui_winit_state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            theme,
            None,
        );

        Self {
            vulkan,
            window,
            context,
            egui_winit_state,
            compositor,
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
            clear_color: options.clear_color,
            surface,
            _exit_tx: exit_tx,
        }
    }

    /// Run a single host frame: poll engine, run egui UI, composite, present.
    ///
    /// # Safety
    /// Caller must ensure the Vulkan device and all handles are valid.
    pub(crate) unsafe fn frame(
        &mut self,
        ui: &mut impl FnMut(
            &egui::Context,
            &EngineStatus,
            &mut E::UiState,
            &E::EngineState,
            &EngineHandle<E>,
        ),
    ) {
        // ── 1. Check for new engine frame ──────────────────────────────────
        if let Some(completed) = self.frame_rx.try_recv() {
            // CPU-wait on the timeline semaphore for the engine's signal_value.
            // This ensures the GPU has finished rendering into this target.
            let semaphores = [self.render_target_pool.timeline()];
            let values = [completed.signal_value];
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            self.vulkan
                .device
                .wait_semaphores(&wait_info, u64::MAX)
                .expect("wait_semaphores failed");

            // Figure out which target index the engine just finished.
            // The completed frame's image tells us which pool image it used.
            let finished_index = if completed.image == self.render_target_pool.image(0) {
                0
            } else {
                1
            };

            // Update compositor to sample this image
            self.compositor
                .set_engine_viewport(self.render_target_pool.image_view(finished_index));

            // Reclaim the previously composited target (if any) and send it back
            // to the engine for reuse.
            if let Some(prev_index) = self.composited_target_index {
                let target = self.render_target_pool.make_target(prev_index);
                // If send fails, the engine thread has exited — that's fine.
                let _ = self.target_tx.send(target);
            }

            // Track: the newly composited target is the one the engine just finished.
            // The engine target index is now "none" (it will get a new one when we
            // reclaim the current composited target on the next frame).
            self.composited_target_index = Some(finished_index);
            self.engine_target_index = None;
        }

        // ── 2. If engine crashed and nothing composited, show black ────────
        let health_val = self.health.health.load(Ordering::Acquire);
        if (health_val == HEALTH_CRASHED || health_val == HEALTH_STOPPED)
            && self.composited_target_index.is_none()
        {
            self.compositor.set_engine_viewport_black();
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
            viewport_texture_id: self.compositor.engine_viewport_texture_id(),
            frames_delivered,
            last_frame_time,
        };

        // ── 4. Read latest engine state ────────────────────────────────────
        let engine_state = self.engine_state_reader.read();

        // ── 5. Run egui frame ──────────────────────────────────────────────
        let raw_input = self.egui_winit_state.take_egui_input(&self.window);

        let mut ui_state = E::UiState::default();
        let full_output = self.context.run(raw_input, |ctx| {
            ui(ctx, &engine_status, &mut ui_state, &engine_state, &self.engine_handle);
        });

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output: _,
        } = full_output;

        self.egui_winit_state
            .handle_platform_output(&self.window, platform_output);

        // ── 6. Publish UI state ────────────────────────────────────────────
        self.ui_state_writer.publish(ui_state);

        // ── 7. Handle restart request ──────────────────────────────────────
        if let Some(new_engine) = self.engine_handle.take_restart() {
            self.restart_engine(new_engine);
        }

        // ── 8. Tessellate and composite ────────────────────────────────────
        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let size = self.window.inner_size();
        let screen_size = [size.width, size.height];
        let scale_factor = self.window.scale_factor() as f32 * self.context.zoom_factor();

        // Advance the shared timeline so the compositor signals a value
        // that the engine can later wait on.
        let signal_value = self.render_target_pool.next_signal_value();

        let result = self.compositor.render_frame(
            clipped_primitives,
            textures_delta,
            scale_factor,
            screen_size,
            self.clear_color,
            self.render_target_pool.timeline(),
            signal_value,
        );

        // ── 9. Handle swapchain out of date ────────────────────────────────
        if let Err(vk::Result::ERROR_OUT_OF_DATE_KHR) = result {
            self.compositor.recreate_swapchain();
        }

        // ── 10. Request redraw ─────────────────────────────────────────────
        // Always request redraw — the engine is continuously producing frames
        // and the UI needs to stay responsive.
        self.window.request_redraw();
    }

    /// Handle a winit window event. Returns `true` if egui consumed the event.
    pub(crate) fn handle_window_event(
        &mut self,
        event: &egui_winit::winit::event::WindowEvent,
    ) -> bool {
        let response = self
            .egui_winit_state
            .on_window_event(&self.window, event);
        if response.repaint {
            self.window.request_redraw();
        }
        response.consumed
    }

    /// Signal that the application should exit.
    pub(crate) fn request_exit(&mut self) {
        // Send shutdown event to engine
        let _ = self.event_tx.send(crate::event::EngineEvent::Shutdown);
    }

    /// Borrow the host window.
    pub(crate) fn window(&self) -> &egui_winit::winit::window::Window {
        &self.window
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

        // 4. Destroy old render targets, create fresh ones.
        self.render_target_pool.destroy();

        let extent = self.compositor.swapchain_extent();
        let format = vk::Format::B8G8R8A8_UNORM;
        self.render_target_pool = RenderTargetPool::new(
            &self.vulkan.instance,
            &self.vulkan.device,
            self.vulkan.physical_device,
            extent,
            format,
            self.vulkan.host_queue_family_index,
            self.vulkan.engine_queue_family_index,
        );

        // 5. Reset compositor to black.
        self.compositor.set_engine_viewport_black();

        // 6. Create new channels.
        let (target_tx, target_rx) = mailbox::target_channel();
        let (frame_tx, frame_rx) = mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::channel();

        // 7. New state exchange.
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // 8. New health state and engine handle.
        let health = EngineHealthState::new();
        let engine_handle = EngineHandle::new(Arc::clone(&health));

        // 9. Spawn new engine thread.
        let engine_ctx = EngineContext {
            device: self.vulkan.device.clone(),
            queue: self.vulkan.engine_queue,
            queue_family_index: self.vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
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

    /// Destroy all host resources. Must be called before dropping.
    ///
    /// # Safety
    /// Caller must ensure the Vulkan device is valid and no GPU work is in flight.
    pub(crate) unsafe fn destroy(&mut self) {
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

        // Wait for GPU idle.
        self.vulkan
            .device
            .device_wait_idle()
            .expect("device_wait_idle failed");

        // Destroy in reverse creation order.
        self.render_target_pool.destroy();
        self.compositor.destroy();

        // Destroy surface.
        let surface_loader =
            ash::khr::surface::Instance::new(&self.vulkan.entry, &self.vulkan.instance);
        surface_loader.destroy_surface(self.surface, None);
    }
}
