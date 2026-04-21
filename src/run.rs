use egui_winit::winit::{
    self,
    application::ApplicationHandler,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowAttributes,
};
use std::{process::ExitCode, time::Duration};

use crate::engine::EngineRenderer;
use crate::host::{EngineHandle, Host};
use crate::storage::Storage;
use crate::types::{EngineStatus, RunOption, VulkanContext};

/// User-event type sent through winit's event-loop proxy.
///
/// A struct with cfg-gated fields — mirrors the v1 `IntegrationEvent` pattern
/// so that the `EventLoop` user-event generic stays `UserEvent` regardless of
/// feature configuration. When no feature contributes fields, this is an
/// empty ZST — no events can be sent, which is correct for that config.
#[derive(Debug)]
pub struct UserEvent {
    #[cfg(feature = "accesskit")]
    pub accesskit: egui_winit::accesskit_winit::Event,
}

#[cfg(feature = "accesskit")]
impl From<egui_winit::accesskit_winit::Event> for UserEvent {
    fn from(event: egui_winit::accesskit_winit::Event) -> Self {
        Self { accesskit: event }
    }
}

/// egui-ash v2 entry point.
///
/// Drives the winit event loop. Consumes `vulkan` and `engine`; returns the
/// `ExitCode` produced when the window is closed.
///
/// `app_id` identifies the application for on-disk persistence (window
/// geometry and egui memory) when the `persistence` feature is enabled and
/// the corresponding `RunOption` flags are set.
///
/// # Panics
///
/// Panics if the winit event loop cannot be created or fails to run, if the
/// window cannot be created, or if `Host::new` asserts on a mismatched queue
/// family (see `VulkanContext` docs for the alpha restriction).
pub fn run<E: EngineRenderer>(
    app_id: impl Into<String>,
    vulkan: VulkanContext,
    engine: E,
    options: RunOption,
    ui: impl FnMut(
            &egui::Context,
            &EngineStatus,
            &mut E::UiState,
            &E::EngineState,
            &EngineHandle<E>,
            &mut Storage,
        ) + 'static,
) -> ExitCode {
    let app_id: String = app_id.into();
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");
    let event_loop_proxy = event_loop.create_proxy();
    let (exit_tx, exit_rx) = std::sync::mpsc::channel();

    let mut state = AppState {
        app_id,
        vulkan: Some(vulkan),
        options,
        engine: Some(engine),
        ui,
        host: None,
        exit_tx,
        event_loop_proxy,
    };

    event_loop
        .run_app(&mut state)
        .expect("Failed to run event loop");

    exit_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap_or(ExitCode::FAILURE)
}

struct AppState<E: EngineRenderer, F> {
    app_id: String,
    vulkan: Option<VulkanContext>,
    options: RunOption,
    engine: Option<E>,
    ui: F,
    host: Option<Host<E>>,
    exit_tx: std::sync::mpsc::Sender<ExitCode>,
    /// Held so `init_accesskit` can hand a proxy clone to egui-winit's
    /// adapter when the `accesskit` feature is enabled. Unused without
    /// the feature — empty `UserEvent` means no proxy events to deliver.
    #[cfg_attr(not(feature = "accesskit"), allow(dead_code))]
    event_loop_proxy: winit::event_loop::EventLoopProxy<UserEvent>,
}

impl<E, F> ApplicationHandler<UserEvent> for AppState<E, F>
where
    E: EngineRenderer,
    F: FnMut(
            &egui::Context,
            &EngineStatus,
            &mut E::UiState,
            &E::EngineState,
            &EngineHandle<E>,
            &mut Storage,
        ) + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Second (and later) Resumed events — forward as a lifecycle event
        // and return. Mobile platforms emit Resumed after Suspended; desktop
        // typically fires Resumed exactly once. Surface recreation for full
        // mobile support is not yet implemented.
        if self.host.is_some() {
            if let Some(host) = self.host.as_ref() {
                host.handle_lifecycle_event(crate::event::AppLifecycleEvent::Resumed);
            }
            return;
        }

        // -- Construct Storage (always). With the `persistence` feature
        // enabled it opens (or creates) the on-disk RON store for this
        // app_id; without the feature it's a zero-sized stub. Passed to
        // Host so the UI closure can `set_value`/`get_value` freely.
        let storage = Storage::initialize(&self.app_id);

        // -- Assemble a ViewportBuilder: start from user's if supplied, else
        // a minimal default; then overlay any persisted window settings.
        // `mut` is only used under the `persistence` feature (to apply
        // saved settings); harmless elsewhere.
        #[allow(unused_mut)]
        let mut vb = self
            .options
            .viewport_builder
            .clone()
            .unwrap_or_else(|| egui::ViewportBuilder::default().with_title("egui-ash"));

        #[cfg(feature = "persistence")]
        if self.options.persistent_windows {
            if let Some(windows) = storage.get_windows() {
                if let Some(settings) = windows.get(&egui::ViewportId::ROOT) {
                    // zoom_factor defaults to 1.0 pre-first-frame; egui
                    // applies the real zoom after setup runs. This is
                    // consistent with egui-winit's own startup path.
                    vb = settings.initialize_viewport_builder(1.0, event_loop, vb);
                }
            }
        }

        // -- Extract ViewportBuilder → WindowAttributes. Cover every field
        // winit can honour at creation time (v1 only extracted a subset,
        // which silently dropped persisted position/maximized/fullscreen).
        let mut window_attributes = WindowAttributes::default().with_visible(true);
        if let Some(title) = &vb.title {
            window_attributes = window_attributes.with_title(title.clone());
        }
        if let Some(pos) = vb.position {
            window_attributes = window_attributes
                .with_position(winit::dpi::LogicalPosition::new(pos.x as f64, pos.y as f64));
        }
        if let Some(size) = vb.inner_size {
            window_attributes = window_attributes
                .with_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(size) = vb.min_inner_size {
            window_attributes = window_attributes
                .with_min_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(size) = vb.max_inner_size {
            window_attributes = window_attributes
                .with_max_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(decorations) = vb.decorations {
            window_attributes = window_attributes.with_decorations(decorations);
        }
        if let Some(resizable) = vb.resizable {
            window_attributes = window_attributes.with_resizable(resizable);
        }
        if let Some(transparent) = vb.transparent {
            window_attributes = window_attributes.with_transparent(transparent);
        }
        if let Some(maximized) = vb.maximized {
            window_attributes = window_attributes.with_maximized(maximized);
        }
        if let Some(fullscreen) = vb.fullscreen {
            if fullscreen {
                window_attributes = window_attributes
                    .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
        }

        let window = event_loop.create_window(window_attributes).unwrap();

        // -- macOS post-creation fixup for outer position: winit's
        // WindowAttributes::with_position only takes effect on some platforms.
        // WindowSettings::initialize_window handles the macOS edge case by
        // calling set_outer_position after the window exists.
        #[cfg(feature = "persistence")]
        if self.options.persistent_windows {
            if let Some(windows) = storage.get_windows() {
                if let Some(settings) = windows.get(&egui::ViewportId::ROOT) {
                    settings.initialize_window(&window);
                }
            }
        }

        let vulkan = self.vulkan.take().expect("VulkanContext already consumed");
        let engine = self.engine.take().expect("Engine already consumed");

        // SAFETY: winit delivers `resumed` on the main thread, which is
        // where the event loop runs — so the Vulkan entry, instance, device,
        // and window handles in `vulkan` are accessed from the thread that
        // created them. `vulkan` was just taken from `self.vulkan` and is a
        // single owned value that outlives `Host` via the host's lifetime
        // contract. `Host::drop` is not relied on — `destroy()` is invoked
        // explicitly on `CloseRequested` before the event loop exits.
        let host = unsafe {
            Host::new(
                vulkan,
                engine,
                &self.options,
                window,
                event_loop,
                self.exit_tx.clone(),
                storage,
            )
        };
        self.host = Some(host);

        // Wire AccessKit adapter so egui widgets reach screen readers via
        // the user_event channel. Must run after Host is stored because the
        // adapter holds a reference to the window.
        #[cfg(feature = "accesskit")]
        if let Some(host) = self.host.as_mut() {
            host.init_accesskit(event_loop, self.event_loop_proxy.clone());
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(host) = self.host.as_mut() else {
            return;
        };

        if matches!(event, winit::event::WindowEvent::CloseRequested) {
            // ROOT close exits the app; non-root close targets that one
            // viewport only — handled inside Host as part of the B9
            // multi-viewport lifecycle (step 7 deferred-destruction queue).
            if window_id == host.root_window_id() {
                // SAFETY: `destroy()` tears down all Vulkan resources and
                // must be called exactly once before the Host is dropped
                // or reused. We invoke it here on CloseRequested (the
                // terminal event) and immediately set `self.host = None`
                // on the next line so no further method is called on the
                // torn-down Host — winit may still deliver one more
                // `about_to_wait` after `exit()`, and the `None` guard
                // in `about_to_wait` bails out in that case.
                unsafe {
                    host.destroy();
                }
                self.host = None;
                self.exit_tx.send(ExitCode::SUCCESS).ok();
                event_loop.exit();
                return;
            }
            // Non-root CloseRequested: routed to Host for per-viewport
            // teardown. At this step no non-root viewports exist yet, so
            // the handler is a no-op. Wired properly in B9 step 7.
            host.handle_window_event(window_id, &event);
            return;
        }

        host.handle_window_event(window_id, &event);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let Some(host) = self.host.as_ref() else {
            return;
        };
        host.handle_device_event(event);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(host) = self.host.as_ref() {
            host.handle_lifecycle_event(crate::event::AppLifecycleEvent::Suspended);
        }
    }

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(host) = self.host.as_ref() {
            host.handle_lifecycle_event(crate::event::AppLifecycleEvent::MemoryWarning);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(host) = self.host.as_ref() {
            host.handle_lifecycle_event(crate::event::AppLifecycleEvent::LoopExiting);
        }
    }

    #[cfg(feature = "accesskit")]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        let Some(host) = self.host.as_mut() else {
            return;
        };
        host.handle_accesskit_event(&event.accesskit);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(host) = self.host.as_mut() else {
            return;
        };
        // SAFETY: `host` is live (not yet destroyed) — the `Some` branch
        // above rules out the post-`CloseRequested` window where we've
        // already called `host.destroy()` and set `self.host = None`.
        // `frame()` wraps unsafe Vulkan submit/present calls; called from
        // the event-loop thread, single-threaded per frame.
        unsafe {
            host.frame(&mut self.ui);
        }

        // Handle programmatic exit requested via `EngineHandle::exit` during
        // the UI closure. Drives the same teardown path as `CloseRequested`.
        if let Some(code) = host.take_exit_request() {
            // SAFETY: same invariants as the `CloseRequested` destroy site —
            // host is live, destroy is called exactly once, self.host is
            // cleared before the next about_to_wait tick so no further
            // methods run on the torn-down Host.
            unsafe {
                host.destroy();
            }
            self.host = None;
            self.exit_tx.send(code).ok();
            event_loop.exit();
        }
    }
}
