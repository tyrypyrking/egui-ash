use egui_winit::winit::{
    self,
    event_loop::{ActiveEventLoop, EventLoopBuilder},
};
use raw_window_handle::HasDisplayHandle as _;
use std::{
    ffi::CStr,
    mem::ManuallyDrop,
    process::ExitCode,
    sync::{Arc, Mutex},
};

use crate::{
    app::{App, AppCreator, CreationContext},
    event,
    integration::Integration,
    renderer::ImageRegistry,
    Allocator, Theme,
};
#[cfg(feature = "persistence")]
use crate::{storage, utils};

/// egui-ash run option.
pub struct RunOption {
    /// window clear color.
    pub clear_color: [f32; 4],
    /// viewport builder for root window.
    pub viewport_builder: Option<egui::ViewportBuilder>,
    /// follow system theme.
    pub follow_system_theme: bool,
    /// default theme.
    pub default_theme: Theme,
    #[cfg(feature = "persistence")]
    pub persistent_windows: bool,
    #[cfg(feature = "persistence")]
    pub persistent_egui_memory: bool,
    /// vk::PresentModeKHR
    pub present_mode: ash::vk::PresentModeKHR,
}
impl Default for RunOption {
    fn default() -> Self {
        Self {
            clear_color: [0.0, 0.0, 0.0, 1.0],
            viewport_builder: None,
            follow_system_theme: true,
            default_theme: Theme::Light,
            #[cfg(feature = "persistence")]
            persistent_windows: true,
            #[cfg(feature = "persistence")]
            persistent_egui_memory: true,
            present_mode: ash::vk::PresentModeKHR::FIFO,
        }
    }
}

/// exit signal sender for exit app.
#[derive(Debug, Clone)]
pub struct ExitSignal {
    tx: std::sync::mpsc::Sender<ExitCode>,
}
impl ExitSignal {
    /// send exit signal.
    pub fn send(&self, exit_code: ExitCode) {
        self.tx.send(exit_code).unwrap();
    }
}

///egui-ash run function.
///
/// ```
/// fn main() {
///     egui_winit_ash::run("my_app", MyAppCreator, RunOption::default());
/// }
/// ```
pub fn run<C: AppCreator<A> + 'static, A: Allocator + 'static>(
    app_id: impl Into<String>,
    creator: C,
    run_option: RunOption,
) -> ExitCode {
    let app_id = app_id.into();

    let event_loop = EventLoopBuilder::default()
        .build()
        .expect("Failed to create event loop");

    let (exit_signal_tx, exit_signal_rx) = std::sync::mpsc::channel();
    let exit_signal = ExitSignal { tx: exit_signal_tx };

    let exit_code = Arc::new(Mutex::new(ExitCode::SUCCESS));
    let mut app_handler = AppHandler {
        app_id,
        state: AppState::Starting { creator },
        run_option,
        exit_signal,
        exit_signal_rx,
        exit_code: exit_code.clone(),
        _allocator_marker: std::marker::PhantomData,
    };
    event_loop
        .run_app(&mut app_handler)
        .expect("Failed to run event loop");
    let code = *exit_code.lock().unwrap();
    code
}

/// Enum to track the state of the app and variables specific to each state.
enum AppState<C: AppCreator<A>, A: Allocator + 'static> {
    Starting {
        creator: C,
    },
    Running {
        app: C::App,
        integration: ManuallyDrop<Integration<A>>,
    },
}

/// The top-level application manager for an `egui-ash` managed application.
struct AppHandler<C: AppCreator<A>, A: Allocator + 'static> {
    app_id: String,
    state: AppState<C, A>,
    run_option: RunOption,
    exit_signal: ExitSignal,
    exit_signal_rx: std::sync::mpsc::Receiver<ExitCode>,
    exit_code: Arc<Mutex<ExitCode>>,
    _allocator_marker: std::marker::PhantomData<A>,
}

impl<A: Allocator + 'static, C: AppCreator<A>> winit::application::ApplicationHandler
    for AppHandler<C, A>
{
    /// Initialize the application, including the main window, if not already running.
    /// Otherwise, signal that the application is being resumed.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let creator = match &mut self.state {
            AppState::Starting { creator } => creator,
            AppState::Running { app, integration } => {
                log::info!("App resuming...");
                let app_event = event::Event::AppEvent {
                    event: event::AppEvent::Resumed,
                };
                app.handle_event(app_event);
                integration.paint_all(event_loop, app);
                log::info!("App resumed");
                return;
            }
        };
        log::info!("App starting...");

        #[cfg(feature = "persistence")]
        let storage =
            storage::Storage::from_app_id(&self.app_id).expect("Failed to create storage");

        let context = egui::Context::default();
        #[cfg(feature = "persistence")]
        if self.run_option.persistent_egui_memory {
            if let Some(memory) = storage.get_egui_memory() {
                context.memory_mut(|m| *m = memory);
            }
        }

        context.set_embed_viewports(false);
        match self.run_option.default_theme {
            Theme::Light => {
                context.set_visuals(egui::Visuals::light());
            }
            Theme::Dark => {
                context.set_visuals(egui::Visuals::dark());
            }
        }

        #[allow(unused_mut)] // only mutable when persistence feature is enabled
        let main_window = if let Some(mut viewport_builder) =
            self.run_option.viewport_builder.clone()
        {
            #[cfg(feature = "persistence")]
            if self.run_option.persistent_windows {
                let egui_zoom_factor = context.zoom_factor();
                let window_settings = storage
                    .get_windows()
                    .and_then(|windows| windows.get(&egui::ViewportId::ROOT).map(|s| s.to_owned()))
                    .map(|mut settings| {
                        settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                            egui_zoom_factor,
                            &event_loop,
                        ));
                        settings.clamp_position_to_monitors(egui_zoom_factor, &event_loop);
                        settings.to_owned()
                    });

                if let Some(window_settings) = window_settings {
                    viewport_builder = window_settings.initialize_viewport_builder(
                        egui_zoom_factor,
                        &event_loop,
                        viewport_builder,
                    );
                }
            }

            egui_winit::create_window(&context, &event_loop, &viewport_builder.with_visible(false))
                .unwrap()
        } else {
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("egui-ash")
                        .with_visible(false),
                )
                .unwrap()
        };

        let instance_extensions = ash_window::enumerate_required_extensions(
            event_loop
                .display_handle()
                .expect("Unable to retrieve a display handle")
                .as_raw(),
        )
        .unwrap();
        let instance_extensions = instance_extensions
            .into_iter()
            .map(|&ext| unsafe { CStr::from_ptr(ext).to_owned() })
            .collect::<Vec<_>>();
        let device_extensions = [ash::khr::swapchain::NAME.to_owned()];

        let (image_registry, image_registry_receiver) = ImageRegistry::new();

        let cc: CreationContext<'_> = CreationContext {
            main_window: &main_window,
            context: context.clone(),
            required_instance_extensions: instance_extensions,
            required_device_extensions: device_extensions.into_iter().collect(),
            image_registry,
            exit_signal: self.exit_signal.clone(),
        };
        let (app, render_state) = creator.create(cc);

        // `ManuallyDrop` is required because the integration object needs to be dropped before
        // the app drops for `gpu_allocator` drop order reasons.
        // TODO: Try to remove this requirement.
        let integration = ManuallyDrop::new(Integration::new(
            &self.app_id,
            &event_loop,
            context,
            main_window,
            render_state,
            self.run_option.present_mode,
            image_registry_receiver,
            #[cfg(feature = "persistence")]
            storage,
            #[cfg(feature = "persistence")]
            self.run_option.persistent_windows,
            #[cfg(feature = "persistence")]
            self.run_option.persistent_egui_memory,
        ));

        self.state = AppState::Running { app, integration };
        log::info!("App started");
    }

    /// Forward user events to the application.
    #[cfg(feature = "accesskit")]
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: T) {
        let AppState::Running { app, integration } = &mut self.state else {
            log::warn!("User event ${event:?} received before app started");
            return;
        };

        integration.handle_accesskit_event(
            &integration_event.accesskit,
            event_loop,
            control_flow,
            &mut app,
        );
        let user_event = event::Event::AccessKitActionRequest(integration_event.accesskit);
        app.handle_event(user_event);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        if let Some(code) = self.exit_signal_rx.try_recv().ok() {
            *self.exit_code.lock().unwrap() = code;
            event_loop.exit();
            return;
        }

        let AppState::Running { app, integration } = &mut self.state else {
            log::warn!("Window event ${event:?} received before app started");
            return;
        };
        let consumed = integration.handle_window_event(
            window_id,
            &event,
            &event_loop,
            self.run_option.follow_system_theme,
            app,
        );
        if consumed {
            return;
        }

        let Some(viewport_id) = integration.viewport_id_from_window_id(window_id) else {
            return;
        };
        let viewport_event = event::Event::ViewportEvent { viewport_id, event };
        app.handle_event(viewport_event);
    }

    /// Forward device events to the application.
    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let AppState::Running { app, .. } = &mut self.state else {
            log::warn!("Device event ${event:?} received before app started");
            return;
        };

        let device_event = event::Event::DeviceEvent { device_id, event };
        app.handle_event(device_event);
    }

    /// Forward the about to wait event to the application.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let AppState::Running { app, integration } = &mut self.state else {
            log::warn!("About To Wait signal received before app started");
            return;
        };

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::AboutToWait,
        };
        app.handle_event(app_event);
        integration.paint_all(event_loop, app);
    }

    /// Forward the suspended event to the application.
    fn suspended(&mut self, _: &ActiveEventLoop) {
        let AppState::Running { app, .. } = &mut self.state else {
            log::warn!("Suspended signal received before app started");
            return;
        };

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::Suspended,
        };
        app.handle_event(app_event);
    }

    /// Forward the exiting event to the application.
    fn exiting(&mut self, _: &ActiveEventLoop) {
        let AppState::Running { app, integration } = &mut self.state else {
            log::warn!("Exiting signal received before app started");
            return;
        };

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::LoopExiting,
        };
        app.handle_event(app_event);
        #[cfg(feature = "persistence")]
        integration.save(app);
        integration.destroy();
        unsafe {
            ManuallyDrop::drop(integration);
        }
    }

    /// Forward the memory warning event to the application.
    fn memory_warning(&mut self, _: &ActiveEventLoop) {
        let AppState::Running { app, .. } = &mut self.state else {
            log::warn!("Memory warning signal received before app started");
            return;
        };

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::MemoryWarning,
        };
        app.handle_event(app_event);
    }
}
