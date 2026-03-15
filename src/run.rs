use egui_winit::winit::{
    self,
    application::ApplicationHandler,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Theme, WindowAttributes},
};
use raw_window_handle::HasDisplayHandle as _;
use std::{
    ffi::{CStr, CString},
    mem::ManuallyDrop,
    process::ExitCode,
    time::Duration,
};

use crate::{
    app::{App, AppCreator, CreationContext},
    event,
    integration::{Integration, IntegrationEvent},
    renderer::ImageRegistry,
    Allocator,
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
    /// `vk::PresentModeKHR`
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
/// egui_winit_ash::run("my_app", MyAppCreator, RunOption::default());
///
/// ```
pub fn run<C: AppCreator<A> + 'static, A: Allocator + 'static>(
    app_id: impl Into<String>,
    creator: C,
    run_option: RunOption,
) -> ExitCode {
    let app_id: String = app_id.into();

    let event_loop = EventLoop::<IntegrationEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");

    /*match run_option.default_theme {
        Theme::Light => {
            context.set_visuals(egui::Visuals::light());
        }
        Theme::Dark => {
            context.set_visuals(egui::Visuals::dark());
        }
    }*/

    let (exit_signal_tx, exit_signal_rx) = std::sync::mpsc::channel();
    let exit_signal = ExitSignal { tx: exit_signal_tx };

    let mut state = State {
        app_id,
        run_option,
        exit_signal,
        creator,
        app: None,
        #[cfg(feature = "accesskit")]
        event_loop_proxy: event_loop.create_proxy(),
        integration: None,
    };

    event_loop
        .run_app(&mut state)
        .expect("Failed to run event loop");

    exit_signal_rx.recv_timeout(Duration::from_secs(1)).unwrap()
}

struct State<C, A>
where
    C: AppCreator<A> + 'static,
    A: Allocator + 'static,
{
    app_id: String,
    run_option: RunOption,
    exit_signal: ExitSignal,
    creator: C,
    app: Option<C::App>,
    integration: Option<ManuallyDrop<Box<Integration<A>>>>,
    #[cfg(feature = "accesskit")]
    event_loop_proxy: winit::event_loop::EventLoopProxy<IntegrationEvent>,
}

impl<C, A> State<C, A>
where
    C: AppCreator<A> + 'static,
    A: Allocator + 'static,
{
    #[cfg(feature = "persistence")]
    fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        context: &egui::Context,
        storage: &storage::Storage,
    ) -> winit::window::Window {
        let Some(ref viewport_builder) = self.run_option.viewport_builder else {
            let window_attributes = WindowAttributes::default()
                .with_visible(false)
                .with_title("egui-ash");
            return event_loop.create_window(window_attributes).unwrap();
        };

        #[cfg(feature = "persistence")]
        if self.run_option.persistent_windows {
            let egui_zoom_factor = context.zoom_factor();
            let window_settings = storage
                .get_windows()
                .and_then(|windows| windows.get(&egui::ViewportId::ROOT).map(|s| s.to_owned()))
                .map(|mut settings| {
                    settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                        egui_zoom_factor,
                        event_loop,
                    ));
                    settings.clamp_position_to_monitors(egui_zoom_factor, event_loop);
                    settings.to_owned()
                });

            if let Some(window_settings) = window_settings {
                self.run_option.viewport_builder =
                    Some(window_settings.initialize_viewport_builder(
                        egui_zoom_factor,
                        event_loop,
                        viewport_builder.clone(),
                    ));
            }
        }

        let window_attributes = WindowAttributes::default().with_visible(false);
        event_loop.create_window(window_attributes).unwrap()
    }

    #[cfg(not(feature = "persistence"))]
    fn create_window(
        &mut self,
        event_loop: &ActiveEventLoop,
        _context: &egui::Context,
    ) -> winit::window::Window {
        let window_attributes = WindowAttributes::default()
            .with_visible(false)
            .with_title("egui-ash");
        event_loop.create_window(window_attributes).unwrap();

        let window_attributes = WindowAttributes::default().with_visible(false);
        event_loop.create_window(window_attributes).unwrap()
    }

    fn initial_setup(&mut self, event_loop: &ActiveEventLoop) {
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

        #[cfg(feature = "persistence")]
        let main_window = self.create_window(event_loop, &context, &storage);
        #[cfg(not(feature = "persistence"))]
        let main_window = self.create_window(event_loop, &context);

        let (image_registry, image_registry_receiver) = ImageRegistry::new();

        let instance_extensions = required_instance_extensions(event_loop);
        let device_extensions = vec![ash::khr::swapchain::NAME.to_owned()];

        let cc = CreationContext {
            //Display handle, Window handle
            main_window: &main_window,
            context: context.clone(),
            required_instance_extensions: instance_extensions,
            required_device_extensions: device_extensions,
            image_registry,
            exit_signal: self.exit_signal.clone(),
        };
        let (app, render_state) = self.creator.create(cc);

        // ManuallyDrop<Box<>> is required because:
        // 1. Integration must drop before App for gpu_allocator drop order reasons.
        // 2. Box gives a stable heap address so the raw pointer in
        //    register_immediate_viewport_renderer remains valid.
        let mut integration = Box::new(Integration::new(
            &self.app_id,
            event_loop,
            context,
            main_window,
            render_state,
            self.run_option.present_mode,
            image_registry_receiver,
            Some(self.run_option.default_theme),
            #[cfg(feature = "accesskit")]
            &self.event_loop_proxy,
            #[cfg(feature = "persistence")]
            storage,
            #[cfg(feature = "persistence")]
            self.run_option.persistent_windows,
            #[cfg(feature = "persistence")]
            self.run_option.persistent_egui_memory,
        ));

        // Register callback NOW that integration has a stable heap address
        integration.register_immediate_viewport_renderer(event_loop);

        self.integration = Some(ManuallyDrop::new(integration));
        self.app = Some(app);
    }
}

fn required_instance_extensions(event_loop: &ActiveEventLoop) -> Vec<CString> {
    let instance_extensions = ash_window::enumerate_required_extensions(
        event_loop
            .display_handle()
            .expect("Unable to retrieve a display handle")
            .as_raw(),
    )
    .unwrap();

    instance_extensions
        .iter()
        .map(|&ext| unsafe { CStr::from_ptr(ext).to_owned() })
        .collect::<Vec<_>>()
}

impl<C, A> ApplicationHandler<IntegrationEvent> for State<C, A>
where
    C: AppCreator<A> + 'static,
    A: Allocator + 'static,
{
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        self.initial_setup(event_loop);

        // ------- HANDLE RESUMED
        let app_event = event::Event::AppEvent {
            event: event::AppEvent::Resumed,
        };
        let (integration, app) = (
            self.integration.as_mut().unwrap(),
            self.app.as_mut().unwrap(),
        );
        app.handle_event(app_event);
        integration.paint_all(event_loop, app);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let (integration, app) = (
            self.integration.as_mut().unwrap(),
            self.app.as_mut().unwrap(),
        );

        let consumed = integration.handle_window_event(
            window_id,
            &event,
            event_loop,
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

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let app = self.app.as_mut().unwrap();

        let device_event = event::Event::DeviceEvent { device_id, event };
        app.handle_event(device_event);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: IntegrationEvent) {
        #[cfg(feature = "accesskit")]
        {
            let (integration, app) = (
                self.integration.as_mut().unwrap(),
                self.app.as_mut().unwrap(),
            );

            integration.handle_accesskit_event(&_event.accesskit, _event_loop, app);
            let user_event = event::Event::AccessKitActionRequest(_event.accesskit);
            app.handle_event(user_event);
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        let app = self.app.as_mut().unwrap();

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::Suspended,
        };
        app.handle_event(app_event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let (integration, app) = (
            self.integration.as_mut().unwrap(),
            self.app.as_mut().unwrap(),
        );

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::AboutToWait,
        };
        app.handle_event(app_event);
        integration.paint_all(event_loop, app);
    }

    fn memory_warning(&mut self, _event_loop: &ActiveEventLoop) {
        let app = self.app.as_mut().unwrap();

        let app_event = event::Event::AppEvent {
            event: event::AppEvent::MemoryWarning,
        };
        app.handle_event(app_event);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let (integration, app) = (
            self.integration.as_mut().unwrap(),
            self.app.as_mut().unwrap(),
        );

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
        self.exit_signal.send(ExitCode::SUCCESS);
    }
}

/*
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Ok(code) = exit_signal_rx.try_recv() {
                *exit_code_clone.lock().unwrap() = code;
                event_loop.exit();
                return;
            }
*/
