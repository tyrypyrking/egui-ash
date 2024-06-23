use egui_winit::winit::{self, event_loop::EventLoopBuilder};
use raw_window_handle::HasDisplayHandle;
use std::{
    ffi::CStr,
    mem::ManuallyDrop,
    process::ExitCode,
    sync::{Arc, Mutex},
};

use crate::{
    app::{App, AppCreator, CreationContext},
    event,
    integration::{Integration, IntegrationEvent},
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

    let device_extensions = [ash::khr::swapchain::NAME.to_owned()];

    let event_loop = EventLoopBuilder::<IntegrationEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");

    #[cfg(feature = "persistence")]
    let storage = storage::Storage::from_app_id(&app_id).expect("Failed to create storage");

    let context = egui::Context::default();
    #[cfg(feature = "persistence")]
    if run_option.persistent_egui_memory {
        if let Some(memory) = storage.get_egui_memory() {
            context.memory_mut(|m| *m = memory);
        }
    }

    context.set_embed_viewports(false);
    match run_option.default_theme {
        Theme::Light => {
            context.set_visuals(egui::Visuals::light());
        }
        Theme::Dark => {
            context.set_visuals(egui::Visuals::dark());
        }
    }

    #[allow(unused_mut)] // only mutable when persistence feature is enabled
    let main_window = if let Some(mut viewport_builder) = run_option.viewport_builder {
        #[cfg(feature = "persistence")]
        if run_option.persistent_windows {
            let window_settings = storage
                .get_windows()
                .and_then(|windows| windows.get(&egui::ViewportId::ROOT).map(|s| s.to_owned()))
                .map(|mut settings| {
                    let egui_zoom_factor = context.zoom_factor();
                    settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                        egui_zoom_factor,
                        &event_loop,
                    ));
                    settings.clamp_position_to_monitors(egui_zoom_factor, &event_loop);
                    settings.to_owned()
                });

            if let Some(window_settings) = window_settings {
                viewport_builder = window_settings.initialize_viewport_builder(viewport_builder);
            }
        }

        egui_winit::create_winit_window_builder(
            &context,
            &event_loop,
            viewport_builder.with_visible(false),
        )
        .with_visible(false)
        .build(&event_loop)
        .unwrap()
    } else {
        winit::window::WindowBuilder::new()
            .with_title("egui-ash")
            .with_visible(false)
            .build(&event_loop)
            .unwrap()
    };

    let instance_extensions = ash_window::enumerate_required_extensions(
        event_loop
            .display_handle()
            .expect("Error getting display handle")
            .as_raw(),
    )
    .unwrap();
    let instance_extensions = instance_extensions
        .into_iter()
        .map(|&ext| unsafe { CStr::from_ptr(ext).to_owned() })
        .collect::<Vec<_>>();

    let (image_registry, image_registry_receiver) = ImageRegistry::new();

    let (exit_signal_tx, exit_signal_rx) = std::sync::mpsc::channel();
    let exit_signal = ExitSignal { tx: exit_signal_tx };

    let cc = CreationContext {
        main_window: &main_window,
        context: context.clone(),
        required_instance_extensions: instance_extensions,
        required_device_extensions: device_extensions.into_iter().collect(),
        image_registry,
        exit_signal,
    };
    let (mut app, render_state) = creator.create(cc);

    // ManuallyDrop is required because the integration object needs to be dropped before
    // the app drops for gpu_allocator drop order reasons.
    let mut integration = ManuallyDrop::new(Integration::new(
        &app_id,
        &event_loop,
        context,
        main_window,
        render_state,
        run_option.clear_color,
        run_option.present_mode,
        image_registry_receiver,
        #[cfg(feature = "persistence")]
        storage,
        #[cfg(feature = "persistence")]
        run_option.persistent_windows,
        #[cfg(feature = "persistence")]
        run_option.persistent_egui_memory,
    ));

    let exit_code = Arc::new(Mutex::new(ExitCode::SUCCESS));
    let exit_code_clone = exit_code.clone();
    event_loop
        .run(move |event, event_loop| {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            if let Some(code) = exit_signal_rx.try_recv().ok() {
                *exit_code_clone.lock().unwrap() = code;
                event_loop.exit();
                return;
            }
            match event {
                winit::event::Event::NewEvents(start_cause) => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::NewEvents(start_cause),
                    };
                    app.handle_event(app_event);
                }
                winit::event::Event::WindowEvent {
                    event, window_id, ..
                } => {
                    let consumed = integration.handle_window_event(
                        window_id,
                        &event,
                        &event_loop,
                        run_option.follow_system_theme,
                        &mut app,
                    );
                    if consumed {
                        return;
                    }

                    let Some(viewport_id) = integration.viewport_id_from_window_id(window_id)
                    else {
                        return;
                    };
                    let viewport_event = event::Event::ViewportEvent { viewport_id, event };
                    app.handle_event(viewport_event);
                }
                winit::event::Event::DeviceEvent { device_id, event } => {
                    let device_event = event::Event::DeviceEvent { device_id, event };
                    app.handle_event(device_event);
                }
                #[allow(unused_variables)] // only used when accesskit feature is enabled
                winit::event::Event::UserEvent(integration_event) => {
                    #[cfg(feature = "accesskit")]
                    {
                        integration.handle_accesskit_event(
                            &integration_event.accesskit,
                            event_loop,
                            control_flow,
                            &mut app,
                        );
                        let user_event =
                            event::Event::AccessKitActionRequest(integration_event.accesskit);
                        app.handle_event(user_event);
                    }
                }
                winit::event::Event::Suspended => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::Suspended,
                    };
                    app.handle_event(app_event);
                }
                winit::event::Event::Resumed => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::Resumed,
                    };
                    app.handle_event(app_event);
                    integration.paint_all(event_loop, &mut app);
                }
                winit::event::Event::AboutToWait => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::AboutToWait,
                    };
                    app.handle_event(app_event);
                    integration.paint_all(event_loop, &mut app);
                }
                winit::event::Event::MemoryWarning => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::MemoryWarning,
                    };
                    app.handle_event(app_event);
                }
                winit::event::Event::LoopExiting => {
                    let app_event = event::Event::AppEvent {
                        event: event::AppEvent::LoopExiting,
                    };
                    app.handle_event(app_event);
                    #[cfg(feature = "persistence")]
                    integration.save(&mut app);
                    integration.destroy();
                    unsafe {
                        ManuallyDrop::drop(&mut integration);
                    }
                }
            }
        })
        .expect("Failed to run event loop");
    let code = exit_code.lock().unwrap();
    code.clone()
}
