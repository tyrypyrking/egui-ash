use egui_winit::winit::{
    self,
    application::ApplicationHandler,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowAttributes,
};
use std::{process::ExitCode, time::Duration};

use crate::engine::EngineRenderer;
use crate::host::{EngineHandle, Host};
use crate::types::{EngineStatus, RunOption, VulkanContext};

/// egui-ash v2 entry point.
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
    ) + 'static,
) -> ExitCode {
    let _app_id: String = app_id.into();
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let (exit_tx, exit_rx) = std::sync::mpsc::channel();

    let mut state = AppState {
        vulkan: Some(vulkan),
        options,
        engine: Some(engine),
        ui,
        host: None,
        exit_tx,
    };

    event_loop
        .run_app(&mut state)
        .expect("Failed to run event loop");

    exit_rx
        .recv_timeout(Duration::from_secs(1))
        .unwrap_or(ExitCode::FAILURE)
}

struct AppState<E: EngineRenderer, F> {
    vulkan: Option<VulkanContext>,
    options: RunOption,
    engine: Option<E>,
    ui: F,
    host: Option<Host<E>>,
    exit_tx: std::sync::mpsc::Sender<ExitCode>,
}

impl<E, F> ApplicationHandler for AppState<E, F>
where
    E: EngineRenderer,
    F: FnMut(
        &egui::Context,
        &EngineStatus,
        &mut E::UiState,
        &E::EngineState,
        &EngineHandle<E>,
    ) + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let mut window_attributes = WindowAttributes::default().with_visible(true);

        if let Some(vb) = self.options.viewport_builder.as_ref() {
            if let Some(title) = &vb.title {
                window_attributes = window_attributes.with_title(title.clone());
            }
            if let Some(size) = vb.inner_size {
                window_attributes = window_attributes.with_inner_size(
                    winit::dpi::LogicalSize::new(size.x as f64, size.y as f64),
                );
            }
            if let Some(size) = vb.min_inner_size {
                window_attributes = window_attributes.with_min_inner_size(
                    winit::dpi::LogicalSize::new(size.x as f64, size.y as f64),
                );
            }
            if let Some(size) = vb.max_inner_size {
                window_attributes = window_attributes.with_max_inner_size(
                    winit::dpi::LogicalSize::new(size.x as f64, size.y as f64),
                );
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
        } else {
            window_attributes = window_attributes.with_title("egui-ash");
        }

        let window = event_loop.create_window(window_attributes).unwrap();

        let vulkan = self.vulkan.take().expect("VulkanContext already consumed");
        let engine = self.engine.take().expect("Engine already consumed");

        let host = unsafe {
            Host::new(
                vulkan,
                engine,
                &self.options,
                window,
                event_loop,
                self.exit_tx.clone(),
            )
        };
        self.host = Some(host);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let Some(host) = self.host.as_mut() else {
            return;
        };

        if matches!(event, winit::event::WindowEvent::CloseRequested) {
            unsafe {
                host.destroy();
            }
            self.exit_tx.send(ExitCode::SUCCESS).ok();
            event_loop.exit();
            return;
        }

        host.handle_window_event(&event);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(host) = self.host.as_mut() else {
            return;
        };
        unsafe {
            host.frame(&mut self.ui);
        }
    }
}
