#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit;
pub use egui_winit::winit;

pub enum AppEvent {
    NewEvents(winit::event::StartCause),
    Suspended,
    Resumed,
    AboutToWait,
    LoopExiting,
    MemoryWarning,
}

pub enum Event<'a> {
    DeferredViewportCreated {
        viewport_id: egui::ViewportId,
        window: &'a winit::window::Window,
    },
    ViewportEvent {
        viewport_id: egui::ViewportId,
        event: winit::event::WindowEvent,
    },
    AppEvent {
        event: AppEvent,
    },
    DeviceEvent {
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    },
    #[cfg(feature = "accesskit")]
    AccessKitActionRequest(accesskit_winit::Event),
}
