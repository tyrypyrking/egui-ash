use ash::vk;

// ── v2 event types ──

/// Input events forwarded from the host to the engine thread.
pub enum EngineEvent {
    /// Pointer moved within the engine viewport.
    /// Position is in viewport-local physical pixels, (0,0) = top-left.
    Pointer {
        position: [f32; 2],
        buttons: PointerButtons,
    },
    /// Key pressed/released while engine viewport is focused.
    Key {
        key: egui::Key,
        pressed: bool,
        modifiers: egui::Modifiers,
    },
    /// Scroll within the engine viewport.
    Scroll { delta: [f32; 2] },
    /// Engine viewport resized.
    Resize { extent: vk::Extent2D },
    /// Engine viewport gained/lost focus.
    Focus(bool),
    /// Graceful shutdown signal. Sent before the target channel closes.
    Shutdown,
}

impl std::fmt::Debug for EngineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pointer { position, buttons } => f
                .debug_struct("Pointer")
                .field("position", position)
                .field("buttons", buttons)
                .finish(),
            Self::Key { key, pressed, modifiers } => f
                .debug_struct("Key")
                .field("key", key)
                .field("pressed", pressed)
                .field("modifiers", modifiers)
                .finish(),
            Self::Scroll { delta } => f.debug_struct("Scroll").field("delta", delta).finish(),
            Self::Resize { extent } => f
                .debug_struct("Resize")
                .field("width", &extent.width)
                .field("height", &extent.height)
                .finish(),
            Self::Focus(focused) => f.debug_tuple("Focus").field(focused).finish(),
            Self::Shutdown => write!(f, "Shutdown"),
        }
    }
}

/// Button state for pointer events.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointerButtons {
    pub primary: bool,
    pub secondary: bool,
    pub middle: bool,
}

// ── v1 event types (kept for v1 module compatibility, removed in Task 9) ──

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
