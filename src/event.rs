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
    /// Raw winit device event — raw mouse deltas, raw key scancodes, etc.
    ///
    /// Unlike `Pointer` / `Key`, these are device-level events not tied to
    /// any particular window. Useful for FPS-style camera controls that
    /// need raw mouse deltas (via `DeviceEvent::MouseMotion`) rather than
    /// per-window cursor positions. Forwarded as-is from winit; not
    /// filtered or processed by the host.
    Device(egui_winit::winit::event::DeviceEvent),
    /// Application lifecycle transitions forwarded from winit.
    Lifecycle(AppLifecycleEvent),
    /// Graceful shutdown signal. Sent before the target channel closes.
    Shutdown,
}

/// Application-level lifecycle events forwarded to the engine.
///
/// `Resumed` and `Suspended` primarily matter on mobile (Android / iOS)
/// where the window surface is destroyed and recreated as the app moves
/// between foreground and background. On desktop the first `Resumed` is
/// absorbed internally by `run()` to create the window, so only subsequent
/// resumes (if any) reach the engine.
#[derive(Debug, Clone, Copy)]
pub enum AppLifecycleEvent {
    /// App moved to the background (mobile: surface destroyed).
    Suspended,
    /// App moved to the foreground after a prior Suspended.
    Resumed,
    /// OS is signalling memory pressure — release caches if possible.
    MemoryWarning,
    /// Event loop is about to exit. Last chance to persist transient state
    /// that isn't already covered by the `persistence` feature's automatic
    /// window/egui-memory save.
    LoopExiting,
}

impl std::fmt::Debug for EngineEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pointer { position, buttons } => f
                .debug_struct("Pointer")
                .field("position", position)
                .field("buttons", buttons)
                .finish(),
            Self::Key {
                key,
                pressed,
                modifiers,
            } => f
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
            Self::Device(event) => f.debug_tuple("Device").field(event).finish(),
            Self::Lifecycle(event) => f.debug_tuple("Lifecycle").field(event).finish(),
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
