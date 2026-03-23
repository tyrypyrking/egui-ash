use ash::vk;
use std::time::Duration;

/// User-provided Vulkan context. The user creates all Vulkan objects
/// before calling `run()` and destroys them after `run()` returns.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    /// Queue for host (egui rendering + compositing). Must differ from engine_queue.
    pub host_queue: vk::Queue,
    pub host_queue_family_index: u32,
    /// Queue for engine scene rendering. Must differ from host_queue.
    pub engine_queue: vk::Queue,
    pub engine_queue_family_index: u32,
}

/// Context provided to the engine on its dedicated thread.
pub struct EngineContext {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub initial_extent: vk::Extent2D,
    pub format: vk::Format,
}

/// A render target lent to the engine by the host.
pub struct RenderTarget {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub timeline: vk::Semaphore,
    pub wait_value: u64,
    pub signal_value: u64,
    pub(crate) acquire_barrier: Option<vk::ImageMemoryBarrier2<'static>>,
    pub(crate) release_barrier: Option<vk::ImageMemoryBarrier2<'static>>,
}

impl RenderTarget {
    /// Consume this target and produce a CompletedFrame.
    /// Call after submitting GPU work that signals `signal_value`.
    pub fn complete(self) -> CompletedFrame {
        CompletedFrame {
            image: self.image,
            signal_value: self.signal_value,
            release_barrier: self.release_barrier,
        }
    }

    /// Acquire barrier for cross-queue-family ownership transfer.
    /// None if both queues share the same family.
    pub fn acquire_barrier(&self) -> Option<&vk::ImageMemoryBarrier2<'static>> {
        self.acquire_barrier.as_ref()
    }

    /// Release barrier for cross-queue-family ownership transfer.
    /// None if both queues share the same family.
    pub fn release_barrier(&self) -> Option<&vk::ImageMemoryBarrier2<'static>> {
        self.release_barrier.as_ref()
    }
}

/// Returned by `RenderTarget::complete()` after the engine finishes rendering.
pub struct CompletedFrame {
    pub(crate) image: vk::Image,
    pub(crate) signal_value: u64,
    pub(crate) release_barrier: Option<vk::ImageMemoryBarrier2<'static>>,
}

/// Engine health observable from the UI closure.
pub struct EngineStatus {
    pub health: EngineHealth,
    pub viewport_texture_id: egui::TextureId,
    pub frames_delivered: u64,
    pub last_frame_time: Option<Duration>,
}

/// Engine health states.
#[derive(Debug, Clone)]
pub enum EngineHealth {
    Starting,
    Running,
    Stopped,
    Crashed { message: String },
}

impl EngineHealth {
    pub fn is_alive(&self) -> bool {
        matches!(self, Self::Starting | Self::Running)
    }

    pub fn is_crashed(&self) -> bool {
        matches!(self, Self::Crashed { .. })
    }
}

/// Error returned by `EngineHandle::restart()`.
#[derive(Debug)]
pub enum EngineRestartError {
    /// Engine is still running — stop it first.
    StillRunning,
}

impl std::fmt::Display for EngineRestartError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StillRunning => write!(f, "engine is still running"),
        }
    }
}
impl std::error::Error for EngineRestartError {}

/// Run options for the host.
pub struct RunOption {
    /// Swapchain present mode.
    pub present_mode: vk::PresentModeKHR,
    /// Clear color behind the engine viewport.
    pub clear_color: [f32; 4],
    /// Viewport builder for the root window.
    pub viewport_builder: Option<egui::ViewportBuilder>,
    /// Follow system theme.
    pub follow_system_theme: bool,
    /// Default theme.
    pub default_theme: egui_winit::winit::window::Theme,
    #[cfg(feature = "persistence")]
    pub persistent_windows: bool,
    #[cfg(feature = "persistence")]
    pub persistent_egui_memory: bool,
}

impl Default for RunOption {
    fn default() -> Self {
        Self {
            present_mode: vk::PresentModeKHR::FIFO,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            viewport_builder: None,
            follow_system_theme: true,
            default_theme: egui_winit::winit::window::Theme::Light,
            #[cfg(feature = "persistence")]
            persistent_windows: true,
            #[cfg(feature = "persistence")]
            persistent_egui_memory: true,
        }
    }
}
