use ash::vk;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// User-provided Vulkan context. The user creates all Vulkan objects
/// before calling `run()` and destroys them after `run()` returns.
///
/// # Required Vulkan features
///
/// The device **must** be created with the following features enabled
/// (all core in Vulkan 1.2):
///
/// - `VkPhysicalDeviceVulkan12Features::timelineSemaphore`
/// - `VkPhysicalDeviceVulkan12Features::descriptorBindingSampledImageUpdateAfterBind`
/// - `VkPhysicalDeviceVulkan12Features::descriptorBindingUpdateUnusedWhilePending`
///
/// These are used internally by the compositor for texture descriptor
/// updates while previous frames are still in flight.
///
/// # 1.0.0-alpha restriction: single queue family
///
/// `host_queue_family_index` and `engine_queue_family_index` **must be equal**
/// in 1.0.0-alpha. Cross-queue-family ownership transfer of the engine
/// viewport image is not yet wired up on the host side — the compositor does
/// not execute the matching acquire barrier when the engine releases the
/// image, which would be UB per the Vulkan spec. `Host::new` asserts this.
///
/// On hardware that only exposes one graphics queue family (common on Intel
/// integrated, AMD RADV), pick any queue from that family for both `host_queue`
/// and `engine_queue`. If they end up being the same `VkQueue` handle, supply
/// `queue_mutex: Some(...)` to serialise submits.
///
/// See `docs/known-limitations.md` for the cross-family roadmap.
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    /// Queue for host (egui rendering + compositing).
    /// Usually a distinct `VkQueue` from `engine_queue`; see `queue_mutex`
    /// below for the single-queue-family fallback path.
    pub host_queue: vk::Queue,
    pub host_queue_family_index: u32,
    /// Queue for engine scene rendering (used from the engine thread).
    pub engine_queue: vk::Queue,
    pub engine_queue_family_index: u32,
    /// Optional mutex serialising access to `host_queue` / `engine_queue`.
    ///
    /// When the two queue handles are distinct (the normal case), set to
    /// `None` — both threads can submit concurrently.
    ///
    /// When the underlying hardware only exposes a single graphics queue
    /// (common on Intel integrated and AMD RADV on RDNA3), the caller is
    /// forced to share one `VkQueue` between host and engine. In that case
    /// callers must supply a shared `Arc<Mutex<()>>` here; egui-ash will
    /// lock it around every `vkQueueSubmit` / `vkQueuePresentKHR` it makes,
    /// and the engine thread is expected to lock the same mutex around its
    /// own submits. That's enough to satisfy Vulkan's "host access to a
    /// `VkQueue` must be externally synchronised" requirement.
    pub queue_mutex: Option<Arc<Mutex<()>>>,
}

/// Context provided to the engine on its dedicated thread.
pub struct EngineContext {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub initial_extent: vk::Extent2D,
    pub format: vk::Format,
    /// Optional mutex for serialising queue access when host and engine
    /// share the same `VkQueue`. The engine **must** lock this around every
    /// `vkQueueSubmit` / `vkQueueSubmit2` call. `None` when the queues are
    /// distinct.
    pub queue_mutex: Option<Arc<Mutex<()>>>,
    /// Handle for registering user-owned Vulkan images as egui textures.
    /// Shared with the UI-thread handle exposed via `EngineHandle::image_registry`.
    /// Typically used from `EngineRenderer::init` to register static
    /// scene-output textures that the UI can display in panels.
    pub image_registry: crate::image_registry::ImageRegistry,
}

/// A render target lent to the engine by the host.
///
/// The engine must wait on `wait_semaphore` for at least `wait_value` before
/// writing to `image`, and must signal `signal_semaphore` with `signal_value`
/// when its GPU work on `image` completes. `wait_semaphore` and
/// `signal_semaphore` are distinct timelines — each has exactly one signaler
/// (engine for `signal_semaphore`, host compositor for `wait_semaphore`).
pub struct RenderTarget {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    pub wait_semaphore: vk::Semaphore,
    pub wait_value: u64,
    pub signal_semaphore: vk::Semaphore,
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
        }
    }

    /// Acquire barrier for cross-queue-family ownership transfer.
    ///
    /// Always `None` in 1.0.0-alpha — see `VulkanContext` docs for the
    /// single-queue-family restriction. Engines should still call this and
    /// execute the returned barrier if `Some` so they remain source-compatible
    /// when cross-family support lands.
    pub fn acquire_barrier(&self) -> Option<&vk::ImageMemoryBarrier2<'static>> {
        self.acquire_barrier.as_ref()
    }

    /// Release barrier for cross-queue-family ownership transfer.
    ///
    /// Always `None` in 1.0.0-alpha — see `VulkanContext` docs for the
    /// single-queue-family restriction. Engines should still call this and
    /// execute the returned barrier if `Some` so they remain source-compatible
    /// when cross-family support lands.
    pub fn release_barrier(&self) -> Option<&vk::ImageMemoryBarrier2<'static>> {
        self.release_barrier.as_ref()
    }
}

/// Returned by `RenderTarget::complete()` after the engine finishes rendering.
pub struct CompletedFrame {
    pub(crate) image: vk::Image,
    pub(crate) signal_value: u64,
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
///
/// Passed to [`run`](crate::run) by value. `Default` produces a reasonable
/// set for desktop apps (FIFO present, black clear color, 30 s autosave,
/// persistence on).
pub struct RunOption {
    /// Swapchain present mode. `FIFO` is safe everywhere (guaranteed
    /// available per the Vulkan spec) and vsyncs. For low-latency use
    /// `MAILBOX` if available on the target platform; for uncapped
    /// framerate use `IMMEDIATE`.
    pub present_mode: vk::PresentModeKHR,
    /// Clear color for pixels not covered by egui or the engine viewport
    /// (e.g., margin around the engine viewport texture when it's shown at
    /// less than full extent). RGBA in linear space, 0..=1 per component.
    pub clear_color: [f32; 4],
    /// Initial viewport configuration for the root window. If `None`, a
    /// minimal default is used (title "egui-ash", no fixed size). Any
    /// fields set here propagate to winit's `WindowAttributes` at window
    /// creation time. Persisted window geometry (when the `persistence`
    /// feature is enabled and [`Self::persistent_windows`] is `true`)
    /// overrides the builder's position / inner_size / maximized /
    /// fullscreen on subsequent launches.
    pub viewport_builder: Option<egui::ViewportBuilder>,
    /// If `true`, `egui::Context::set_visuals` follows the OS's current
    /// light/dark preference. `false` pins the theme to [`Self::default_theme`].
    pub follow_system_theme: bool,
    /// Fallback theme used when [`Self::follow_system_theme`] is `false`
    /// or the OS doesn't report a preference.
    pub default_theme: egui_winit::winit::window::Theme,
    /// If `true`, window position and size are saved on clean shutdown
    /// (and on every [`Self::auto_save_interval`] tick) to the per-`app_id`
    /// RON file under the platform's data directory, and restored on next
    /// launch. Default: `true`.
    #[cfg(feature = "persistence")]
    pub persistent_windows: bool,
    /// If `true`, `egui::Memory` (collapsed header state, scroll
    /// positions, text-edit state, etc.) is saved and restored alongside
    /// window geometry. Default: `true`.
    #[cfg(feature = "persistence")]
    pub persistent_egui_memory: bool,
    /// Interval at which persisted state (window geometry, egui memory,
    /// and any user values written via [`Storage::set_value`](crate::Storage::set_value))
    /// is flushed to disk. `None` disables periodic flush — state is only
    /// written on clean shutdown. Default: `Some(Duration::from_secs(30))`.
    ///
    /// The periodic flush exists for crash safety — if the process dies
    /// between user actions, the most recent in-memory state still
    /// survives up to one interval old.
    #[cfg(feature = "persistence")]
    pub auto_save_interval: Option<Duration>,
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
            #[cfg(feature = "persistence")]
            auto_save_interval: Some(Duration::from_secs(30)),
        }
    }
}
