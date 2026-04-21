//! User-facing texture registration — v1-parity restoration §5.B5.
//!
//! `ImageRegistry` allows users to register their own Vulkan images
//! (via `vk::ImageView` + `vk::Sampler`) and receive an `egui::TextureId`
//! back, which can then be passed to `egui::Image::new(...)` to display
//! the image inside any egui panel.
//!
//! The registry is channel-based: user code sends `RegistryCommand`s
//! through an mpsc queue; the `Compositor` drains the queue at the
//! start of each frame and allocates / updates / frees descriptor sets.
//! This keeps the user-facing API thread-safe while descriptor-set
//! allocation stays on the host-render thread.

use ash::vk;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};

/// Commands from user code to the compositor's user-texture table.
pub(crate) enum RegistryCommand {
    Register {
        id: u64,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    },
    Update {
        id: u64,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    },
    Unregister {
        id: u64,
    },
}

/// User-facing handle for registering Vulkan images with egui-ash.
///
/// Clone to share across threads. Pass to `register()` to allocate a
/// new `egui::TextureId` backed by a descriptor set that the compositor
/// will bind when egui issues a draw call against that ID.
///
/// Obtained two ways, both backed by the same channel:
/// - On the UI thread via [`EngineHandle::image_registry`](crate::EngineHandle::image_registry).
/// - On the engine thread via [`EngineContext::image_registry`](crate::EngineContext).
#[derive(Clone)]
pub struct ImageRegistry {
    tx: mpsc::Sender<RegistryCommand>,
    next_id: Arc<AtomicU64>,
}

impl ImageRegistry {
    /// Register a Vulkan image for use in egui panels.
    ///
    /// Returns a [`UserTextureHandle`] whose [`id()`](UserTextureHandle::id)
    /// you pass to `egui::Image::new(id)` (or equivalents). Dropping the
    /// handle sends an unregister command — the descriptor set is freed on
    /// the next frame, and the user remains responsible for destroying the
    /// underlying `vk::ImageView` and `vk::Sampler`.
    ///
    /// # Image state requirements
    ///
    /// The image **must** be in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
    /// whenever egui-ash samples from it. The caller is responsible for
    /// recording the layout transition on their own queue and ensuring
    /// synchronisation with the host queue that samples the image — e.g.
    /// via a timeline semaphore signalled after the transition.
    ///
    /// # Descriptor indexing
    ///
    /// Internally the descriptor set uses the same
    /// `UPDATE_AFTER_BIND_POOL` layout as the engine-viewport descriptor.
    /// That means `update()` and `Drop` are safe to call even while a
    /// previous frame referencing the same ID is still in flight; see
    /// `VulkanContext` docs for the required Vulkan 1.2 features.
    pub fn register(&self, image_view: vk::ImageView, sampler: vk::Sampler) -> UserTextureHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let _ = self.tx.send(RegistryCommand::Register {
            id,
            image_view,
            sampler,
        });
        UserTextureHandle {
            id: egui::TextureId::User(id),
            raw_id: id,
            tx: self.tx.clone(),
        }
    }
}

impl std::fmt::Debug for ImageRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageRegistry").finish()
    }
}

/// Handle returned from [`ImageRegistry::register`].
///
/// Carries the egui texture ID and a channel handle. Dropping this value
/// sends an unregister command to the compositor. Not `Send` — if you
/// need to share ownership, wrap in `Arc<UserTextureHandle>`.
pub struct UserTextureHandle {
    id: egui::TextureId,
    raw_id: u64,
    tx: mpsc::Sender<RegistryCommand>,
}

impl UserTextureHandle {
    /// The egui texture ID. Pass to `egui::Image::new(handle.id())` or
    /// equivalents.
    pub fn id(&self) -> egui::TextureId {
        self.id
    }

    /// Replace the image view and sampler backing this texture ID. Safe
    /// to call while previous frames are still sampling the old binding
    /// because the underlying descriptor set layout uses
    /// `UPDATE_AFTER_BIND_POOL`.
    ///
    /// Note: the caller must still ensure the *old* image is not
    /// destroyed until any in-flight frames using it have completed —
    /// typically via a timeline semaphore or `device_wait_idle`.
    pub fn update(&self, image_view: vk::ImageView, sampler: vk::Sampler) {
        let _ = self.tx.send(RegistryCommand::Update {
            id: self.raw_id,
            image_view,
            sampler,
        });
    }
}

impl Drop for UserTextureHandle {
    fn drop(&mut self) {
        // Best-effort: channel may already be closed if Host destroyed.
        let _ = self
            .tx
            .send(RegistryCommand::Unregister { id: self.raw_id });
    }
}

impl std::fmt::Debug for UserTextureHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserTextureHandle")
            .field("id", &self.id)
            .finish()
    }
}

/// Construct both ends of the registry channel. Called once from
/// `Host::new`. The `ImageRegistry` end is handed to user code via
/// `EngineHandle` + `EngineContext`; the receiver end is owned by the
/// `Compositor` and drained each frame.
pub(crate) fn new_pair() -> (ImageRegistry, mpsc::Receiver<RegistryCommand>) {
    let (tx, rx) = mpsc::channel();
    // User IDs start at 0 and grow monotonically. `u64::MAX` is
    // reserved for the engine viewport texture — see `compositor.rs`.
    let next_id = Arc::new(AtomicU64::new(0));
    let registry = ImageRegistry { tx, next_id };
    (registry, rx)
}
