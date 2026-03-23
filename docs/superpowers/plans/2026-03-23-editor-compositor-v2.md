# Editor/Compositor v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite egui-ash from a turn-key application framework into an editor/compositor library where the user's ash engine is a fault-isolated guest receiving lent render targets.

**Architecture:** Host (library, main thread) owns the winit event loop, egui context, swapchain, and compositing pipeline. Engine (user code, dedicated thread) implements `EngineRenderer` and renders into double-buffered VkImages synchronized via a timeline semaphore. State exchange uses `arc_swap::ArcSwap<T>` with user-defined `UiState`/`EngineState` types. Fault isolation via `catch_unwind`.

**Tech Stack:** Rust, ash 0.38, egui 0.33, egui-winit 0.33, arc_swap, Vulkan 1.2 (timeline semaphores)

**Spec:** `docs/superpowers/specs/2026-03-23-editor-compositor-redesign.md`

---

## File Structure

### New files (v2 core)

| File | Responsibility |
|---|---|
| `src/types.rs` | Public API types: `VulkanContext`, `EngineContext`, `RenderTarget`, `CompletedFrame`, `EngineStatus`, `EngineHealth`, `EngineEvent`, `EngineHandle`, `RunOption` |
| `src/engine.rs` | `EngineRenderer` trait definition |
| `src/mailbox.rs` | `Mailbox` (single-slot overwrite channel), `TargetSender`/`TargetReceiver` (bounded channel wrapper for render targets) |
| `src/state_exchange.rs` | `StateWriter<T>` / `StateReader<T>` wrappers around `ArcSwap<T>` |
| `src/render_targets.rs` | `RenderTargetPool` — creates, destroys, and manages the two VkImages + timeline semaphore + barrier generation |
| `src/compositor.rs` | Host-side compositing pipeline: swapchain, egui render pass/pipeline, egui texture management (including engine viewport as egui User texture) |
| `src/engine_thread.rs` | `engine_thread_main()` function — the `catch_unwind` loop that drives the engine |
| `src/host.rs` | `Host` struct — owns egui context, swapchain, compositor, channels; orchestrates per-frame protocol |
| `src/run.rs` | `run()` entry point — builds event loop, creates Host, spawns engine thread, implements `ApplicationHandler` |
| `src/event.rs` | `EngineEvent` enum + input forwarding conversion from winit events |
| `src/lib.rs` | Module declarations and public re-exports |

### Files removed (v1 code)

| File | Reason |
|---|---|
| `src/app.rs` | Replaced by `EngineRenderer` trait + `ui` closure |
| `src/allocator.rs` | Engine manages its own memory |
| `src/gpu_allocator.rs` | Same |
| `src/integration.rs` | Replaced by `Host` |
| `src/renderer.rs` | Replaced by `compositor.rs` (egui-only rendering) + `render_targets.rs` |
| `src/viewport_context.rs` | Replaced by `compositor.rs` + `render_targets.rs` |
| `src/utils.rs` | Barrier utilities moved inline where needed |

### Files kept (modified)

| File | Changes |
|---|---|
| `src/storage.rs` | Unchanged (persistence feature) |
| `Cargo.toml` | Add `arc-swap` dependency, remove `gpu-allocator` optional dep |

### Shaders

No new shaders needed. The engine viewport texture is registered as an egui User texture and drawn by egui's existing `vert.spv`/`frag.spv` shaders (in `src/shaders/spv/`) when the user places an `egui::Image` widget. Existing shaders in `src/shaders/spv/` are kept.

---

## Task Dependency Graph

```
Task 1 (Cargo.toml + types) ─────────────┐
Task 2 (EngineRenderer trait) ────────────┤
Task 3 (mailbox + state_exchange) ────────┼─→ Task 7 (engine_thread)
Task 4 (render_targets) ──────────────────┤        │
Task 5 (event) ───────────────────────────┘        │
Task 6 (compositor) ──────────────────────────→ Task 8 (host) ──→ Task 9 (run.rs) ──→ Task 10 (cleanup + example)
```

Tasks 1-6 are independent leaf tasks. Task 7 depends on 1-5. Task 8 depends on 6-7. Task 9 depends on 8. Task 10 depends on 9.

---

### Task 1: Cargo.toml + Public API Types

**Files:**
- Modify: `Cargo.toml`
- Create: `src/types.rs`

- [ ] **Step 1: Update Cargo.toml**

Add `arc-swap` dependency. Remove `gpu-allocator` optional dependency and feature. The `Allocator` trait is gone in v2 — engines manage their own memory.

```toml
# Add to [dependencies]:
arc-swap = "1.7"

# Remove from [dependencies]:
# gpu-allocator = { version = "0.27.0", ... }

# Remove from [features]:
# gpu-allocator = [ "dep:gpu-allocator" ]
```

Keep all other dependencies (`ash`, `egui`, `egui-winit`, `bytemuck`, `anyhow`, `log`, `raw-window-handle`, persistence deps).

- [ ] **Step 2: Create `src/types.rs` with all public structs and enums**

```rust
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
```

- [ ] **Step 3: Verify it compiles in isolation**

Run: `cargo check` (will fail on missing modules — that's expected. Just verify `types.rs` has no syntax errors by temporarily adding `mod types;` to `lib.rs` and commenting out other modules)

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/types.rs
git commit -m "feat(v2): add public API types and update dependencies"
```

---

### Task 2: EngineRenderer Trait

**Files:**
- Create: `src/engine.rs`

- [ ] **Step 1: Create `src/engine.rs`**

```rust
use crate::types::{CompletedFrame, EngineContext, EngineEvent, RenderTarget};

/// Trait implemented by the user's rendering engine.
/// Runs on a dedicated thread spawned by the library.
///
/// # Panic Safety
/// If any method panics, the library catches it via `catch_unwind`.
/// The UI remains responsive, showing a black viewport.
/// Use `EngineHandle::restart()` from the UI to spawn a fresh instance.
///
/// # Drop
/// `destroy()` is called on clean shutdown only. Implement `Drop` for
/// GPU resource cleanup that must happen even after a panic.
pub trait EngineRenderer: Send + 'static {
    /// State written by the UI closure, read by the engine each frame.
    type UiState: Send + Sync + Clone + Default;

    /// State written by the engine, read by the UI closure each frame.
    type EngineState: Send + Sync + Clone + Default;

    /// Called once on the engine thread after spawning.
    /// Create command pools, pipelines, buffers here.
    fn init(&mut self, ctx: EngineContext);

    /// Render a frame into the provided target.
    ///
    /// The engine must:
    /// 1. Wait on `target.wait_value` (GPU wait via timeline semaphore in submit)
    /// 2. Include `target.acquire_barrier()` in the command buffer if `Some`
    /// 3. Render into `target.image`
    /// 4. Include `target.release_barrier()` in the command buffer if `Some`
    /// 5. Signal `target.signal_value` via timeline semaphore in submit
    /// 6. Return `target.complete()`
    fn render(
        &mut self,
        target: RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> CompletedFrame;

    /// Process an input event forwarded from the host.
    /// Called on the engine thread between frames.
    fn handle_event(&mut self, event: EngineEvent);

    /// Called on clean shutdown before the engine thread exits.
    fn destroy(&mut self);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/engine.rs
git commit -m "feat(v2): add EngineRenderer trait definition"
```

---

### Task 3: Mailbox + State Exchange Primitives

**Files:**
- Create: `src/mailbox.rs`
- Create: `src/state_exchange.rs`

- [ ] **Step 1: Create `src/mailbox.rs`**

A single-slot overwrite channel (mailbox) for `CompletedFrame`, and a thin wrapper around `std::sync::mpsc` for render target exchange.

```rust
use std::sync::{mpsc, Mutex};

use crate::types::CompletedFrame;

// ── Mailbox: single-slot overwrite channel ──────────────────────

/// Sender end of the mailbox. Overwrites the previous value if unread.
pub(crate) struct MailboxSender {
    slot: std::sync::Arc<MailboxInner>,
}

/// Receiver end of the mailbox. Returns the latest value or None.
pub(crate) struct MailboxReceiver {
    slot: std::sync::Arc<MailboxInner>,
}

struct MailboxInner {
    data: Mutex<Option<CompletedFrame>>,
    /// Set to true when the sender is dropped (engine crashed/stopped).
    closed: std::sync::atomic::AtomicBool,
}

pub(crate) fn mailbox() -> (MailboxSender, MailboxReceiver) {
    let inner = std::sync::Arc::new(MailboxInner {
        data: Mutex::new(None),
        closed: std::sync::atomic::AtomicBool::new(false),
    });
    (
        MailboxSender { slot: inner.clone() },
        MailboxReceiver { slot: inner },
    )
}

impl MailboxSender {
    /// Store a frame, overwriting any previous unread frame.
    pub(crate) fn send(&self, frame: CompletedFrame) {
        let mut lock = self.slot.data.lock().unwrap();
        *lock = Some(frame);
    }
}

impl Drop for MailboxSender {
    fn drop(&mut self) {
        self.slot.closed.store(true, std::sync::atomic::Ordering::Release);
    }
}

impl MailboxReceiver {
    /// Take the latest frame if available. Non-blocking.
    pub(crate) fn try_recv(&self) -> Option<CompletedFrame> {
        let mut lock = self.slot.data.lock().unwrap();
        lock.take()
    }

    /// Returns true if the sender has been dropped (engine stopped/crashed).
    pub(crate) fn is_closed(&self) -> bool {
        self.slot.closed.load(std::sync::atomic::Ordering::Acquire)
    }
}

// ── Target channel: bounded mpsc for RenderTarget ───────────────

pub(crate) use mpsc::SyncSender as TargetSender;
pub(crate) use mpsc::Receiver as TargetReceiver;

use crate::types::RenderTarget;

/// Create a bounded-capacity channel for render targets.
/// Capacity 2 = double buffering.
pub(crate) fn target_channel() -> (TargetSender<RenderTarget>, TargetReceiver<RenderTarget>) {
    mpsc::sync_channel(2)
}
```

- [ ] **Step 2: Create `src/state_exchange.rs`**

```rust
use arc_swap::ArcSwap;
use std::sync::Arc;

/// Write end of the state exchange. Publishes new snapshots.
pub(crate) struct StateWriter<T> {
    swap: Arc<ArcSwap<T>>,
}

/// Read end of the state exchange. Reads the latest snapshot.
pub(crate) struct StateReader<T> {
    swap: Arc<ArcSwap<T>>,
}

pub(crate) fn state_exchange<T: Default>() -> (StateWriter<T>, StateReader<T>) {
    let swap = Arc::new(ArcSwap::from_pointee(T::default()));
    (
        StateWriter { swap: swap.clone() },
        StateReader { swap },
    )
}

impl<T> StateWriter<T> {
    /// Publish a new state snapshot. Lock-free.
    pub(crate) fn publish(&self, value: T) {
        self.swap.store(Arc::new(value));
    }
}

impl<T: Clone> StateReader<T> {
    /// Read the latest state snapshot. Lock-free.
    /// Returns a clone of the current value.
    pub(crate) fn read(&self) -> T {
        (**self.swap.load()).clone()
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add src/mailbox.rs src/state_exchange.rs
git commit -m "feat(v2): add mailbox and state exchange primitives"
```

---

### Task 4: Render Target Pool

**Files:**
- Create: `src/render_targets.rs`

This module creates and manages the two VkImages, their views, memory, the timeline semaphore, and generates the correct barriers based on queue family configuration.

- [ ] **Step 1: Create `src/render_targets.rs`**

```rust
use ash::vk;

/// Manages two render target VkImages, their views, device memory,
/// and the shared timeline semaphore.
pub(crate) struct RenderTargetPool {
    device: ash::Device,
    images: [vk::Image; 2],
    image_views: [vk::ImageView; 2],
    memory: [vk::DeviceMemory; 2],
    timeline: vk::Semaphore,
    timeline_value: u64,
    extent: vk::Extent2D,
    format: vk::Format,
    host_queue_family: u32,
    engine_queue_family: u32,
    cross_family: bool,
}

impl RenderTargetPool {
    /// Create a new pool with two render target images at the given extent.
    ///
    /// # Safety
    /// `device` must be valid. `physical_device` must match `device`.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        extent: vk::Extent2D,
        format: vk::Format,
        host_queue_family: u32,
        engine_queue_family: u32,
    ) -> Self {
        let cross_family = host_queue_family != engine_queue_family;

        // Always EXCLUSIVE sharing mode. Cross-family ownership transfers
        // are handled via explicit VkImageMemoryBarrier2 barriers.
        let create_image = |dev: &ash::Device| -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT    // engine writes
                    | vk::ImageUsageFlags::SAMPLED           // host samples
                    | vk::ImageUsageFlags::TRANSFER_DST,     // for clear-to-black
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let image = dev.create_image(&image_info, None).unwrap();

            let mem_reqs = dev.get_image_memory_requirements(image);
            let mem_props = instance.get_physical_device_memory_properties(physical_device);
            let mem_type_index = find_memory_type(
                &mem_props,
                mem_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(mem_reqs.size)
                .memory_type_index(mem_type_index);
            let memory = dev.allocate_memory(&alloc_info, None).unwrap();
            dev.bind_image_memory(image, memory, 0).unwrap();

            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let view = dev.create_image_view(&view_info, None).unwrap();

            (image, memory, view)
        };

        let (img0, mem0, view0) = create_image(device);
        let (img1, mem1, view1) = create_image(device);

        // Timeline semaphore, initial value 0
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let timeline = device.create_semaphore(&sem_info, None).unwrap();

        Self {
            device: device.clone(),
            images: [img0, img1],
            image_views: [view0, view1],
            memory: [mem0, mem1],
            timeline,
            timeline_value: 0,
            extent,
            format,
            host_queue_family,
            engine_queue_family,
            cross_family,
        }
    }

    /// Generate the next RenderTarget to send to the engine.
    /// `index` is 0 or 1 (which of the two images).
    pub(crate) fn make_target(&mut self, index: usize) -> crate::types::RenderTarget {
        let wait_value = self.timeline_value;
        self.timeline_value += 1;
        let signal_value = self.timeline_value;

        let acquire_barrier = if self.cross_family {
            Some(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(self.host_queue_family)
                    .dst_queue_family_index(self.engine_queue_family)
                    .image(self.images[index])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            )
        } else {
            None
        };

        let release_barrier = if self.cross_family {
            Some(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(self.engine_queue_family)
                    .dst_queue_family_index(self.host_queue_family)
                    .image(self.images[index])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    }),
            )
        } else {
            None
        };

        crate::types::RenderTarget {
            image: self.images[index],
            image_view: self.image_views[index],
            extent: self.extent,
            format: self.format,
            timeline: self.timeline,
            wait_value,
            signal_value,
            acquire_barrier,
            release_barrier,
        }
    }

    pub(crate) fn timeline(&self) -> vk::Semaphore {
        self.timeline
    }

    pub(crate) fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub(crate) fn format(&self) -> vk::Format {
        self.format
    }

    pub(crate) fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    /// Destroy all Vulkan resources.
    ///
    /// # Safety
    /// Must be called after `vkDeviceWaitIdle`. No GPU work may reference these resources.
    pub(crate) unsafe fn destroy(&mut self) {
        for &view in &self.image_views {
            self.device.destroy_image_view(view, None);
        }
        for &image in &self.images {
            self.device.destroy_image(image, None);
        }
        for &mem in &self.memory {
            self.device.free_memory(mem, None);
        }
        self.device.destroy_semaphore(self.timeline, None);
    }
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> u32 {
    for i in 0..mem_props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return i;
        }
    }
    panic!("failed to find suitable memory type");
}
```

- [ ] **Step 2: Commit**

```bash
git add src/render_targets.rs
git commit -m "feat(v2): add render target pool with timeline semaphore"
```

---

### Task 5: Event Types + Input Forwarding

**Files:**
- Create: `src/event.rs` (replaces the old v1 event.rs)

- [ ] **Step 1: Create `src/event.rs`**

```rust
use ash::vk;

/// Input events forwarded from the host to the engine thread.
#[derive(Debug)]
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

/// Button state for pointer events.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointerButtons {
    pub primary: bool,
    pub secondary: bool,
    pub middle: bool,
}
```

- [ ] **Step 2: Commit**

```bash
git add src/event.rs
git commit -m "feat(v2): add engine event types for input forwarding"
```

---

### Task 6: Compositor (egui rendering + engine viewport as User texture)

**Files:**
- Create: `src/compositor.rs`

No new shaders needed. The engine viewport texture is registered as an egui User texture — egui's existing `vert.spv`/`frag.spv` shaders (in `src/shaders/spv/`) draws it when the user places an `egui::Image` widget.

**Note:** The existing v1 code in `renderer.rs` and `viewport_context.rs` contains working egui Vulkan rendering code (descriptor sets, pipeline, vertex/index buffers, texture management). The compositor should adapt and simplify this code for the v2 single-viewport compositing model.

- [ ] **Step 1: Create `src/compositor.rs`**

The compositor manages:
- Swapchain creation and recreation
- egui texture management (managed textures for font atlas + User textures)
- The engine viewport registered as an egui User texture (descriptor set pointing at engine render target or black image)
- Render pass + egui pipeline
- Per-frame command buffer recording

Key structures:

```rust
use ash::vk;
use std::collections::HashMap;

/// Per-texture Vulkan resources for egui managed textures.
pub(crate) struct ManagedTexture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    image_view: vk::ImageView,
    descriptor_set: vk::DescriptorSet,
}

/// The host-side compositing pipeline.
/// Owns swapchain, egui rendering resources, and the engine viewport texture slot.
pub(crate) struct Compositor {
    // Vulkan handles (borrowed from VulkanContext)
    device: ash::Device,
    host_queue: vk::Queue,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,

    // Swapchain
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::Format,
    present_mode: vk::PresentModeKHR,

    // Sync
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    // Render pass + framebuffers
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    // Command pool + buffers
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    // egui pipeline (adapted from v1 renderer.rs)
    egui_pipeline_layout: vk::PipelineLayout,
    egui_pipeline: vk::Pipeline,
    egui_descriptor_pool: vk::DescriptorPool,
    egui_descriptor_set_layout: vk::DescriptorSetLayout,
    egui_sampler: vk::Sampler,

    // Engine viewport — registered as egui User texture
    // The descriptor set is updated to point at the engine render target
    // or the black placeholder image.
    engine_viewport_texture_id: egui::TextureId,
    engine_viewport_descriptor_set: vk::DescriptorSet,
    black_image: vk::Image,
    black_image_view: vk::ImageView,
    black_image_memory: vk::DeviceMemory,

    // egui managed textures (font atlas, etc.)
    managed_textures: HashMap<egui::TextureId, ManagedTexture>,

    // Vertex/index buffers (per frame-in-flight)
    vertex_buffers: Vec<vk::Buffer>,
    vertex_buffer_memory: Vec<vk::DeviceMemory>,
    index_buffers: Vec<vk::Buffer>,
    index_buffer_memory: Vec<vk::DeviceMemory>,
}
```

The implementation should adapt the existing egui rendering logic from `src/renderer.rs` (lines ~400-900 handle texture updates, vertex buffer uploads, and egui draw command recording) and `src/viewport_context.rs` (swapchain management, render pass setup).

Key methods:

```rust
impl Compositor {
    /// Create the compositor. Creates swapchain, render pass, egui pipeline,
    /// the black placeholder image, and registers the engine viewport
    /// as an egui User texture.
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        host_queue: vk::Queue,
        host_queue_family: u32,
        surface: vk::SurfaceKHR,
        present_mode: vk::PresentModeKHR,
    ) -> Self { /* ... */ }

    /// Update the engine viewport descriptor to point at a new image view.
    /// Called when a new CompletedFrame arrives. The image must be in
    /// SHADER_READ_ONLY_OPTIMAL layout.
    pub(crate) unsafe fn set_engine_viewport(&mut self, image_view: vk::ImageView) {
        // vkUpdateDescriptorSets on engine_viewport_descriptor_set
        // to point at the new image_view with egui_sampler
    }

    /// Set the engine viewport to the black placeholder image.
    /// Called on engine crash/stop.
    pub(crate) unsafe fn set_engine_viewport_black(&mut self) {
        self.set_engine_viewport(self.black_image_view);
    }

    /// The egui::TextureId for the engine viewport. Users reference
    /// this in their ui closure to place the viewport widget.
    pub(crate) fn engine_viewport_texture_id(&self) -> egui::TextureId {
        self.engine_viewport_texture_id
    }

    /// Record and submit a frame:
    /// 1. Update egui managed textures (font atlas deltas)
    /// 2. Upload vertex/index buffers
    /// 3. Record render pass with egui draw commands
    ///    (engine viewport is drawn by egui as a User texture)
    /// 4. Submit to host queue with timeline semaphore signal
    /// 5. Present
    pub(crate) unsafe fn render_frame(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        scale_factor: f32,
        screen_size: [u32; 2],
        // Timeline semaphore to signal after compositing
        // (so engine knows host is done with the render target)
        timeline: vk::Semaphore,
        signal_value: u64,
    ) -> Result<(), vk::Result> { /* ... */ }

    /// Recreate swapchain (e.g., on window resize).
    pub(crate) unsafe fn recreate_swapchain(
        &mut self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) { /* ... */ }

    /// Destroy all Vulkan resources.
    pub(crate) unsafe fn destroy(&mut self) { /* ... */ }
}
```

This is the most code-heavy task. The implementation should pull egui rendering logic from the existing `renderer.rs` and simplify it (no multi-viewport, no `Allocator` generic — use raw `vkAllocateMemory` for internal buffers since they're small and few).

- [ ] **Step 2: Compile-test the module structure**

Temporarily wire `mod compositor;` into `lib.rs` and run `cargo check` to catch type errors.

- [ ] **Step 3: Commit**

```bash
git add src/compositor.rs
git commit -m "feat(v2): add compositor for egui rendering with engine viewport as User texture"
```

---

### Task 7: Engine Thread

**Files:**
- Create: `src/engine_thread.rs`

Depends on: Tasks 1-5 (types, engine trait, mailbox, render_targets, event)

- [ ] **Step 1: Create `src/engine_thread.rs`**

```rust
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use crate::engine::EngineRenderer;
use crate::event::EngineEvent;
use crate::mailbox::{MailboxSender, TargetReceiver};
use crate::state_exchange::{StateReader, StateWriter};
use crate::types::EngineContext;

// Health encoding for AtomicU8
pub(crate) const HEALTH_STARTING: u8 = 0;
pub(crate) const HEALTH_RUNNING: u8 = 1;
pub(crate) const HEALTH_STOPPED: u8 = 2;
pub(crate) const HEALTH_CRASHED: u8 = 3;

/// Shared health state between engine thread and host.
pub(crate) struct EngineHealthState {
    pub(crate) health: AtomicU8,
    pub(crate) crash_message: std::sync::Mutex<Option<String>>,
    pub(crate) frames_delivered: std::sync::atomic::AtomicU64,
    pub(crate) last_frame_time_ns: std::sync::atomic::AtomicU64,
}

impl EngineHealthState {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            health: AtomicU8::new(HEALTH_STARTING),
            crash_message: std::sync::Mutex::new(None),
            frames_delivered: std::sync::atomic::AtomicU64::new(0),
            last_frame_time_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

/// Spawn the engine thread. Returns the JoinHandle.
pub(crate) fn spawn_engine_thread<E: EngineRenderer>(
    mut engine: E,
    ctx: EngineContext,
    targets_rx: TargetReceiver,
    frames_tx: MailboxSender,
    events_rx: mpsc::Receiver<EngineEvent>,
    ui_state_reader: StateReader<E::UiState>,
    engine_state_writer: StateWriter<E::EngineState>,
    health: Arc<EngineHealthState>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("egui-ash-engine".into())
        .spawn(move || {
            let result =
                std::panic::catch_unwind(AssertUnwindSafe(|| {
                    engine.init(ctx);
                    health.health.store(HEALTH_RUNNING, Ordering::Release);

                    let mut engine_state = E::EngineState::default();

                    while let Ok(target) = targets_rx.recv() {
                        // Drain input events
                        for event in events_rx.try_iter() {
                            engine.handle_event(event);
                        }

                        // Read latest UI state
                        let ui_state = ui_state_reader.read();

                        let frame_start = Instant::now();
                        let frame = engine.render(target, &ui_state, &mut engine_state);
                        let frame_time = frame_start.elapsed();

                        // Publish engine state
                        engine_state_writer.publish(engine_state.clone());

                        // Update stats
                        health.frames_delivered.fetch_add(1, Ordering::Relaxed);
                        health.last_frame_time_ns.store(
                            frame_time.as_nanos() as u64,
                            Ordering::Relaxed,
                        );

                        // Send completed frame
                        frames_tx.send(frame);
                    }

                    engine.destroy();
                }));

            match result {
                Ok(()) => {
                    health.health.store(HEALTH_STOPPED, Ordering::Release);
                }
                Err(panic_info) => {
                    let message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    *health.crash_message.lock().unwrap() = Some(message);
                    health.health.store(HEALTH_CRASHED, Ordering::Release);
                }
            }
        })
        .expect("failed to spawn engine thread")
}
```

- [ ] **Step 2: Commit**

```bash
git add src/engine_thread.rs
git commit -m "feat(v2): add engine thread with catch_unwind fault isolation"
```

---

### Task 8: Host Orchestrator

**Files:**
- Create: `src/host.rs`

Depends on: Tasks 6-7 (compositor, engine_thread)

The Host struct owns all host-side resources and implements the per-frame protocol described in the spec.

- [ ] **Step 1: Create `src/host.rs`**

```rust
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use ash::vk;
use egui_winit::winit;
use raw_window_handle::HasDisplayHandle as _;

use crate::compositor::Compositor;
use crate::engine::EngineRenderer;
use crate::engine_thread::{self, EngineHealthState, HEALTH_CRASHED, HEALTH_RUNNING, HEALTH_STOPPED};
use crate::event::EngineEvent;
use crate::mailbox::{self, MailboxReceiver};
use crate::render_targets::RenderTargetPool;
use crate::state_exchange::{self, StateReader, StateWriter};
use crate::types::*;

/// Handle for restarting the engine from the UI closure.
pub struct EngineHandle<E: EngineRenderer> {
    restart_request: std::cell::RefCell<Option<E>>,
    health: Arc<EngineHealthState>,
}

impl<E: EngineRenderer> EngineHandle<E> {
    pub(crate) fn new(health: Arc<EngineHealthState>) -> Self {
        Self {
            restart_request: std::cell::RefCell::new(None),
            health,
        }
    }

    pub fn restart(&self, engine: E) -> Result<(), EngineRestartError> {
        let h = self.health.health.load(std::sync::atomic::Ordering::Acquire);
        if h == engine_thread::HEALTH_RUNNING || h == engine_thread::HEALTH_STARTING {
            return Err(EngineRestartError::StillRunning);
        }
        *self.restart_request.borrow_mut() = Some(engine);
        Ok(())
    }

    pub(crate) fn take_restart_request(&self) -> Option<E> {
        self.restart_request.borrow_mut().take()
    }
}

/// The host orchestrator. Owns all host-side resources.
pub(crate) struct Host<E: EngineRenderer> {
    // Vulkan context (references to user-owned objects)
    vulkan: VulkanContext,

    // Window (stored here so it lives as long as the Host)
    window: winit::window::Window,

    // Compositor
    compositor: Compositor,

    // Render target management
    render_target_pool: RenderTargetPool,
    target_tx: Option<mpsc::SyncSender<crate::types::RenderTarget>>,
    frame_rx: MailboxReceiver,
    // Track which image index (0 or 1) is currently with the engine
    // and which was last composited. None = no frame composited yet.
    engine_target_index: usize,       // index sent to engine
    composited_target_index: Option<usize>,  // index currently in compositor
    pending_resize: Option<vk::Extent2D>,

    // Engine events
    event_tx: mpsc::SyncSender<EngineEvent>,

    // State exchange
    ui_state_writer: StateWriter<E::UiState>,
    engine_state_reader: StateReader<E::EngineState>,
    ui_state: E::UiState,  // persisted across frames

    // Engine thread
    engine_thread: Option<std::thread::JoinHandle<()>>,
    health: Arc<EngineHealthState>,

    // Engine handle (shared with UI closure via reference)
    engine_handle: EngineHandle<E>,

    // Exit signal
    exit_tx: Option<std::sync::mpsc::Sender<std::process::ExitCode>>,

    // egui
    context: egui::Context,
    egui_winit_state: egui_winit::State,
}

impl<E: EngineRenderer> Host<E> {
    /// Create the host and spawn the engine thread.
    ///
    /// # Panics
    /// Panics if the device does not support timeline semaphores.
    pub(crate) unsafe fn new(
        vulkan: VulkanContext,
        engine: E,
        options: &RunOption,
        window: winit::window::Window,
        event_loop: &winit::event_loop::ActiveEventLoop,
        exit_tx: std::sync::mpsc::Sender<std::process::ExitCode>,
    ) -> Self {
        // Validate timeline semaphore support
        let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut timeline_features);
        vulkan.instance.get_physical_device_features2(
            vulkan.physical_device,
            &mut features2,
        );
        assert!(
            timeline_features.timeline_semaphore == vk::TRUE,
            "egui-ash v2 requires timelineSemaphore device feature (Vulkan 1.2+)"
        );

        let context = egui::Context::default();
        context.set_embed_viewports(false);

        // Create surface
        let surface = ash_window::create_surface(
            &vulkan.entry,
            &vulkan.instance,
            window.display_handle().unwrap().into(),
            window.window_handle().unwrap().into(),
            None,
        )
        .unwrap();

        // Create compositor
        let compositor = Compositor::new(
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            vulkan.host_queue,
            vulkan.host_queue_family_index,
            surface,
            options.present_mode,
        );

        // Determine initial extent
        let window_size = window.inner_size();
        let initial_extent = vk::Extent2D {
            width: window_size.width.max(1),
            height: window_size.height.max(1),
        };

        let format = vk::Format::B8G8R8A8_SRGB;

        // Create render target pool
        let render_target_pool = RenderTargetPool::new(
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            initial_extent,
            format,
            vulkan.host_queue_family_index,
            vulkan.engine_queue_family_index,
        );

        // Channels
        let (target_tx, target_rx) = crate::mailbox::target_channel();
        let (frame_tx, frame_rx) = crate::mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::sync_channel(64);

        // State exchange
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // Health
        let health = EngineHealthState::new();

        // Engine context
        let engine_ctx = EngineContext {
            device: vulkan.device.clone(),
            queue: vulkan.engine_queue,
            queue_family_index: vulkan.engine_queue_family_index,
            initial_extent,
            format,
        };

        // Spawn engine thread
        let engine_thread = engine_thread::spawn_engine_thread(
            engine,
            engine_ctx,
            target_rx,
            frame_tx,
            event_rx,
            ui_state_reader,
            engine_state_writer,
            health.clone(),
        );

        // egui-winit state
        let egui_winit_state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            None,
            None,
            None,
        );

        let engine_handle = EngineHandle::new(health.clone());

        // Send first render target (index 0) to engine
        let mut pool = render_target_pool;
        let target = pool.make_target(0);
        target_tx.send(target).unwrap();

        Self {
            vulkan,
            window,
            compositor,
            render_target_pool: pool,
            target_tx: Some(target_tx),
            frame_rx,
            engine_target_index: 0,
            composited_target_index: None,
            pending_resize: None,
            event_tx,
            ui_state_writer,
            engine_state_reader,
            ui_state: E::UiState::default(),
            engine_thread: Some(engine_thread),
            health,
            engine_handle,
            exit_tx: Some(exit_tx),
            context,
            egui_winit_state,
        }
    }

    /// Run one frame: check engine, run egui, composite, present.
    pub(crate) unsafe fn frame(
        &mut self,
        ui: &mut impl FnMut(
            &egui::Context,
            &EngineStatus,
            &mut E::UiState,
            &E::EngineState,
            &EngineHandle<E>,
        ),
    ) {
        // ── 1. Check for new engine frame ──
        if let Some(completed) = self.frame_rx.try_recv() {
            // CPU wait on timeline (should be near-instant)
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(&[self.render_target_pool.timeline()])
                .values(&[completed.signal_value]);
            self.vulkan.device
                .wait_semaphores(&wait_info, u64::MAX)
                .unwrap();

            // Look up image view from the image handle
            let completed_index = self.engine_target_index;
            let image_view = self.render_target_pool.image_view(completed_index);

            // Update compositor to sample this image
            self.compositor.set_engine_viewport(image_view);

            // Reclaim the PREVIOUS composited target back to the engine
            if let Some(prev_index) = self.composited_target_index {
                let reclaim_target = self.render_target_pool.make_target(prev_index);
                if let Some(ref tx) = self.target_tx {
                    let _ = tx.send(reclaim_target);
                }
                // The engine now has prev_index, update tracking
                self.engine_target_index = prev_index;
            }

            self.composited_target_index = Some(completed_index);
        }

        // If engine crashed and no frame was composited, show black
        if self.frame_rx.is_closed() && self.composited_target_index.is_none() {
            self.compositor.set_engine_viewport_black();
        }

        // ── 2. Build engine status ──
        let health_val = self.health.health.load(std::sync::atomic::Ordering::Acquire);
        let engine_health = match health_val {
            engine_thread::HEALTH_STARTING => EngineHealth::Starting,
            engine_thread::HEALTH_RUNNING => EngineHealth::Running,
            engine_thread::HEALTH_STOPPED => EngineHealth::Stopped,
            engine_thread::HEALTH_CRASHED => EngineHealth::Crashed {
                message: self.health.crash_message.lock().unwrap()
                    .clone().unwrap_or_default(),
            },
            _ => unreachable!(),
        };

        if engine_health.is_crashed() {
            self.compositor.set_engine_viewport_black();
        }

        let status = EngineStatus {
            health: engine_health,
            viewport_texture_id: self.compositor.engine_viewport_texture_id(),
            frames_delivered: self.health.frames_delivered
                .load(std::sync::atomic::Ordering::Relaxed),
            last_frame_time: {
                let ns = self.health.last_frame_time_ns
                    .load(std::sync::atomic::Ordering::Relaxed);
                if ns > 0 { Some(Duration::from_nanos(ns)) } else { None }
            },
        };

        // ── 3. Read latest engine state ──
        let engine_state = self.engine_state_reader.read();

        // ── 4. Run egui frame lifecycle ──
        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let full_output = self.context.run(raw_input, |ctx| {
            ui(ctx, &status, &mut self.ui_state, &engine_state, &self.engine_handle);
        });

        // Publish UI state after closure returns
        self.ui_state_writer.publish(self.ui_state.clone());

        // Handle platform output (clipboard, cursor, etc.)
        self.egui_winit_state.handle_platform_output(
            &self.window,
            full_output.platform_output,
        );

        // ── 5. Handle restart request ──
        if let Some(new_engine) = self.engine_handle.take_restart_request() {
            self.restart_engine(new_engine);
        }

        // ── 6. Tesselate and render ──
        let clipped_primitives = self.context.tessellate(
            full_output.shapes,
            full_output.pixels_per_point,
        );

        let screen_size = self.window.inner_size();
        let timeline = self.render_target_pool.timeline();
        // The host signals the next even timeline value after compositing
        // so the engine knows it's safe to reuse the render target.
        let signal_value = /* next timeline value from pool */ 0; // pool tracks this

        let _ = self.compositor.render_frame(
            clipped_primitives,
            full_output.textures_delta,
            full_output.pixels_per_point,
            [screen_size.width, screen_size.height],
            timeline,
            signal_value,
        );

        // ── 7. Request next frame ──
        if full_output.repaint_after.is_zero() {
            self.window.request_redraw();
        }
    }

    /// Forward a winit event to egui-winit.
    pub(crate) fn handle_window_event(
        &mut self,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let response = self.egui_winit_state.on_window_event(
            &self.window,
            event,
        );
        response.consumed
    }

    /// Programmatic exit (called from UI via exit signal).
    pub(crate) fn request_exit(&mut self) {
        if let Some(tx) = self.exit_tx.take() {
            let _ = tx.send(std::process::ExitCode::SUCCESS);
        }
    }

    /// Restart engine with a new instance.
    unsafe fn restart_engine(&mut self, engine: E) {
        // 1. Drop target sender to close the channel
        self.target_tx.take();

        // 2. Wait for engine thread to exit
        if let Some(handle) = self.engine_thread.take() {
            let _ = handle.join();
        }

        // 3. Wait for all GPU work to complete
        self.vulkan.device.device_wait_idle().unwrap();

        // 4. Destroy old render targets, create fresh ones
        self.render_target_pool.destroy();
        let extent = self.render_target_pool.extent();
        let format = self.render_target_pool.format();
        self.render_target_pool = RenderTargetPool::new(
            &self.vulkan.instance,
            &self.vulkan.device,
            self.vulkan.physical_device,
            extent,
            format,
            self.vulkan.host_queue_family_index,
            self.vulkan.engine_queue_family_index,
        );

        // 5. Reset compositor to black
        self.compositor.set_engine_viewport_black();
        self.composited_target_index = None;

        // 6. Create new channels
        let (target_tx, target_rx) = crate::mailbox::target_channel();
        let (frame_tx, frame_rx) = crate::mailbox::mailbox();
        let (event_tx, event_rx) = mpsc::sync_channel(64);

        // 7. New state exchange
        let (ui_state_writer, ui_state_reader) = state_exchange::state_exchange::<E::UiState>();
        let (engine_state_writer, engine_state_reader) =
            state_exchange::state_exchange::<E::EngineState>();

        // 8. Reset health
        let health = EngineHealthState::new();
        self.engine_handle = EngineHandle::new(health.clone());

        // 9. Spawn new engine thread
        let engine_ctx = EngineContext {
            device: self.vulkan.device.clone(),
            queue: self.vulkan.engine_queue,
            queue_family_index: self.vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
        };
        let engine_thread = engine_thread::spawn_engine_thread(
            engine, engine_ctx, target_rx, frame_tx, event_rx,
            ui_state_reader, engine_state_writer, health.clone(),
        );

        // 10. Send first target
        let target = self.render_target_pool.make_target(0);
        target_tx.send(target).unwrap();

        // 11. Update Host fields
        self.target_tx = Some(target_tx);
        self.frame_rx = frame_rx;
        self.event_tx = event_tx;
        self.ui_state_writer = ui_state_writer;
        self.engine_state_reader = engine_state_reader;
        self.engine_thread = Some(engine_thread);
        self.health = health;
        self.engine_target_index = 0;
    }

    /// Clean shutdown.
    pub(crate) unsafe fn destroy(&mut self) {
        // Signal engine shutdown
        let _ = self.event_tx.try_send(EngineEvent::Shutdown);
        // Drop target sender to close the channel
        self.target_tx.take();

        // Wait for engine thread
        if let Some(handle) = self.engine_thread.take() {
            let _ = handle.join();
        }

        self.vulkan.device.device_wait_idle().unwrap();
        self.render_target_pool.destroy();
        self.compositor.destroy();
    }
}
```

Key fixes from review:
- **Window stored in Host** (C3) — `window` field added, threaded to `on_window_event` and `take_egui_input`
- **Correct `on_window_event` signature** (C4) — passes `&self.window`, not `&self.context`
- **Full egui frame lifecycle** (C5) — `take_egui_input` → `context.run()` → `tessellate` → `render_frame`
- **UiState persisted across frames** (I1) — `ui_state: E::UiState` field, not recreated each frame
- **`restart()` validates health** (I3) — checks `HEALTH_RUNNING`/`HEALTH_STARTING` before accepting
- **Timeline semaphore feature validation** (I4) — asserts `timelineSemaphore` in `Host::new()`
- **`window.request_redraw()`** (I5) — called based on `repaint_after`
- **Programmatic exit** (I6) — `request_exit()` method + `exit_tx` field
- **CompletedFrame → image view lookup** (I7) — tracks `engine_target_index` to map completed frames to pool indices
- **Render target reclaim protocol** (I8) — full implementation: reclaims previous composited target when receiving a new frame
- **Restart fully implemented** — `restart_engine()` method: wait idle, destroy old pool, create fresh, spawn new thread

- [ ] **Step 2: Commit**

```bash
git add src/host.rs
git commit -m "feat(v2): add host orchestrator with full frame protocol"
```

---

### Task 9: Entry Point (`run.rs`)

**Files:**
- Rewrite: `src/run.rs`
- Rewrite: `src/lib.rs`

Depends on: Task 8 (host)

- [ ] **Step 1: Rewrite `src/run.rs`**

```rust
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
///
/// Creates the winit event loop, builds the host compositor, spawns the
/// engine thread, and enters the event loop. Returns when the user
/// closes the window or sends an exit signal.
///
/// # Requirements
/// - Vulkan 1.2+ with `timelineSemaphore` feature enabled on the device
/// - `panic = "unwind"` for engine fault isolation
/// - Two separate VkQueues in `VulkanContext`
///
/// # Ownership
/// `run()` does NOT destroy VkInstance or VkDevice. The caller must
/// destroy these after `run()` returns.
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
    vulkan: Option<VulkanContext>,  // Option so we can take() ownership
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
        let window_attributes = WindowAttributes::default()
            .with_visible(true)
            .with_title("egui-ash");
        let window = event_loop.create_window(window_attributes).unwrap();

        let vulkan = self.vulkan.take().expect("VulkanContext already consumed");
        let engine = self.engine.take().expect("Engine already consumed");

        // Window ownership moves into Host — it lives as long as Host does
        let host = unsafe {
            Host::new(vulkan, engine, &self.options, window, event_loop, self.exit_tx.clone())
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
            unsafe { host.destroy(); }
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
        unsafe { host.frame(&mut self.ui); }
    }
}
```

- [ ] **Step 2: Rewrite `src/lib.rs`**

```rust
mod compositor;
mod engine;
mod engine_thread;
pub mod event;
mod host;
mod mailbox;
mod render_targets;
mod run;
mod state_exchange;
mod types;
#[cfg(feature = "persistence")]
pub mod storage;

pub use egui_winit::winit;
pub use raw_window_handle;

pub use engine::*;
pub use event::{EngineEvent, PointerButtons};
pub use host::EngineHandle;
pub use run::*;
pub use types::*;
```

- [ ] **Step 3: Delete v1 files**

Remove the old v1 modules that are no longer referenced:
- `src/app.rs`
- `src/allocator.rs`
- `src/gpu_allocator.rs`
- `src/integration.rs`
- `src/renderer.rs`
- `src/viewport_context.rs`
- `src/utils.rs`

- [ ] **Step 4: Run `cargo check`**

Expected: compilation errors in the compositor (unfinished implementation) and host (skeleton code). The module structure and public API types should resolve correctly.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(v2): rewrite run.rs and lib.rs, remove v1 modules"
```

---

### Task 10: Integration — Wire Everything Together + Minimal Example

**Files:**
- Modify: `src/compositor.rs` (flesh out the implementation)
- Modify: `src/host.rs` (flesh out frame protocol, resize, restart)
- Modify: `src/run.rs` (window management refinements)
- Create: `examples/v2_simple/main.rs`
- Modify: `Cargo.toml` (update dev-dependencies, add example)

This is the integration task where all the pieces get wired together and tested with a real Vulkan instance.

- [ ] **Step 1: Complete `compositor.rs` implementation**

Adapt the egui rendering pipeline from the existing v1 `renderer.rs` code:
- Port `create_render_pass`, `create_pipeline`, `create_descriptor_set_layout` from `viewport_context.rs`
- Port texture upload logic from `renderer.rs` `ManagedTextures`
- Port vertex/index buffer upload from `viewport_context.rs`
- Register the engine viewport as an egui User texture (descriptor set that points at the engine render target or black image). egui's existing `vert.spv`/`frag.spv` handles drawing it — no separate compositing pass needed.

Reference files:
- `src/viewport_context.rs:58-200` — render pass and pipeline creation
- `src/renderer.rs:400-700` — texture management
- `src/viewport_context.rs:400-600` — vertex/index buffer management and draw recording

- [ ] **Step 2: Wire lazy resize into host.rs**

In `Host::frame()`, compare the egui-reported viewport widget size with `render_target_pool.extent()`. If they differ:
1. Set `pending_resize = Some(new_extent)`
2. When the next `CompletedFrame` arrives, intercept the reclaim: destroy old pool, create new one at the pending extent, send first new target, clear pending flag
3. Send `EngineEvent::Resize` to the engine

- [ ] **Step 3: Wire persistence (deferred)**

Persistence (`storage.rs`) is kept as a module but not wired into the v2 Host in this pass. The `app_id` parameter is plumbed through but unused. Persistence integration (saving/restoring window state, egui memory) is deferred to a follow-up task.

- [ ] **Step 3: Create minimal `v2_simple` example**

```rust
// examples/v2_simple/main.rs
// Minimal example: solid color engine + egui UI
use ash::vk;
use egui_ash::{EngineRenderer, VulkanContext, RunOption, run};
use egui_ash::event::EngineEvent;

struct ColorEngine {
    device: Option<ash::Device>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

#[derive(Default, Clone)]
struct UiState {
    clear_color: [f32; 3],
}

#[derive(Default, Clone)]
struct EngState {
    frame_count: u64,
}

impl EngineRenderer for ColorEngine {
    type UiState = UiState;
    type EngineState = EngState;

    fn init(&mut self, ctx: egui_ash::EngineContext) {
        // Create command pool + buffer for this engine
        // ...
    }

    fn render(
        &mut self,
        target: egui_ash::RenderTarget,
        ui_state: &UiState,
        engine_state: &mut EngState,
    ) -> egui_ash::CompletedFrame {
        engine_state.frame_count += 1;
        // Record: clear target.image to ui_state.clear_color
        // Submit with timeline semaphore wait/signal
        target.complete()
    }

    fn handle_event(&mut self, _event: EngineEvent) {}
    fn destroy(&mut self) { /* cleanup */ }
}

fn main() {
    // Create Vulkan instance, device, two queues
    // ... (using ash directly, similar to examples/common/vkutils.rs)

    let vulkan = VulkanContext { /* ... */ };
    let engine = ColorEngine { /* ... */ };

    let exit_code = run(
        "v2_simple",
        vulkan,
        engine,
        RunOption::default(),
        move |ctx, status, ui_state, engine_state, engine_handle| {
            egui::SidePanel::left("controls").show(ctx, |ui| {
                ui.heading("Engine Control");
                ui.color_edit_button_rgb(&mut ui_state.clear_color);
                ui.label(format!("Engine frames: {}", engine_state.frame_count));
                if status.health.is_crashed() {
                    if ui.button("Restart").clicked() {
                        // engine_handle.restart(ColorEngine::new()).ok();
                    }
                }
            });
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    ui.available_size(),
                ));
            });
        },
    );
    std::process::exit(exit_code.report());
}
```

- [ ] **Step 4: Run the example**

```bash
cargo run --release --example v2_simple
```

Expected: Window appears with egui side panel and engine viewport showing a solid color controlled by the color picker. Frame counter increments in the UI.

- [ ] **Step 5: Test crash recovery**

Add a button to the example that triggers `panic!("test crash")` in the engine. Verify:
- Viewport goes black
- UI remains responsive
- `EngineHealth::Crashed` is displayed
- Restart button spawns a new engine

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(v2): complete integration, add v2_simple example"
```

---

## Implementation Notes

### Code to adapt from v1

The following v1 code should be adapted (not copied verbatim) for v2:

| v1 Source | v2 Destination | What to take |
|---|---|---|
| `viewport_context.rs:58-130` | `compositor.rs` | Render pass creation, surface format selection |
| `viewport_context.rs:130-200` | `compositor.rs` | Pipeline creation (vertex/fragment shaders, layout) |
| `viewport_context.rs:200-400` | `compositor.rs` | Swapchain creation/recreation |
| `viewport_context.rs:400-600` | `compositor.rs` | Vertex/index buffer management |
| `viewport_context.rs:600-800` | `compositor.rs` | Command buffer recording for egui draw |
| `renderer.rs:100-400` | `compositor.rs` | Descriptor set management, texture upload |
| `renderer.rs:400-700` | `compositor.rs` | ManagedTextures (egui font/user textures) |

### What's NOT adapted

- `renderer.rs` `ViewportOps` / `ViewportOpsImpl` — gone (no more type-erased closures)
- `renderer.rs` `EguiCommand` — gone (host always composites)
- `renderer.rs` `ImageRegistry` + mpsc — gone (engine renders to lent target)
- `integration.rs` multi-viewport management — v2 is single engine viewport
- `allocator.rs` — gone (engine manages own memory, host uses raw vkAllocateMemory)

### New dependency

Add to `Cargo.toml`:
```toml
arc-swap = "1.7"
```
