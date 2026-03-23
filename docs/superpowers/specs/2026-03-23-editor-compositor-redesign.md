# egui-ash v2: Editor/Compositor Architecture

**Date:** 2026-03-23
**Status:** Approved design, pending implementation

## Overview

Ground-up redesign of `egui-ash` from a turn-key application framework into an editor/compositor library. egui-ash becomes the **host** application shell managing the winit event loop and egui rendering. The user's ash rendering engine becomes a fault-isolated **guest** that receives a lent render target to draw into.

### Design Goals

- Library no longer owns VkInstance/VkDevice — user creates Vulkan objects and passes them in
- Engine runs on a dedicated thread with panic isolation
- egui and engine frame rates are independent, loosely synchronized
- If the engine crashes, the UI remains fully responsive
- State synchronization is explicit via typed shared state snapshots
- Event-driven input forwarding with graceful degradation

### Requirements

- **Vulkan 1.2** or `VK_KHR_timeline_semaphore` extension enabled on the device
- `timelineSemaphore` device feature must be enabled
- User must provide **two separate queues** (host and engine), from the same or different queue families
- `panic = "unwind"` in the application's `Cargo.toml` profile (required for `catch_unwind` fault isolation). If `panic = "abort"` is set, engine crashes will terminate the entire process.
- `run()` does **not** destroy VkInstance or VkDevice. The user is responsible for destroying these after `run()` returns.

## Architecture

### Three Actors

1. **Host** (library) — owns winit event loop, egui context, swapchain, compositing pipeline. Runs on the main thread.
2. **Engine** (user code) — implements `EngineRenderer` trait. Runs on a dedicated thread spawned by the library. Receives render targets, draws into them, returns completed frames.
3. **Mailbox** — the synchronization bridge. Two channels + two VkImages + a timeline semaphore. Fully owned by the host, with handles lent to the engine.

### Vulkan Resource Ownership

| Resource | Owner | Notes |
|---|---|---|
| VkInstance, VkDevice, VkPhysicalDevice | **User** (created before `run()`) | Library receives references |
| Swapchain, swapchain images | **Host** | For final presentation |
| Render target VkImages (x2) | **Host** | Lent to engine via mailbox |
| Timeline semaphore | **Host** | Shared with engine for GPU sync |
| Engine command pool + buffers | **Engine** | Engine's own queue submissions |
| egui render pass, pipeline, textures | **Host** | egui compositing internals |
| Engine-internal resources | **Engine** | Fully opaque to host |

## Synchronization Protocol

### Double-Buffered Mailbox with Timeline Semaphore

Two render target images: **A** and **B**. One timeline semaphore with a monotonically increasing counter.

```
Timeline values:  even = host done compositing,  odd = engine done rendering

Frame 0:  Host gives A to engine (value=0)
          Engine renders into A, signals value=1 (via vkQueueSubmit signal)
          Host waits on value=1 (via vkWaitSemaphores on CPU)
          Host composites A (samples as texture in host queue submit)
          Host signals value=2 (via vkQueueSubmit signal on host queue)
          Host gives A back to engine

Frame N:  While engine renders into A, host composites B (last good frame)
          Frames are decoupled — neither blocks the other
```

**Host-side signaling:** The host's compositing render pass submits to the host queue with a `VkTimelineSemaphoreSubmitInfo` that signals the next even value. This ensures the engine can safely wait on that value before reusing the render target. The host waits for the engine's signal value on the **CPU** via `vkWaitSemaphores` (not a queue wait) before transitioning the image to `SHADER_READ_ONLY_OPTIMAL` — this wait should be near-instant since the engine signals before sending the `CompletedFrame`.

### Queue Family Ownership Transfers

If host and engine queues are from **the same queue family**: no ownership transfers needed. The timeline semaphore provides all required synchronization.

If host and engine queues are from **different queue families**: the library inserts `VkImageMemoryBarrier` ownership transfers:
- **Engine → Host:** Engine's submit includes a release barrier (`srcQueueFamilyIndex=engine, dstQueueFamilyIndex=host`). Host's composite submit includes a matching acquire barrier.
- **Host → Engine:** Host's composite submit includes a release barrier. Engine's render submit includes a matching acquire barrier.

The library detects whether the queue families differ at initialization and conditionally includes these barriers. Same-family is the fast path (no barriers beyond the timeline semaphore).

### CPU-Side Channels

```
available_targets:  Host --> Engine   (bounded, capacity 2)
completed_frames:   Engine --> Host   (bounded, capacity 1, mailbox semantics)
engine_events:      Host --> Engine   (bounded, capacity 64)
```

The `completed_frames` channel uses **mailbox semantics**: if the engine produces a frame faster than the host consumes, the new frame replaces the old one. The host always gets the latest frame. This requires a custom single-slot mailbox primitive (not `std::sync::mpsc`, which does not support overwrite). Implementation: a `Mutex<Option<CompletedFrame>>` with a `Condvar`, or a lock-free triple buffer. The `Mutex` approach is simpler and the contention is negligible (one producer, one consumer, sub-microsecond critical section).

### Host Per-Frame Protocol

1. `try_recv` on `completed_frames` — got a new frame?
2. **Yes:** Wait on its timeline value (should already be signaled), composite it into the swapchain, then send the *previous* target back via `available_targets`
3. **No:** Composite the last good frame again (already in host-side texture, zero cost)

### Engine Per-Frame Protocol

1. `recv` on `available_targets` — blocks until a target is available (backpressure)
2. Wait on the target's timeline value (ensures host is done compositing it)
3. Record and submit command buffer rendering into the target
4. Signal the next timeline value
5. Send completed frame via `completed_frames`

### Input Forwarding

egui captures all input via winit. Engine-relevant events are forwarded through the `engine_events` channel using `try_send` — if the channel is full, events are dropped. This is correct behavior for real-time input (mouse moves coalesce naturally).

The `RenderTarget` struct carries the current viewport size, so the engine always gets correct dimensions even if it missed a resize event.

### Failure Modes

| Scenario | Host blocks? | UI responsive? | Viewport shows |
|---|---|---|---|
| Engine healthy | No | Yes | Live frames |
| Engine slow (frames) | No | Yes | Stale frame, updates when ready |
| Engine slow (input) | No | Yes | Live frames, some input dropped |
| Engine hung | No | Yes | Last good frame (frozen) |
| Engine crashed | No | Yes | Black + status indicator |
| GPU device lost | No | Briefly | Graceful shutdown |

## Public API

### User-Provided Vulkan Context

```rust
pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: ash::vk::PhysicalDevice,
    pub device: ash::Device,
    /// Queue used by the host for egui rendering and compositing.
    pub host_queue: ash::vk::Queue,
    pub host_queue_family_index: u32,
    /// Queue used by the engine for scene rendering. Must be a different
    /// VkQueue than host_queue (may be from the same or different queue family).
    pub engine_queue: ash::vk::Queue,
    pub engine_queue_family_index: u32,
}
```

The user creates both queues before calling `run()`. Using separate queues satisfies Vulkan's external synchronization requirement — each thread submits to its own queue without a mutex. If both queues are from the same family, no queue family ownership transfers are needed on render target images. If they differ, the library inserts the required `VkImageMemoryBarrier` ownership transfers automatically (see Synchronization Protocol).

### Engine Trait

```rust
pub trait EngineRenderer: Send + 'static {
    /// State written by UI, read by engine each frame
    type UiState: Send + Sync + Clone + Default;
    /// State written by engine, read by UI each frame
    type EngineState: Send + Sync + Clone + Default;

    /// Called once on the engine thread.
    fn init(&mut self, ctx: EngineContext);

    /// Render into the target. Reads latest UI state, writes engine state.
    fn render(
        &mut self,
        target: RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> CompletedFrame;

    /// Process an input event forwarded from the host.
    fn handle_event(&mut self, event: EngineEvent);

    /// Called on clean shutdown.
    fn destroy(&mut self);
}
```

### Entry Point

```rust
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
) -> ExitCode;
```

### Render Target Types

```rust
pub struct EngineContext {
    pub device: ash::Device,
    pub queue: ash::vk::Queue,
    pub queue_family_index: u32,
    pub initial_extent: ash::vk::Extent2D,
    /// Format of the render target images. Always B8G8R8A8_SRGB.
    pub format: ash::vk::Format,
}

pub struct RenderTarget {
    pub image: ash::vk::Image,
    pub image_view: ash::vk::ImageView,
    pub extent: ash::vk::Extent2D,
    pub format: ash::vk::Format,
    pub timeline: ash::vk::Semaphore,
    pub wait_value: u64,
    pub signal_value: u64,
    // Queue family ownership transfer barriers (if cross-family).
    // The engine must include these in its submit if non-null.
    pub(crate) acquire_barrier: Option<ash::vk::ImageMemoryBarrier2>,
    pub(crate) release_barrier: Option<ash::vk::ImageMemoryBarrier2>,
}

impl RenderTarget {
    /// Build a CompletedFrame after rendering. Captures the image handle
    /// and signal value so the engine cannot misreport them.
    pub fn complete(self) -> CompletedFrame {
        CompletedFrame {
            image: self.image,
            signal_value: self.signal_value,
            release_barrier: self.release_barrier,
        }
    }

    /// Returns the acquire barrier the engine must include in its
    /// command buffer before writing to the image, if queue families differ.
    /// Returns None if both queues share the same family.
    pub fn acquire_barrier(&self) -> Option<&ash::vk::ImageMemoryBarrier2> {
        self.acquire_barrier.as_ref()
    }

    /// Returns the release barrier the engine must include in its
    /// command buffer after writing to the image, if queue families differ.
    /// Returns None if both queues share the same family.
    pub fn release_barrier(&self) -> Option<&ash::vk::ImageMemoryBarrier2> {
        self.release_barrier.as_ref()
    }
}

pub struct CompletedFrame {
    pub(crate) image: ash::vk::Image,
    pub(crate) signal_value: u64,
    pub(crate) release_barrier: Option<ash::vk::ImageMemoryBarrier2>,
}
```

### Engine Status & Health

```rust
pub struct EngineStatus {
    pub health: EngineHealth,
    /// Texture ID for the engine viewport. Shows engine output when
    /// running, black when crashed/stopped.
    pub viewport_texture_id: egui::TextureId,
    pub frames_delivered: u64,
    pub last_frame_time: Option<Duration>,
}

pub enum EngineHealth {
    Starting,
    Running,
    Stopped,
    Crashed { message: String },
}

impl EngineHealth {
    pub fn is_alive(&self) -> bool;
    pub fn is_crashed(&self) -> bool;
}
```

### Engine Restart

```rust
pub struct EngineHandle<E: EngineRenderer> { /* ... */ }

impl<E: EngineRenderer> EngineHandle<E> {
    /// Restart the engine with a new instance.
    /// Only callable when health is Crashed or Stopped.
    pub fn restart(&self, engine: E) -> Result<(), EngineRestartError>;
}
```

### Input Events

```rust
pub enum EngineEvent {
    Pointer { position: [f32; 2], buttons: PointerButtons },
    Key { key: Key, pressed: bool, modifiers: Modifiers },
    Scroll { delta: [f32; 2] },
    Resize { extent: ash::vk::Extent2D },
    Focus(bool),
    /// Sent before clean shutdown. Gives the engine a chance to flush
    /// state before the available_targets channel closes.
    Shutdown,
}
```

**Coordinate system:** `EngineEvent::Pointer::position` is in **viewport-local physical pixels** — (0,0) is the top-left of the engine viewport widget, not the window. The user is responsible for any further coordinate transforms (e.g., to normalized device coordinates or world space).

```
```

### State Synchronization

Two user-defined associated types on `EngineRenderer`:

- `UiState` — written by the `ui` closure, read by the engine each frame
- `EngineState` — written by the engine, read by the `ui` closure each frame

Both are double-buffered by the library using `arc_swap::ArcSwap<T>`. Each side publishes its state via `ArcSwap::store()` (lock-free) and reads the other side's latest snapshot via `ArcSwap::load()` (lock-free `Arc` clone). No mutexes, no blocking, no contention.

**Mechanism:**
- Host side: after `ui` closure returns, library publishes the `UiState` via `arc_swap::ArcSwap::store(Arc::new(ui_state.clone()))`. Engine reads via `load()` at frame start.
- Engine side: after `render()` returns, library publishes the `EngineState` via `store()`. Host reads via `load()` before calling the `ui` closure.
- The `Clone` happens once per frame per side. For typical editor state (transforms, selections, flags), this is negligible.

Each side is authoritative over its own state type. For bidirectional values (e.g., position controlled by both slider and gizmo), the user defines an authority convention using a flag in `UiState`.

## Compositing Pipeline

### Host Per-Frame Render Sequence

```
1. Run egui  -->  2. Acquire swapchain image  -->  3. Composite  -->  4. Present
```

**Step 1 — Run egui:** Host calls the `ui` closure with latest `EngineState` snapshot. User places an `egui::Image` widget using `status.viewport_texture_id` wherever they want the engine viewport.

**Step 2 — Acquire swapchain image:** Standard `vkAcquireNextImageKHR`.

**Step 3 — Composite:** Single render pass, two layers:
1. **Engine layer:** Sample the engine's completed render target (or black image if crashed) as a texture. Draw at the position egui laid out.
2. **egui layer:** Standard egui vertex/index buffer draw on top. egui panels/windows naturally occlude the viewport.

**Step 4 — Present:** `vkQueuePresentKHR`. Reclaim the composited render target back to the engine.

### Resize Handling

When egui lays out the viewport at a different size than the current render targets:

1. Host marks the resize as pending and records the new extent
2. Host continues compositing the last good frame, scaled to the new size
3. When the engine returns its current `CompletedFrame`, the host does **not** send the target back. Instead it destroys both old VkImages.
4. Host creates new render target VkImages at the new extent
5. Host sends one new target to the engine via `available_targets` (the other is held for the next exchange)
6. Host sends `EngineEvent::Resize` via the input channel

This is a **lazy resize** — the host never blocks waiting for targets to become idle. It simply intercepts targets as they flow back through the mailbox. During the transition (typically 1-3 frames), the viewport shows the last frame scaled to fill.

## Engine Thread Lifecycle

### Startup Sequence

1. `run()` called
2. Host creates winit event loop, egui context, swapchain, render targets (x2), timeline semaphore, channels
3. Host spawns engine thread inside `catch_unwind` boundary
4. Engine thread calls `engine.init(EngineContext)`
5. Engine enters render loop: `recv` target -> drain events -> clone UiState -> render -> publish EngineState -> send frame
6. Host sends first render target, enters winit event loop

### Clean Shutdown

1. User closes window / exit signal sent
2. Host sends `EngineEvent::Shutdown` via the input channel
3. Host drops `available_targets` sender
4. Engine drains events, sees `Shutdown`, can flush state
5. Engine's next `recv()` returns `Err`, breaks loop
6. Engine calls `destroy()`
7. Engine thread exits
8. Host joins engine thread, waits device idle
9. Host destroys render targets, timeline semaphore, egui resources, swapchain

### Crash Behavior

1. Engine panics inside `catch_unwind`
2. Health set to `Crashed { message }`
3. Engine thread exits
4. Host detects closed channel on next `try_recv`
5. Host swaps viewport texture to pre-allocated **black image**
6. `EngineStatus` reflects crash in next `ui` call
7. UI remains fully responsive, can offer restart

### Crash Presentation

On engine crash, the viewport immediately goes **black** — not a frozen last frame. A frozen frame is misleading; black is an unambiguous signal. The `ui` closure receives `EngineHealth::Crashed { message }` and can overlay diagnostics, restart buttons, or any application-specific crash UI.

### Engine Restart

The `ui` closure captures an `EngineHandle<E>`. After a crash:
1. User clicks "Restart" in UI
2. `engine_handle.restart(MyEngine::new())` called
3. Library calls `vkDeviceWaitIdle` to ensure no in-flight GPU work
4. Fresh render targets created, new engine thread spawned
5. Engine goes through `init()` again

### Fault Isolation Boundaries

| Failure | Caught? | Behavior |
|---|---|---|
| Rust panic | Yes | Health -> Crashed, UI keeps running |
| Stack overflow | Yes (usually) | Same |
| `std::process::abort()` | No | Entire process dies |
| Segfault (unsafe) | No | Entire process dies |
| VK_ERROR_DEVICE_LOST | Not a panic | Engine detects via VkResult, should signal shutdown |
| Infinite loop | Not a crash | Host never blocks, keeps compositing last frame |

Fault isolation is best-effort for Rust-level failures. GPU faults and undefined behavior from `unsafe` are outside the containment boundary.

**`catch_unwind` implementation note:** `EngineRenderer` is `Send + 'static` but not `UnwindSafe`. The library wraps the engine in `AssertUnwindSafe` before passing to `catch_unwind`. This is sound because the engine thread owns all its state exclusively — no shared mutable references can be left in an inconsistent state visible to other threads. The only cross-thread communication is through channels and `ArcSwap`, which are inherently unwind-safe.

## Engine Lifecycle Guidance (README content)

The engine runs on a dedicated thread with panic isolation. If your engine panics, the egui UI continues running — the viewport shows black and `EngineStatus::health` becomes `Crashed`.

**Handling crashes:**
- Check `status.health` in your UI closure to detect crashes
- Use `EngineHandle::restart()` to spawn a fresh engine instance
- The crashed engine's thread has already exited — no cleanup needed from your side
- GPU resources from the crashed engine are **not** automatically freed. Design your engine's `Drop` impl to clean up even after a panic.

**Design considerations:**
- `EngineRenderer::destroy()` is called on clean shutdown only. For crash resilience, rely on `Drop` for GPU resource cleanup.
- The library calls `vkDeviceWaitIdle` before restart to ensure no GPU work references the old engine's resources.
- If your engine uses `unsafe` code, a crash may leave GPU state corrupted. Consider validating device state in `init()` after a restart.
- For engines with expensive initialization, consider keeping reusable resources (pipeline caches, shader modules) outside the `EngineRenderer` and passing them into `new()` on restart.
- `EngineHandle::restart()` takes the new engine by value. Construct a fresh `EngineRenderer` instance inside the `ui` closure or from state captured by the closure. Since the `ui` closure is `FnMut + 'static`, all state it references must be owned or `'static` — this is inherent to the event loop model.
- `panic = "abort"` in your release profile disables fault isolation entirely. Engine panics will terminate the process. Use `panic = "unwind"` if you need crash recovery.

**Multi-viewport scope:** This design covers a single engine viewport per `run()` call. Multiple engine viewports (e.g., a 3D scene + a texture preview) are out of scope for v2 but could be supported by allowing multiple `EngineRenderer` instances in a future version.

## Removed from v1

| Removed | Replaced by |
|---|---|
| `App` trait | `ui` closure + `EngineRenderer` trait |
| `AppCreator` trait | User creates Vulkan objects before `run()` |
| `CreationContext` | `VulkanContext` + `EngineContext` |
| `AshRenderState<A>` | `VulkanContext` |
| `HandleRedraw` / `RedrawHandler` | Library always composites |
| `EguiCommand` | Engine never touches egui rendering |
| `ImageRegistry` / user texture mpsc | Engine renders to lent target, composited as texture |
| `Allocator` trait + generics | Engine manages its own memory |
| `Arc<Mutex<>>` chains | Timeline semaphore + channels |
| `presenters.rs` | `ViewportContext` (Round 1), further simplified in v2 |
