# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build the library
cargo build

# Run clippy (active cleanup branch: clippy-cleanup)
cargo clippy

# Run tests
cargo test

# Run an example
cargo run --release --example egui_ash_simple
cargo run --release --example egui_ash_vulkan
cargo run --release --example images
cargo run --release --example multi_viewports
cargo run --release --example native_image
cargo run --release --example scene_view
cargo run --release --example tiles

# Build with optional features
cargo build --features gpu-allocator
cargo build --features persistence
```

## Architecture

`egui-ash` is a Rust library crate that integrates [egui](https://github.com/emilk/egui) with [ash](https://github.com/MaikKlein/ash) (Vulkan bindings). It manages the winit event loop, Vulkan swapchains, and egui rendering so that consumers only need to implement two traits.

### Public API (`src/app.rs`, `src/run.rs`)

The library exposes three things to consumers:

1. **`App` trait** — implement `ui()` for egui drawing, `request_redraw()` to decide between egui-only rendering (`HandleRedraw::Auto`) or custom Vulkan rendering (`HandleRedraw::Handle(fn)`), and optionally `handle_event()` and `save()`.

2. **`AppCreator<A: Allocator>` trait** — implement `create()` which receives a `CreationContext` (with required Vulkan instance/device extensions, image registry, exit signal) and returns `(App, AshRenderState<A>)`. This is where users create all their Vulkan objects.

3. **`run(app_id, creator, RunOption)` function** — the entry point that drives the winit event loop.

### Internal layers

- **`src/run.rs`** — Builds the winit `EventLoop`, calls `AppCreator::create()`, constructs `Integration`, and drives the event loop. Uses `ManuallyDrop<Integration>` to control drop order (Integration before App, for gpu_allocator).

- **`src/integration.rs`** — Core orchestrator. Manages the `ViewportIdMap<Viewport>` (one per egui viewport/window), handles winit window events via `egui_winit::State`, dispatches paint via `Presenters` and `Renderer`. Calls back into `App::request_redraw()` to get either auto or user-supplied `RedrawHandler`.

- **`src/presenters.rs`** — Per-viewport swapchain management. Owns `vk::SwapchainKHR`, surfaces, semaphores, fences, and command buffers. Handles swapchain creation/recreation on resize.

- **`src/renderer.rs`** — Vulkan rendering of egui primitives. Manages descriptor sets, textures (`ImageRegistry`/`ImageRegistryReceiver` for user textures), vertex/index buffers, and the egui render pass. `EguiCommand` is the handle passed to user `HandleRedraw` closures to submit egui draw calls.

- **`src/allocator.rs`** — Abstraction traits: `Allocator`, `Allocation`, `AllocationCreateInfo`, `MemoryLocation`. Users implement these or use the built-in `gpu-allocator` feature.

- **`src/gpu_allocator.rs`** — Implements the `Allocator` trait for `Arc<Mutex<gpu_allocator::vulkan::Allocator>>` (enabled by `gpu-allocator` feature).

- **`src/event.rs`** — Event enum passed to `App::handle_event()`: wraps winit `WindowEvent`, `DeviceEvent`, `AppEvent`, and optionally `AccessKitActionRequest`.

- **`src/storage.rs`** — Persistence via RON files (enabled by `persistence` feature). Stores egui memory and window settings keyed by `app_id`.

### Shaders

Pre-compiled SPIR-V shaders live in `src/shaders/spv/` (included in the published crate). Source GLSL is in `src/shaders/src/`. Recompile with `src/shaders/shader_compile.bat`.

### Examples

Examples share code via `examples/common/`:
- `vkutils.rs` — Vulkan setup helpers (instance, device, queues)
- `model_renderer.rs` / `model_renderer_shared.rs` — 3D model rendering with the Suzanne mesh
- `scene_view.rs` / `scene_view_full.rs` — Render-to-texture scene view widget
- `pane.rs` / `scene.rs` — `egui_tiles` pane/scene abstractions

Common shaders (model + triangle) live in `examples/common/shaders/`.

## Feature Flags

| Feature | Effect |
|---|---|
| `gpu-allocator` | Enables `Arc<Mutex<gpu_allocator::vulkan::Allocator>>` as a ready-made `Allocator` impl |
| `persistence` | Saves/restores window layout and egui memory to disk via RON |
| `wayland`, `x11` | Passed through to `egui-winit` |
| `accesskit` | Accessibility support via `egui-winit` |
