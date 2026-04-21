# egui-ash

[![Latest version](https://img.shields.io/crates/v/egui-ash.svg)](https://crates.io/crates/egui-ash)
[![Documentation](https://docs.rs/egui-ash/badge.svg)](https://docs.rs/egui-ash)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache2.0](https://img.shields.io/badge/license-Apache2.0-blue.svg)
[![egui: 0.33.3](https://img.shields.io/badge/egui-0.33.3-orange)](https://docs.rs/egui/0.33.3/egui)
[![ash: 0.38](https://img.shields.io/badge/ash-0.38-orange)](https://docs.rs/ash/0.38/ash)

[egui](https://github.com/emilk/egui) integration for [ash](https://github.com/ash-rs/ash) (Vulkan).

egui-ash drives the winit event loop, owns the swapchain, and composites egui over a dedicated render engine thread so consumers can focus on their own rendering and UI without plumbing boilerplate.

## Architecture

- **Engine thread.** User render code runs on a dedicated thread, isolated by `catch_unwind`. A panic in the engine does not take down the UI — the compositor shows a black viewport and the app can restart the engine in-place via `EngineHandle::restart(...)`.
- **Lock-free state exchange.** `UiState` flows from the UI closure to the engine; `EngineState` flows back. Both use `arc-swap` — readers are wait-free.
- **Timeline-semaphore frame handoff.** Double-buffered render targets with per-slot timeline semaphores. The engine signals when it finishes writing; the compositor waits before sampling.
- **Single compositor surface.** The host thread owns the swapchain. v1's per-viewport custom rendering is intentionally retired; users who want custom Vulkan rendering implement `EngineRenderer`.

## Features

- **Dedicated engine thread** with panic-safe restart (`EngineHandle::restart`)
- **Programmatic exit** (`EngineHandle::exit`)
- **Window geometry + egui memory persistence** (RON file under the platform data dir)
- **User-defined persistent key/value storage** via `Storage::set_value` / `get_value`
- **Periodic auto-save** for crash safety (default 30 s)
- **User texture registration** (`ImageRegistry`) — render to an off-screen Vulkan image and display it in any egui panel
- **Raw device events** forwarded to the engine (for FPS-style camera controls)
- **Lifecycle events** (`Resumed`/`Suspended`/`MemoryWarning`/`LoopExiting`)
- **AccessKit** (via `egui-winit`) — egui widgets reach screen readers

## Usage

```toml
[dependencies]
egui-ash = { version = "1.0.0-alpha.1", features = ["persistence"] }
```

Implement `EngineRenderer`, construct a `VulkanContext`, and call `run`:

```rust,ignore
use egui_ash::{EngineContext, EngineRenderer, RenderTarget, CompletedFrame,
               RunOption, VulkanContext};

#[derive(Default)]
struct MyEngine { /* ... */ }

// State flowing between UI thread and engine thread.
#[derive(Default, Clone)] struct MyUiState;
#[derive(Default, Clone)] struct MyEngineState;

impl EngineRenderer for MyEngine {
    type UiState = MyUiState;
    type EngineState = MyEngineState;

    fn init(&mut self, _ctx: EngineContext) { /* allocate pipelines etc. */ }

    fn render(
        &mut self,
        _ui_state: &Self::UiState,
        _engine_state_out: &mut Self::EngineState,
        target: RenderTarget,
    ) -> CompletedFrame {
        // Wait on target.wait_semaphore (timeline value target.wait_value),
        // render into target.image, signal target.signal_semaphore with
        // target.signal_value, then:
        target.complete()
    }

    fn handle_event(&mut self, _event: egui_ash::EngineEvent) {}
    fn destroy(&mut self) {}
}

fn main() -> std::process::ExitCode {
    let vulkan: VulkanContext = build_vulkan_context(/* ... */);

    egui_ash::run(
        "my-app",
        vulkan,
        MyEngine::default(),
        RunOption::default(),
        |ctx, status, _ui_state, _engine_state, _engine, _storage| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("egui-ash");
                ui.image(status.viewport_texture_id);
            });
        },
    )
}
# fn build_vulkan_context() -> VulkanContext { todo!() }
```

See the [`examples/`](examples/) directory for:

| Example              | Demonstrates                                              |
|----------------------|-----------------------------------------------------------|
| `v2_simple`          | Minimal egui UI with a solid-color engine viewport        |
| `egui_ash_simple`    | Theme switching, widgets, font loading                    |
| `egui_ash_vulkan`    | Triangle rendering through `EngineRenderer`               |
| `images`             | egui_extras image loading                                 |
| `scene_view`         | 3D model rendered by the engine, shown in a floating window |
| `tiles`              | `egui_tiles` layout with a scene-view pane                |

## Required Vulkan 1.2 features

Callers **must** enable these on the `VkDevice`:

- `timelineSemaphore`
- `descriptorBindingSampledImageUpdateAfterBind`
- `descriptorBindingUpdateUnusedWhilePending`

Example setup code lives in [`examples/common/vkutils.rs`](examples/common/vkutils.rs).

## Known limitations (1.0.0-alpha)

- **Single queue family only.** `host_queue_family_index` must equal `engine_queue_family_index` — cross-family image ownership transfer is not yet wired host-side. Single-queue hardware is fully supported via `VulkanContext::queue_mutex`. See [`docs/known-limitations.md`](docs/known-limitations.md).
- **Multi-viewport egui UI not yet restored.** `egui::Window::show_viewport` / `show_viewport_immediate` calls are silently ignored in this alpha. Tracked in [`docs/superpowers/plans/2026-04-20-b9-multi-viewport.md`](docs/superpowers/plans/2026-04-20-b9-multi-viewport.md).
- **Per-viewport custom rendering retired.** v1's `HandleRedraw::Handle` pattern is gone — use `EngineRenderer` instead. This is an intentional architectural change, not a bug.
- **`Allocator` trait / `gpu-allocator` feature removed.** The user owns their `VkDevice` and all memory allocation; egui-ash allocates its small internal buffers directly via `vkAllocateMemory`.

## Migrating from 0.4

1.0.0-alpha is a rewrite. Key shape changes:

| v0.4 concept             | v1.0 equivalent                                   |
|--------------------------|---------------------------------------------------|
| `trait App`              | `trait EngineRenderer` + UI closure               |
| `trait AppCreator<A>`    | User constructs `VulkanContext` directly          |
| `AshRenderState<A>`      | `VulkanContext`                                   |
| `HandleRedraw::Handle`   | `EngineRenderer::render`                          |
| `ImageRegistry`          | Same name, obtained via `EngineHandle::image_registry()` |
| `ExitSignal::send(code)` | `EngineHandle::exit(code)`                        |
| `App::save(storage)`     | UI closure has `&mut Storage` as 6th parameter    |
| `App::handle_event`      | `EngineRenderer::handle_event(EngineEvent)`       |
| `Allocator` trait        | **Removed** — user manages allocation externally  |
| `gpu-allocator` feature  | **Removed** — use the crate directly if desired   |

See [`CHANGELOG.md`](CHANGELOG.md) for the full breaking-change list.

## Feature flags

| Feature       | Description                                                     |
|---------------|-----------------------------------------------------------------|
| `persistence` | Enables `Storage` disk I/O; auto-save of window + egui memory.  |
| `accesskit`   | Wires egui-winit's AccessKit adapter (screen-reader support).   |
| `wayland`     | Wayland support (passed through to `egui-winit`).               |
| `x11`         | X11 support (passed through to `egui-winit`).                   |

## License

MIT OR Apache-2.0
