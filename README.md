# egui-ash

[![Latest version](https://img.shields.io/crates/v/egui-ash.svg)](https://crates.io/crates/egui-ash)
[![Documentation](https://docs.rs/egui-ash/badge.svg)](https://docs.rs/egui-ash)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache2.0](https://img.shields.io/badge/license-Apache2.0-blue.svg)
[![egui: 0.28](https://img.shields.io/badge/egui-0.28-orange)](https://docs.rs/egui/0.28/egui)
[![ash: 0.38](https://img.shields.io/badge/ash-0.38-orange)](https://docs.rs/ash/0.38/ash)

[egui](https://github.com/emilk/egui) integration for [ash](https://github.com/MaikKlein/ash) (Vulkan).

Manages the winit event loop, Vulkan swapchains, and egui rendering so that consumers only need to implement two traits. Supports egui multi-viewports, render-to-texture scene views, and custom Vulkan rendering alongside egui.

## Features

- Full egui multi-viewport support (immediate and deferred viewports each get their own swapchain)
- Bring your own allocator via the `Allocator` trait, or use the ready-made `gpu-allocator` integration
- Render custom Vulkan content alongside egui via `HandleRedraw::Handle`
- Embed Vulkan-rendered textures inside egui panels via `ImageRegistry`
- Optional window layout and egui memory persistence via RON files

## Usage

Add the dependency:

```toml
[dependencies]
egui-ash = { version = "0.4", features = ["gpu-allocator"] }
```

Implement `App` and `AppCreator`, then call `run`:

```rust
use egui_ash::{App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption};

struct MyApp;
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello egui-ash!");
        });
    }

    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        // Return Auto to let egui-ash handle presentation,
        // or Handle(Box::new(|size, egui_cmd| { ... })) to drive your own Vulkan render loop.
        HandleRedraw::Auto
    }
}

struct MyAppCreator;
impl AppCreator<Arc<Mutex<gpu_allocator::vulkan::Allocator>>> for MyAppCreator {
    type App = MyApp;

    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<Arc<Mutex<gpu_allocator::vulkan::Allocator>>>) {
        // Create your Vulkan instance, device, allocator, etc. using the extension
        // lists provided in `cc.required_instance_extensions` and
        // `cc.required_device_extensions`, then return them via AshRenderState.
        let ash_render_state = AshRenderState { /* ... */ };
        (MyApp, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "my-app",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default().with_title("My App"),
            ),
            ..Default::default()
        },
    )
}
```

See the [examples](examples/) for complete working code covering egui-only rendering, custom Vulkan rendering, multi-viewports, render-to-texture scene views, and image loading.

## Custom Vulkan rendering

When you need to draw Vulkan content on the same window as egui, return `HandleRedraw::Handle` from `request_redraw`. The closure receives the window size and an `EguiCommand` handle; call `egui_cmd.update_swapchain` on first render and after resize, then `egui_cmd.record` to append the egui draw calls to your command buffer:

```rust
fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
    HandleRedraw::Handle(Box::new({
        let renderer = self.renderer.clone();
        move |size, egui_cmd| renderer.render(size.width, size.height, egui_cmd)
    }))
}
```

## Render-to-texture

Register an off-screen color image view with `ImageRegistry::register_user_texture` to obtain an `egui::TextureId`. Pass that id to `egui::Image` to embed Vulkan-rendered content inside any egui panel. Unregister with `unregister_user_texture` when the image is destroyed.

## Feature flags

| Feature | Description |
|---|---|
| `gpu-allocator` | Implements `Allocator` for `Arc<Mutex<gpu_allocator::vulkan::Allocator>>` |
| `persistence` | Saves/restores window layout and egui memory to disk via RON, keyed by `app_id` |
| `wayland` | Wayland support (passed through to `egui-winit`) |
| `x11` | X11 support (passed through to `egui-winit`) |
| `accesskit` | Accessibility support (passed through to `egui-winit`) |

## License

MIT OR Apache-2.0
