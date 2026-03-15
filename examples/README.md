# egui-ash examples

Run any example from the repository root with:

```
cargo run --release --example <name> --features gpu-allocator
```

---

## simple

A minimal egui-only example. No custom Vulkan rendering — egui-ash handles the swapchain and presentation automatically.

![simple](../screenshots/egui-ash-simple.png)

```
cargo run --release --example egui_ash_simple --features gpu-allocator
```

---

## vulkan

egui alongside a custom Vulkan renderer. The 3D model (Suzanne) is rendered by a hand-rolled Vulkan pipeline that shares the swapchain with egui via `HandleRedraw::Handle` and `EguiCommand`.

![vulkan](../screenshots/egui-ash-vulkan.png)

```
cargo run --release --example egui_ash_vulkan --features gpu-allocator
```

---

## images

Demonstrates egui's image loading: a remote JPEG via URL (using `egui_extras`) and a local SVG file embedded with `egui::include_image!`.

![images](../screenshots/egui-ash-images.png)

```
cargo run --release --example images --features gpu-allocator
```

---

## multi viewports

Multiple egui viewports, each with its own swapchain. The main window renders a triangle; a deferred viewport opens on demand and renders the 3D Suzanne model, with each viewport driven by its own `HandleRedraw::Handle` closure. Also demonstrates immediate viewports.

![multi viewports](../screenshots/egui-ash-multi-viewports.png)

```
cargo run --release --example multi_viewports --features gpu-allocator
```

---

## native image

Loads a BMP from disk using the `image` crate, uploads it to Vulkan, registers it with `ImageRegistry`, and displays it as an egui texture — the pattern for embedding arbitrary GPU images in egui.

![native image](../screenshots/egui-ash-native-image.png)

```
cargo run --release --example native_image --features gpu-allocator
```

---

## scene view

A Vulkan-rendered 3D scene embedded inside an egui window using render-to-texture. The off-screen color image is registered via `ImageRegistry::register_user_texture` and displayed with `egui::Image`. Drag inside the panel to rotate the model.

![scene view](../screenshots/egui-ash-scene-view.png)

```
cargo run --release --example scene_view --features gpu-allocator
```

---

## tiles

A tiled layout built with [`egui_tiles`](https://github.com/rerun-io/egui_tiles). The scene view pane embeds the Vulkan-rendered Suzanne model. A properties pane exposes background color, object material, and lighting controls that update the render in real time via a shared `Arc<Mutex<Scene>>`.

![tiles](../screenshots/egui-ash-tiles.png)

```
cargo run --release --example tiles --features gpu-allocator
```
