# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix seven issues found in code review of the queue mutex, fallback extent, scale factor, and UPDATE_AFTER_BIND changes on the `v2-editor-compositor` branch.

**Architecture:** Six targeted fixes across the library core and shared example code. The changes touch the public API (`EngineContext` gains a `queue_mutex` field), the compositor's internal rendering math, descriptor setup requirements, swapchain recreation guard logic, and the example Vulkan init code. Each task is independent except Task 1 (public API change) which must land before Tasks 2 and 6.

**Tech Stack:** Rust, ash (Vulkan bindings), egui, egui-winit, winit

---

### Task 1: Thread queue_mutex through to EngineContext (CRITICAL)

**Problem:** `VulkanContext::queue_mutex` never reaches the engine thread. On single-queue hardware the engine submits to the same `VkQueue` without locking — UB per Vulkan spec.

**Files:**
- Modify: `src/types.rs:37-43` (add field to `EngineContext`)
- Modify: `src/host.rs:192-198` (pass mutex when constructing `EngineContext` in `Host::new`)
- Modify: `src/host.rs:534-539` (pass mutex when constructing `EngineContext` in `restart_engine`)

- [ ] **Step 1: Add `queue_mutex` to `EngineContext`**

In `src/types.rs`, add the field to `EngineContext`:

```rust
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
    pub queue_mutex: Option<std::sync::Arc<std::sync::Mutex<()>>>,
}
```

- [ ] **Step 2: Pass mutex in `Host::new`**

In `src/host.rs`, the `EngineContext` construction around line 192:

```rust
        let engine_ctx = EngineContext {
            device: vulkan.device.clone(),
            queue: vulkan.engine_queue,
            queue_family_index: vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
            queue_mutex: vulkan.queue_mutex.clone(),
        };
```

- [ ] **Step 3: Pass mutex in `restart_engine`**

In `src/host.rs`, the `EngineContext` construction around line 534:

```rust
        let engine_ctx = EngineContext {
            device: self.vulkan.device.clone(),
            queue: self.vulkan.engine_queue,
            queue_family_index: self.vulkan.engine_queue_family_index,
            initial_extent: extent,
            format,
            queue_mutex: self.vulkan.queue_mutex.clone(),
        };
```

- [ ] **Step 4: Build and verify**

Run: `cargo check -p egui-ash --message-format=short`
Expected: compiles cleanly (examples will fail until Task 6)

---

### Task 2: Fix double-zoom in scale_factor derivation (HIGH)

**Problem:** `(sc_extent.width / viewport_rect.width()) * zoom_factor()` applies zoom twice — `viewport_rect()` already incorporates zoom via `pixels_per_point`. Produces wrong rendering when zoom != 1.0.

**Files:**
- Modify: `src/host.rs:406-408`

- [ ] **Step 1: Remove the extra zoom_factor multiplication**

In `src/host.rs`, replace the scale_factor computation (around line 403-408):

Old:
```rust
        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let sc_extent = self.compositor.swapchain_extent();
        let screen_size = [sc_extent.width, sc_extent.height];
        let egui_screen = self.context.viewport_rect();
        let w_points = egui_screen.width().max(1.0);
        let scale_factor = (sc_extent.width as f32 / w_points) * self.context.zoom_factor();
```

New:
```rust
        let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
        let sc_extent = self.compositor.swapchain_extent();
        let screen_size = [sc_extent.width, sc_extent.height];
        // scale_factor = physical pixels per egui point.
        // viewport_rect() is in egui points (already zoom-adjusted), so
        // dividing the swapchain's pixel width by it gives the correct
        // pixels-per-point for this framebuffer — no extra zoom needed.
        let egui_screen = self.context.viewport_rect();
        let w_points = egui_screen.width().max(1.0);
        let scale_factor = sc_extent.width as f32 / w_points;
```

- [ ] **Step 2: Update the block comment above**

Replace the `scale_factor` paragraph in the block comment (the lines mentioning zoom_factor derivation):

Old:
```
        // `scale_factor` here has the meaning "physical pixels per egui
        // point" — the compositor's record path divides screen_size by it
        // to get `width_points` for the vertex shader push constant, and
        // multiplies scissor rects by it to go from points to pixels. We
        // derive it directly from the relationship between the swapchain
        // (physical) and egui's screen rect (points) so it's robust to
        // whatever convention `window.inner_size()` follows on the current
        // platform.
```

New:
```
        // `scale_factor` = physical swapchain pixels per egui point.
        // The compositor divides screen_size by it to get `width_points`
        // for the vertex shader push constant, and multiplies scissor rects
        // by it to convert points to pixels. We derive it from the ratio of
        // swapchain extent (pixels) to egui's viewport rect (points).
        // viewport_rect() already incorporates zoom_factor via
        // pixels_per_point, so no extra zoom multiplication is needed.
```

- [ ] **Step 3: Build and verify**

Run: `cargo check -p egui-ash --message-format=short`
Expected: compiles cleanly

---

### Task 3: Document UPDATE_AFTER_BIND requirements on VulkanContext (HIGH)

**Problem:** The compositor now requires Vulkan 1.2 descriptor indexing features (`descriptorBindingSampledImageUpdateAfterBind`, `descriptorBindingUpdateUnusedWhilePending`) but neither documents this on the public API nor checks at runtime. The internal comment references `aesthetix-editor`, a downstream consumer.

**Files:**
- Modify: `src/types.rs:5-34` (add doc on `VulkanContext`)
- Modify: `src/compositor.rs:160-165` (fix internal comment)

- [ ] **Step 1: Add required-features documentation to `VulkanContext`**

In `src/types.rs`, expand the doc comment on `VulkanContext`:

Old:
```rust
/// User-provided Vulkan context. The user creates all Vulkan objects
/// before calling `run()` and destroys them after `run()` returns.
pub struct VulkanContext {
```

New:
```rust
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
pub struct VulkanContext {
```

- [ ] **Step 2: Remove aesthetix-editor reference from compositor comment**

In `src/compositor.rs`, replace the requirements comment (around line 160-165):

Old:
```rust
        // Requirements (all enabled in `aesthetix-editor::vulkan_init`):
        //   - `descriptorBindingSampledImageUpdateAfterBind` (VK 1.2)
        //   - `descriptorBindingUpdateUnusedWhilePending`    (VK 1.2)
        //   - layout flag `UPDATE_AFTER_BIND_POOL`
        //   - pool flag   `UPDATE_AFTER_BIND`
```

New:
```rust
        // Requirements (must be enabled at device creation — see VulkanContext docs):
        //   - `descriptorBindingSampledImageUpdateAfterBind` (VK 1.2)
        //   - `descriptorBindingUpdateUnusedWhilePending`    (VK 1.2)
        //   - layout flag `UPDATE_AFTER_BIND_POOL`
        //   - pool flag   `UPDATE_AFTER_BIND`
```

- [ ] **Step 3: Build and verify**

Run: `cargo check -p egui-ash --message-format=short`
Expected: compiles cleanly

---

### Task 4: Guard swapchain recreation against redundant/zero-size rebuilds (MEDIUM)

**Problem:** `handle_window_event` calls `recreate_swapchain()` (which does `device_wait_idle` + full teardown/rebuild) on every `Resized` event, including minimizes (zero-size) and drags (many rapid events). Also, the `caps.current_extent` path in `create_swapchain_inner` doesn't guard against zero extents.

**Files:**
- Modify: `src/host.rs:448-457` (guard the resize handler)
- Modify: `src/compositor.rs:825-838` (add `.max(1)` to non-fallback path)

- [ ] **Step 1: Guard the resize handler in `handle_window_event`**

In `src/host.rs`, replace the `WE::Resized` arm:

Old:
```rust
            WE::Resized(new_size) => {
                self.compositor.set_fallback_extent(vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                });
                unsafe {
                    self.compositor.recreate_swapchain();
                }
            }
```

New:
```rust
            WE::Resized(new_size) => {
                // Skip zero-sized resizes (e.g. minimize on X11/Windows).
                if new_size.width == 0 || new_size.height == 0 {
                    return self
                        .egui_winit_state
                        .on_window_event(&self.window, event)
                        .consumed;
                }

                let new_extent = vk::Extent2D {
                    width: new_size.width,
                    height: new_size.height,
                };
                self.compositor.set_fallback_extent(new_extent);

                // Only force a rebuild if the swapchain extent actually changed.
                // On drivers that report currentExtent == u32::MAX the frame
                // loop won't get ERROR_OUT_OF_DATE_KHR, so we must rebuild here.
                let current = self.compositor.swapchain_extent();
                if current.width != new_extent.width || current.height != new_extent.height {
                    unsafe {
                        self.compositor.recreate_swapchain();
                    }
                }
            }
```

- [ ] **Step 2: Guard the non-fallback extent path against zero**

In `src/compositor.rs`, in `create_swapchain_inner`, change the `else` branch (around line 837):

Old:
```rust
        } else {
            caps.current_extent
        };
```

New:
```rust
        } else {
            // Guard against zero-extent (e.g. minimized window) — Vulkan
            // requires width and height >= 1 for swapchain creation.
            vk::Extent2D {
                width: caps.current_extent.width.max(1),
                height: caps.current_extent.height.max(1),
            }
        };
```

- [ ] **Step 3: Build and verify**

Run: `cargo check -p egui-ash --message-format=short`
Expected: compiles cleanly

---

### Task 5: Use poison-recovering lock on queue_mutex (LOW)

**Problem:** `.lock().unwrap()` will panic if the mutex is poisoned (e.g. engine thread panicked while holding it). This defeats the `catch_unwind` fault isolation in `engine_thread.rs`.

**Files:**
- Modify: `src/compositor.rs` (lines 533, 551, 1650, 1760 — all four lock sites)

- [ ] **Step 1: Replace all four `.lock().unwrap()` calls**

In `src/compositor.rs`, replace all four occurrences of:

```rust
self.queue_mutex.as_ref().map(|m| m.lock().unwrap())
```

with:

```rust
self.queue_mutex.as_ref().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()))
```

There are exactly four sites:
1. Line ~533 (render_frame queue_submit)
2. Line ~551 (render_frame queue_present)
3. Line ~1650 (update_texture full upload)
4. Line ~1760 (update_texture partial blit)

- [ ] **Step 2: Build and verify**

Run: `cargo check -p egui-ash --message-format=short`
Expected: compiles cleanly

---

### Task 6: Update examples for new EngineContext field and Vulkan 1.2 features (HIGH)

**Problem:** After Task 1 adds `queue_mutex` to `EngineContext`, the two example engines that destructure/use `EngineContext` in `init()` will fail to compile. Additionally, `vkutils.rs` must enable the two new required Vulkan 1.2 descriptor indexing features, and the `VulkanContext` construction needs to supply `queue_mutex: None`.

**Files:**
- Modify: `examples/common/vkutils.rs:189-190` (enable descriptor indexing features)
- Modify: `examples/common/vkutils.rs:210-219` (add `queue_mutex: None` to `VulkanContext`)

- [ ] **Step 1: Enable required Vulkan 1.2 features in `vkutils.rs`**

In `examples/common/vkutils.rs`, expand the `vulkan_12_features` (around line 189-190):

Old:
```rust
    let mut vulkan_12_features =
        vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
```

New:
```rust
    let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
        .timeline_semaphore(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_update_unused_while_pending(true);
```

- [ ] **Step 2: Add `queue_mutex: None` to `VulkanContext` construction**

In `examples/common/vkutils.rs`, update the `VulkanContext` struct literal (around line 210-219):

Old:
```rust
    let vulkan_context = VulkanContext {
        entry,
        instance: instance.clone(),
        physical_device,
        device: device.clone(),
        host_queue,
        host_queue_family_index: queue_family,
        engine_queue,
        engine_queue_family_index: queue_family,
    };
```

New:
```rust
    let vulkan_context = VulkanContext {
        entry,
        instance: instance.clone(),
        physical_device,
        device: device.clone(),
        host_queue,
        host_queue_family_index: queue_family,
        engine_queue,
        engine_queue_family_index: queue_family,
        queue_mutex: None,
    };
```

- [ ] **Step 3: Build all examples**

Run: `cargo check --examples --message-format=short`
Expected: compiles cleanly with no errors

---

### Task 7: Final verification

- [ ] **Step 1: Full workspace check**

Run: `cargo clippy --workspace --message-format=short`
Expected: no errors, no new warnings

- [ ] **Step 2: Run tests**

Run: `cargo test --workspace`
Expected: all tests pass
