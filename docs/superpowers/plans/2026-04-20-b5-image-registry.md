# Sub-Plan: B5 — ImageRegistry Restoration

**Parent:** `2026-04-20-v1-parity-restoration.md` §5.B5
**Status:** Active reference doc. Not yet executing.
**Estimated effort:** 4–6 hours.

---

## 1. Goal

Restore v1's `ImageRegistry` capability: allow users to register a `vk::ImageView` + `vk::Sampler` pair and receive back an `egui::TextureId` they can use with `egui::Image::new(...)` / `ui.image(...)`. The user owns the underlying Vulkan image; the library owns the descriptor set.

## 2. Why this matters

v1's `scene_view`, `native_image`, and `tiles` examples all depend on user texture registration. Any downstream app that renders off-screen (thumbnails, 3D previews, video playback, image galleries) uses this pattern. Without it, the only user-visible Vulkan output is the single engine viewport — a significant capability regression.

## 3. Current state in v2

`src/compositor.rs` already has partial scaffolding:
- `const ENGINE_VIEWPORT_USER_ID: u64 = u64::MAX` — reserves the top user-ID slot for the engine viewport.
- `engine_viewport_descriptor_set: vk::DescriptorSet` — the single user-texture slot, hardcoded for the engine.
- `engine_viewport_texture_id: egui::TextureId::User(u64::MAX)` — exposed via `EngineStatus::viewport_texture_id`.
- Descriptor pool has `UPDATE_AFTER_BIND_POOL` flag already set — required so user textures can be re-bound while previous frames are still in flight.

What's missing: a command channel, an n-way user-texture storage, and public API surface.

## 4. Public API

### 4.1 `ImageRegistry`

```rust
pub struct ImageRegistry {
    tx: mpsc::Sender<RegistryCommand>,
    next_id: Arc<AtomicU64>,
}

impl ImageRegistry {
    /// Register a Vulkan image (as an image view + sampler) for use in
    /// egui panels. Returns a handle whose `id()` can be passed to
    /// `egui::Image::new(...)`.
    ///
    /// # Image state requirements
    ///
    /// The image **must** be in `VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL`
    /// whenever egui-ash samples from it — typically for one frame after
    /// the handle is registered or after each `update()` call. The caller
    /// is responsible for recording the layout transition on their own
    /// queue and ensuring synchronization (e.g., via timeline semaphores)
    /// with the host queue that samples the image.
    pub fn register(
        &self,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> UserTextureHandle;
}

impl Clone for ImageRegistry { /* ... */ }
```

`Clone + Send + Sync` so the registry is usable from both the UI thread (via `EngineHandle::image_registry`) and the engine thread (via `EngineContext::image_registry`).

### 4.2 `UserTextureHandle`

```rust
pub struct UserTextureHandle {
    id: egui::TextureId,
    tx: mpsc::Sender<RegistryCommand>,
}

impl UserTextureHandle {
    /// The egui texture ID — pass to `egui::Image::new(id)`.
    pub fn id(&self) -> egui::TextureId { self.id }

    /// Replace the image view + sampler backing this ID. Useful when
    /// re-creating images after a resize or content change. The
    /// UPDATE_AFTER_BIND layout flag makes this safe to call even while
    /// previous frames are still in flight.
    pub fn update(&self, image_view: vk::ImageView, sampler: vk::Sampler);
}

impl Drop for UserTextureHandle {
    fn drop(&mut self) {
        // Send Unregister command. Best-effort — channel may already be closed
        // if the Host has exited.
    }
}
```

`UserTextureHandle` is NOT `Send + Sync` — it's tied to the Drop lifecycle of a specific registration. Users who need to share ownership can wrap in `Arc<UserTextureHandle>`.

### 4.3 Where is `ImageRegistry` obtained?

**Two access points, same underlying channel:**

1. **UI thread (closure):** Add a method to `EngineHandle<E>`:
   ```rust
   impl<E: EngineRenderer> EngineHandle<E> {
       pub fn image_registry(&self) -> &ImageRegistry;
   }
   ```
   Rationale: `EngineHandle` is the user's "runtime interaction" handle. Adding registry alongside `restart()` / `exit()` keeps all cross-system capabilities in one place. The UI closure already has `&EngineHandle<E>`.

2. **Engine thread (init + render):** Add to `EngineContext`:
   ```rust
   pub struct EngineContext {
       // ... existing fields ...
       pub image_registry: ImageRegistry,
   }
   ```
   Rationale: engines often pre-create textures during `init()` (static assets). They need access then, not just during `render()`.

Both are clones of the same channel, so registrations from either thread end up in the same compositor queue.

## 5. Internal types

### 5.1 `RegistryCommand`

```rust
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
    Unregister { id: u64 },
}
```

Owned by the compositor; drained once per frame before any user textures might be bound.

### 5.2 `UserTextureSlot` (compositor-side)

```rust
struct UserTextureSlot {
    descriptor_set: vk::DescriptorSet,
    // vk::ImageView and vk::Sampler are user-owned; not stored long-term.
}
```

Stored in `HashMap<u64, UserTextureSlot>` on the `Compositor`.

## 6. Compositor changes

### 6.1 Descriptor pool

Decision: **separate pool for user textures**, sized to `MAX_USER_TEXTURES = 1024` initially. Keeps user-texture lifetime independent of engine-viewport + managed-texture churn, and simplifies reasoning.

Layout: same `UPDATE_AFTER_BIND` descriptor set layout used for engine viewport — reuse `self.descriptor_set_layout` (already in Compositor).

### 6.2 Channel + command handling

Add to `Compositor`:
```rust
user_tex_rx: mpsc::Receiver<RegistryCommand>,
user_tex_pool: vk::DescriptorPool,
user_textures: HashMap<u64, UserTextureSlot>,
```

Add a `drain_registry_commands(&mut self)` method called at the top of `render_frame`. Process all available commands in order.

### 6.3 Binding at draw time

In `record_egui_commands`, when an `egui::epaint::Primitive` references `TextureId::User(id)`:
- `id == u64::MAX` → bind engine viewport descriptor set (current code path).
- Otherwise → look up in `user_textures` map, bind its descriptor set.
- Missing entry → log once and skip (render with whatever was last bound, or with a fallback "missing texture" slot).

## 7. Layout transition responsibility

**Decision: caller-enforced layout, document clearly.**

Alternative considered: library does a one-shot transition on Register. Rejected because (a) it requires a second command buffer submit on the host queue at arbitrary times, (b) layout transitions from unknown src layouts need to be stomping, which loses content, (c) users frequently already transition their images as part of their own render pipeline and know the current layout.

The API doc on `register()` states the requirement explicitly. Example code in examples/ shows the pattern.

## 8. Lifecycle and cleanup

### 8.1 Command dropping on Host exit

When `Host::destroy` runs, the registry channel's receiver is dropped. Subsequent `tx.send(...)` from user code returns `Err` — silently discarded. `UserTextureHandle::drop` sending Unregister into a closed channel is a no-op, which is correct.

### 8.2 Compositor destroy path

`Compositor::destroy` must:
- Drain any remaining commands from the channel (best-effort).
- Free all descriptor sets allocated from the user-texture pool.
- Destroy the user-texture descriptor pool.

Order: user-texture pool destroyed before the main descriptor pool (reverse creation order).

### 8.3 User's vk::ImageView and Sampler are user-owned

egui-ash does NOT destroy `image_view` or `sampler` passed to `register()`. The user must `destroy_image_view` / `destroy_sampler` themselves after:
1. Unregistering (or dropping the handle).
2. Ensuring no in-flight command buffers still reference it — typically via `device_wait_idle` on shutdown, or via timeline semaphore on resource rotation.

This is the same ownership model as v1.

## 9. Implementation steps (execution order)

1. Create `src/image_registry.rs` with `ImageRegistry`, `UserTextureHandle`, `RegistryCommand` (`pub(crate)`).
2. Wire up the channel: expose a `pub(crate) fn new_pair() -> (ImageRegistry, mpsc::Receiver<RegistryCommand>)` so `Host::new` can construct both ends.
3. Modify `Compositor::new` to accept the receiver + a physical device handle to query descriptor pool limits. Create the user-texture descriptor pool. Store the map.
4. Add `Compositor::drain_registry_commands`. Call it at the top of `render_frame`.
5. In `Compositor::record_egui_commands`: look up user textures by ID, bind correct descriptor set.
6. Add `EngineHandle::image_registry()`. Store the `ImageRegistry` clone on `EngineHandle`.
7. Add `ImageRegistry` to `EngineContext`. Plumb through `Host::new` → `spawn_engine_thread` → engine's `init`.
8. Add `Compositor::destroy` cleanup for user textures + pool.
9. Re-export `ImageRegistry` and `UserTextureHandle` from `lib.rs`.
10. Port the v1 `native_image` or `scene_view` example to v2 as an acceptance test.

## 10. Non-goals for this sub-plan

- Per-frame texture updates from the engine thread (use Update command — already in API).
- Layout transition automation (caller-enforced, documented).
- Dynamic descriptor pool growth (fixed 1024 slots — if someone needs more, they can raise and we address later).
- Texture format variance beyond what egui already supports (RGBA8 + common compressed formats).

## 11. Acceptance

1. `cargo clippy --workspace --all-targets` clean under default, `persistence`, and `accesskit` features.
2. A v2 port of v1's `native_image` example: loads a PNG via `image` crate, uploads to a Vulkan image (user-owned), registers via `ImageRegistry`, displays in an `egui::CentralPanel` using `egui::Image::new(handle.id())`.
3. A v2 port of `scene_view`: engine renders Suzanne to an off-screen image, registers, displays in an `egui::Window`.
4. Both examples compile and run without validation-layer errors.
