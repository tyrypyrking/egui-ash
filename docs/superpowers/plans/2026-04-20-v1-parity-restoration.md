# v1 Parity Restoration Plan (pre-1.0.0-alpha)

**Status:** Active reference doc. Not yet executing.
**Created:** 2026-04-20
**Target branch:** `v2-editor-compositor`
**Consumer base:** ~200 weekly crates.io downloads on `main` (v0.4.0). Real users rely on v1 features.

---

## 1. Purpose and scope

This document enumerates every user-visible feature regression introduced by the v2 rewrite (branch `v2-editor-compositor`) relative to the published v0.4.0 on `main`. It is the source of truth for what needs to land before cutting `1.0.0-alpha.1`.

Each regression is classified into one of two categories:

- **Category A — Intentional architectural shifts.** v2 deliberately replaces the v1 design. Users who relied on these patterns need to migrate their code; the documentation (CHANGELOG + migration guide) must describe the replacement. These are **not** to be restored.
- **Category B — Incidental losses.** Features v1 provided that v2 dropped without architectural justification. These are bugs in the rewrite and must be restored before alpha.

The doc is written so a future session — human or agent — can pick up any single item and execute it without first re-deriving the context.

---

## 2. Why this exists

The first audit of this branch was diff-based and caught the headline regressions (App/AppCreator removed, multi-viewport dropped, gpu-allocator gone, persistence dead). A second, deeper audit comparing `main` and `HEAD` source-by-source surfaced multiple additional regressions that would silently break downstream users on upgrade. The maintainer's mandate: ship 1.0.0-alpha only when the v1 feature surface is either restored or explicitly retired with a documented replacement.

---

## 3. Key file locations

### v2 source (current HEAD)
- `src/lib.rs` — public re-exports (currently: `engine::*`, `EngineHandle`, `run::*`, `types::*`, `EngineEvent`, `PointerButtons`)
- `src/run.rs` — event loop driver, `run()` entry point
- `src/host.rs` — `Host<E>` internal orchestrator, `EngineHandle<E>`
- `src/types.rs` — `VulkanContext`, `EngineContext`, `RenderTarget`, `CompletedFrame`, `EngineStatus`, `EngineHealth`, `EngineRestartError`, `RunOption`
- `src/engine.rs` — `EngineRenderer` trait
- `src/engine_thread.rs` — engine thread spawning, `catch_unwind` isolation
- `src/compositor.rs` — swapchain, egui rendering, texture upload, per-frame submission (2091 lines)
- `src/render_targets.rs` — `RenderTargetPool` (double-buffered images + timeline semaphores)
- `src/event.rs` — `EngineEvent`, `PointerButtons`
- `src/mailbox.rs` — thread-safe mailbox for `CompletedFrame` handoff
- `src/state_exchange.rs` — `Arc<ArcSwap<>>`-based state sharing
- `src/storage.rs` — persistence module (currently `pub(crate)`, dead island)

### v1 source (main branch — reference via `git show main:<path>`)
- `src/lib.rs` — v1 public re-exports
- `src/app.rs` — `App`, `AppCreator`, `CreationContext`, `AshRenderState`, `HandleRedraw`
- `src/run.rs` — v1 `run()`, `RunOption`, `ExitSignal`
- `src/integration.rs` — v1 `Integration` orchestrator (959 lines)
- `src/renderer.rs` — v1 `Renderer`, `ImageRegistry`, `EguiCommand`, `SwapchainUpdateInfo` (1732 lines)
- `src/presenters.rs` — v1 per-viewport presenter (639 lines)
- `src/allocator.rs` — v1 `Allocator` trait
- `src/gpu_allocator.rs` — v1 `gpu-allocator` integration
- `src/event.rs` — v1 `Event`, `AppEvent`, `DeferredViewportCreated`, `AccessKitActionRequest`
- `src/storage.rs` — v1 persistence (identical internals to v2's)

### Related in-tree docs
- `docs/known-limitations.md` — cross-queue-family transfer not implemented (written 2026-04-20)
- `docs/superpowers/plans/2026-03-23-editor-compositor-v2.md` — original v2 design spec
- `docs/superpowers/plans/2026-04-15-review-fixes.md` — the queue-mutex/UPDATE_AFTER_BIND review fixes (applied)

---

## 4. Category A — Intentional architectural shifts (document, do not restore)

These are the design of v2. Users must migrate; the library does not re-expose the v1 shape. CHANGELOG and a migration guide describe each replacement.

### A1. Trait-based App / AppCreator → EngineRenderer + UI closure

**v1 shape:** Users implemented `App` (ui / handle_event / request_redraw / save) and `AppCreator<A>` (factory returning `(App, AshRenderState<A>)`). `run(app_id, creator, options)` drove everything via trait dispatch.

**v2 shape:** Users construct a `VulkanContext` themselves, implement `EngineRenderer` for their render logic (runs on a dedicated thread with fault isolation), and pass a UI closure inline to `run()`. State flows through `UiState` / `EngineState` associated types with `Arc<ArcSwap<>>` exchange.

**Migration guidance:** The rewrite is in the direction of functional composition. v1 `App::ui` becomes the UI closure body; v1 rendering code moves into `EngineRenderer::render`. See "Migration guide" deliverable below.

### A2. `HandleRedraw::Handle(fn)` → `EngineRenderer::render()` on engine thread

**v1 shape:** `App::request_redraw(viewport_id) -> HandleRedraw` allowed returning `HandleRedraw::Handle(Box<FnOnce(PhysicalSize, EguiCommand)>)`. The user's closure recorded Vulkan commands and called `egui_cmd.update_swapchain(...)` + `egui_cmd.record(cmd, swapchain_index)`. Ran synchronously on the event-loop thread.

**v2 shape:** Continuous engine thread renders to a `RenderTarget` (image + timeline semaphores). The host composites the engine's output over egui. The engine thread is the *only* path for user Vulkan rendering.

**Why this is architectural, not incidental:** v2's entire fault-isolation and panic-recovery story depends on the engine running in its own thread with `catch_unwind`. Reintroducing a synchronous main-thread render callback would bypass that guarantee and re-create the problem v2 was designed to solve.

**Migration guidance:** Move the closure body into `EngineRenderer::render`. Swapchain updates are no longer a user concern. Present is implicit. See migration guide.

### A3. `Allocator` trait / `gpu-allocator` feature → user owns VkDevice directly

**v1 shape:** `App` and `AppCreator<A>` were generic over a user-supplied `Allocator`. `AshRenderState<A>` carried the allocator; `ImageRegistry` and `Renderer` were parameterized over it. `gpu-allocator` feature provided a default impl (`Arc<Mutex<gpu_allocator::vulkan::Allocator>>`).

**v2 shape:** User creates `VulkanContext` end-to-end, including whatever allocator they like, before calling `run()`. The library does not need an allocator — it allocates its own small internal resources (descriptor pools, staging buffers, one texture per user image) using `vkAllocateMemory` directly.

**Why this is architectural:** `Allocator: Clone + 'static + Send + Sync` was one of the main sources of `Arc<Mutex<>>` chains the v2 rewrite eliminated. Re-adding it would undo the whole "Round 1 restructure" described in the project-restructure-round1 memory.

**Migration guidance:** Users who used `gpu-allocator` keep doing so outside the library. Examples show how. CHANGELOG notes the feature removal.

### A4. Per-viewport custom rendering (multi_viewports example) → single engine output

**v1 shape:** Each egui viewport (immediate or deferred) could return its own `HandleRedraw::Handle` with independent Vulkan rendering. The v1 `multi_viewports` example demonstrates this.

**v2 shape:** One engine renders a single output texture. That texture is referenced by `TextureId::User(u64::MAX)` and can appear in any egui panel in any window.

**Why this is architectural:** A single engine thread producing a single output is the v2 model. Multiple independent Vulkan pipelines per viewport would require multiple engine threads, which is out of scope for 1.0.

**Migration guidance:** Multi-viewport *egui UI* must still work (see B9 below — possibly a regression to restore). Multi-viewport *custom rendering* is explicitly retired.

### A5. `CreationContext` / `AshRenderState<A>` / `ExitSignal` types

**v1 shape:** `CreationContext` bundled `main_window`, `context`, `required_*_extensions`, `image_registry`, `exit_signal` for the `AppCreator::create` callback.

**v2 shape:** These responsibilities are split. User creates their own Vulkan context (no required-extensions helper). Image registry is a separate regression (B5). Exit signal is a separate regression (B4).

**Why this is architectural:** These types were tightly coupled to the trait-based creation flow. They make no sense in the v2 shape. Their constituent capabilities (image registry, exit signal) are restored individually in Category B.

---

## 5. Category B — Incidental regressions to restore before alpha

Each item below contains the detail needed to execute without prior context.

### B1. Persistence wiring (window geometry + egui memory)

**Status: IN PROGRESS** — partially plumbed before this plan was written. See `src/run.rs` (added `app_id: String` to `AppState`), `src/storage.rs` (module doc updated, `#![allow(dead_code)]` removed), `src/lib.rs` (module made `pub(crate)`).

**What v1 did:**
- `Storage::from_app_id(app_id)` loaded `<data_dir>/<app_id>/app.ron` on startup (RON format, HashMap<String, String>).
- `set_windows` / `get_windows` persisted `HashMap<egui::ViewportId, egui_winit::WindowSettings>` — inner size, outer position, fullscreen, maximized flags.
- `set_egui_memory` / `get_egui_memory` persisted `egui::Memory` (collapsed headers, scroll positions, text-edit state, etc.) via `egui` crate's own `persistence` feature.
- `Integration::handle_event` wrote window settings on `WindowEvent::Resized` / `Moved` / etc., and saved egui memory on `LoopExiting`.
- Automatic 30-second flush driven by a timer (see B3 for the `auto_save_interval` hook — related but separate).

**What v2 has today:**
- `src/storage.rs` module exists and compiles with identical internals.
- `RunOption::persistent_windows` and `persistent_egui_memory` fields exist.
- Nothing in `src/host.rs` or `src/run.rs` ever calls `Storage::from_app_id`.
- `egui::Memory` is never restored to the context; window is created at default size regardless of saved state.

**User impact:** App starts at default size every time even with `persistent_windows: true`. Collapsed panels re-open. Text editors lose cursor position. A user switching from v0.4.0 → 1.0.0-alpha loses all state restoration silently.

**Restoration approach:**

1. **`src/run.rs::AppState::resumed`** — Before creating the window:
   ```rust
   #[cfg(feature = "persistence")]
   let storage: Option<Storage> = if self.options.persistent_windows
       || self.options.persistent_egui_memory
   {
       Storage::from_app_id(&self.app_id).ok()
   } else {
       None
   };
   ```
   If `persistent_windows` + storage loaded: call `storage.get_windows()`, extract the `ViewportId::ROOT` entry, and apply it to the `WindowAttributes` via `WindowSettings::initialize_viewport_builder(1.0, event_loop, vb)` **before** the current manual extraction loop.

2. **Window-attribute extraction** — the current `resumed` only extracts `title`, `inner_size`, `min/max_inner_size`, `decorations`, `resizable`, `transparent`. Add `position`, `maximized`, `fullscreen` extraction so `WindowSettings::initialize_viewport_builder` output actually propagates.

3. **`src/host.rs::Host::new`** — Accept `app_id: String` and `#[cfg(feature = "persistence")] storage: Option<Storage>` as parameters. Store both as fields. After `egui::Context` is constructed, if `persistent_egui_memory` and storage has egui memory, restore it:
   ```rust
   #[cfg(feature = "persistence")]
   if let Some(s) = &storage {
       if options.persistent_egui_memory {
           if let Some(mem) = s.get_egui_memory() {
               context.memory_mut(|m| *m = mem);
           }
       }
   }
   ```

4. **`src/host.rs::Host::destroy`** — If storage exists: capture `WindowSettings::from_window(zoom_factor, &window)`, store keyed by `ViewportId::ROOT`, capture egui memory via `context.memory(|m| m.clone())`, call `storage.flush()` (synchronous-on-drop via `Drop` impl in `storage.rs:149`).

5. **Periodic flush during runtime** — for crash safety. Options: (a) flush on every `WindowEvent::Resized`/`Moved` (matches v1 behavior), (b) timer-driven 30-second flush. Recommend (a) + save-on-destroy; (b) is the `auto_save_interval` feature and is covered separately in B3.

**Files to touch:** `src/run.rs`, `src/host.rs`, `src/storage.rs` (already partially updated).

**`#[cfg(feature = "persistence")]` on parameters:** Rust supports `#[cfg]` on fn arguments (stable). Use it to keep a single `Host::new` signature rather than duplicating the function body.

**Effort:** ~1–2 focused hours.

**Acceptance:** Run an example with `persistent_windows: true`, resize + move the window, close, re-launch — window opens at saved position and size. Same for collapsed panel state with `persistent_egui_memory: true`.

---

### B2. User-level Storage API (public `set_value` / `get_value`)

**What v1 did:**
- `CreationContext::storage` gave `AppCreator::create` access to load persisted user state at startup.
- `Storage::set_value<T: Serialize>` / `get_value<T: DeserializeOwned>` allowed arbitrary typed K/V storage keyed by any string. Stored as RON, alongside the library's own `egui_memory` / `egui_windows` keys.
- Typical v1 usage: apps persisted custom preferences (selected tab, last-opened file path, zoom factor, etc.) between sessions.

**What v2 has today:**
- The methods exist as `#[cfg(test)] pub(crate)` in `src/storage.rs` — no user-accessible path.
- The `Storage` type itself is `pub(crate)` (the whole module was reduced to `pub(crate) mod storage` in this plan cycle).

**User impact:** Downstream apps that stored user preferences via `Storage::set_value` have no v2 equivalent. They'd have to write their own RON file alongside egui-ash's — stepping on the same directory and file pattern.

**Restoration approach — design decision required:**

The clean way to expose this depends on the UI closure shape. Two options:

**Option B2a: Extend the UI closure signature (breaking change to all 6th params).**
```rust
ui: impl FnMut(
    &egui::Context,
    &EngineStatus,
    &mut E::UiState,
    &E::EngineState,
    &EngineHandle<E>,
    &mut Storage,       // NEW — always present; no-op stub when feature off
) + 'static
```

This requires `Storage` to be a type that always exists regardless of feature. When feature is off: a ZST with no-op methods. When feature is on: the real deal. Pros: stable signature. Cons: the stub is slightly weird.

**Option B2b: Add a `storage()` accessor to something always in the closure.**
Add `EngineHandle<E>::storage(&self) -> Option<&StorageHandle>` or expose via a new type. Pros: closure signature stable. Cons: EngineHandle is currently for restart only; mixing concerns muddies the API.

**Option B2c: Expose Storage via EngineRenderer::init's EngineContext.**
Engine reads persisted state once on init. Pros: fits v1 pattern where CreationContext carried Storage. Cons: UI-thread writes don't naturally fit a per-engine-thread Storage.

**Recommendation:** Option B2a. Cleanest from the user's perspective, and `Storage::no_op()` stub is one small file. Feature-gate the *serde internals*, not the type.

**Implementation notes:**
- Make `pub struct Storage` with always-available methods. When `not(feature = "persistence")`: `set_value` becomes a no-op, `get_value` returns `None`, no disk I/O.
- When feature is on: the existing `InnerStorage` wiring applies.
- Add `pub mod storage` back to `lib.rs` (reverse the `pub(crate)` change in this plan cycle).
- Update `src/run.rs` to construct a Storage handle (persistence-enabled or stub) and pass to the UI closure.
- Requires B1 (persistence wiring) to be done first so the Storage instance passed to the closure is the same one Host uses.

**Files to touch:** `src/storage.rs`, `src/lib.rs`, `src/run.rs`, `src/host.rs`, every example using `run(...)`.

**Effort:** ~2–3 hours including the stub design and example updates.

**Acceptance:** Example app calls `storage.set_value("tab", &selected_tab)` in its UI closure, closes, reopens — tab is restored. Without the `persistence` feature enabled, the calls compile and silently do nothing.

---

### B3. `App::save()` + `auto_save_interval()` (periodic crash-safety flush)

**What v1 did:**
- `App::save(&mut self, storage: &mut Storage)` — user-defined hook called periodically + on `LoopExiting`.
- `App::auto_save_interval() -> Duration` — user override; defaults to 30 seconds.
- Integration ran a timer; when fired, called `app.save(storage)` then `storage.flush()`.

**What v2 has today:** Neither. No periodic flush, no explicit user-save hook.

**User impact:** If the app crashes between user actions (engine panic is recovered internally, but OS-level crashes aren't), unsaved preferences are lost. For apps with meaningful persisted state, this is an hourly-annoyance class bug.

**Restoration approach:**
- Add `RunOption::auto_save_interval: Option<Duration>` (default `Some(Duration::from_secs(30))`, `None` disables).
- Add a timer to `Host` (or drive from `run::about_to_wait` by checking elapsed time since last save — simpler, no extra thread).
- When interval elapses: call a hook on the UI closure OR, if we expose Storage directly per B2, users can flush from their own code. Alternative: a `EngineRenderer::save(&mut self, storage: &mut Storage)` method — but that mixes UI concerns into the engine thread, which is wrong.
- Cleanest fit: UI closure gets Storage (B2); user calls `storage.set_value(...)` whenever they want; egui-ash auto-flushes to disk at `auto_save_interval` cadence.

**Dependency:** Requires B2 (user-level Storage) to be done first. B3 then reduces to "add a timer + call storage.flush() periodically".

**Files to touch:** `src/host.rs` (timer + elapsed check), `src/types.rs` (RunOption field).

**Effort:** ~30 minutes on top of B2.

**Acceptance:** Set `auto_save_interval: Some(Duration::from_secs(2))`. App sets a value, waits 3 seconds, kills process with `kill -9`, relaunches — value is restored.

---

### B4. Programmatic exit (`ExitSignal` equivalent)

**What v1 did:**
- `ExitSignal::send(ExitCode)` — sent from any thread to request the event loop exit with a given exit code.
- `CreationContext::exit_signal` gave the user access to this at app creation time; they could stash it in `App`.
- Typical use: "Quit" menu item, error handler that aborts cleanly, background-thread fatal error.

**What v2 has today:** Nothing. The only way to exit is the user clicking the OS window-close button. No public API to trigger exit from code.

**User impact:** Apps with a "File → Quit" menu item cannot actually quit. Apps with a command-line mode (e.g., "load this file or exit if it's corrupt") cannot exit on error. Apps that want to exit after a timer elapses cannot.

**Restoration approach:**

Add an `exit(exit_code)` method to `EngineHandle<E>`, OR — cleaner — introduce a small new type passed to the UI closure. Option (a) reuses `EngineHandle`, consistent with B2c's pattern of expanding the handle. Option (b) keeps concerns separated.

**Recommendation:** Add to `EngineHandle`. It's already "the user's handle to host/runtime control".

**Proposed API:**
```rust
impl<E: EngineRenderer> EngineHandle<E> {
    pub fn exit(&self, code: std::process::ExitCode) {
        // Atomically flag a pending-exit state; checked in run::window_event or about_to_wait.
        // Uses the existing exit_tx channel owned by Host.
    }
}
```

**Implementation notes:**
- Host already has `exit_tx: Sender<ExitCode>` for its own use.
- `EngineHandle` needs access to it. Two ways: clone the sender into the handle (handle holds an `Option<Sender<ExitCode>>`), or thread an `Arc<Mutex<Option<ExitCode>>>` pending-exit state that the event loop polls.
- First is cleaner. Adjust `EngineHandle::new` to take `exit_tx: mpsc::Sender<ExitCode>`.
- In `run::window_event` or `run::about_to_wait`: after the UI closure runs, check if the handle has an exit pending; if so, drive the same cleanup path as `CloseRequested` (call `host.destroy()`, `self.host = None`, `event_loop.exit()`).

**Files to touch:** `src/host.rs` (EngineHandle), `src/run.rs` (exit-pending check).

**Effort:** ~1 hour.

**Acceptance:** Example adds a "Quit" button calling `engine.exit(ExitCode::SUCCESS)`. Clicking it closes the app with the given exit code.

---

### B5. ImageRegistry equivalent (user-registered textures)

**What v1 did:**
- `ImageRegistry` type exposed via `CreationContext::image_registry`.
- `register_user_texture(image_view: vk::ImageView, sampler: vk::Sampler) -> egui::TextureId` — user creates a Vulkan image, passes view+sampler to egui-ash, gets back a TextureId they can use in `egui::Image::new(..)`.
- `unregister_user_texture(id)` — caller-driven cleanup.
- Implementation (`main:src/renderer.rs:1385`+) used an mpsc channel; the main-thread renderer drained it each frame and created descriptor sets.

**What v2 has today:**
- Internal user-texture support exists in the compositor (`src/compositor.rs` has `ManagedTextures` and separate user-texture handling — see `ENGINE_VIEWPORT_USER_ID` const + `engine_viewport_descriptor_set` / `engine_viewport_texture_id` fields).
- The only externally-visible user texture is the engine viewport, via `EngineStatus::viewport_texture_id`.
- No public API for the user to register their own textures.

**User impact:** The entire `scene_view` / `native_image` / image-loading-by-hand pattern is broken. Apps that render thumbnails, 3D previews in UI panels, or use any `egui::Image` with a user-owned Vulkan image cannot do so.

**Restoration approach:**

This is the biggest item. The compositor already does most of the work (per above) — it allocates descriptor sets, supports `UPDATE_AFTER_BIND`. What's missing is a public handle.

**Proposed API:**
```rust
pub struct UserTextureHandle {
    id: egui::TextureId,
    // Back-reference to the compositor via a channel, like v1 did.
    tx: mpsc::Sender<UserTextureCommand>,
}

impl UserTextureHandle {
    pub fn id(&self) -> egui::TextureId { self.id }
    pub fn update(&self, image_view: vk::ImageView, sampler: vk::Sampler) { ... }
}

impl Drop for UserTextureHandle {
    fn drop(&mut self) { /* tx.send(Unregister(self.id)) */ }
}

pub struct ImageRegistry {
    tx: mpsc::Sender<UserTextureCommand>,
    next_id: AtomicU64,
}

impl ImageRegistry {
    pub fn register(&self, image_view: vk::ImageView, sampler: vk::Sampler) -> UserTextureHandle { ... }
}
```

**Where the user gets `ImageRegistry`:** Via `EngineHandle` (continuing the pattern), or via UI closure directly. Must be accessible *both* to the UI thread (for panel-mounted textures uploaded at runtime) and — ideally — to the engine thread (for textures the engine creates).

**Implementation notes:**
- Command channel drained in `Host::frame` before the compositor records. Commands:
  - `Register { id, image_view, sampler }` → compositor allocates descriptor set, writes it.
  - `Update { id, image_view, sampler }` → compositor rewrites the existing descriptor set (UPDATE_AFTER_BIND makes this safe mid-flight).
  - `Unregister { id }` → compositor frees descriptor, marks ID reusable.
- The compositor already has `_user_textures` scaffolding in `src/compositor.rs` — extend it, don't rewrite.
- `UserTextureHandle::Drop` emits Unregister. Users don't have to remember.
- Don't forget: user images need to be in `SHADER_READ_ONLY_OPTIMAL` layout when bound. Either document this requirement or do a one-shot transition on registration (simpler for the user, matches v1).

**Files to touch:** `src/compositor.rs` (extend existing user-texture paths), new `src/image_registry.rs`, `src/host.rs` (thread the registry into `Host`), `src/lib.rs` (re-export).

**Effort:** ~4–6 hours. Largest single B-item. Worth a separate plan doc of its own when execution starts.

**Dependency:** None (independent).

**Acceptance:** Port the v1 `native_image` example to v2 — load a PNG, register as user texture, display in `egui::Image`. Port v1 `scene_view` — engine renders a Suzanne model off-screen, registers the result as a user texture, appears in a resizable panel.

---

### B6. Raw `DeviceEvent` forwarding

**What v1 did:**
- `Event::DeviceEvent { device_id, event }` forwarded `winit::event::DeviceEvent` to `App::handle_event`. Variants include `MouseMotion { delta }` (raw mouse deltas, relevant for FPS-style cameras), `Key`, `Button`, etc.

**What v2 has today:** Not forwarded. `EngineEvent` has only processed events (`Pointer { position, buttons }` is already-processed per-viewport window coordinates).

**User impact:** FPS-style camera controls (raw mouse deltas + pointer lock) can't be implemented. Controller handling via DeviceEvent is impossible. Specific to 3D engine use cases — probably affects a smaller but more intense slice of the user base.

**Restoration approach:**

Add a new `EngineEvent` variant:
```rust
pub enum EngineEvent {
    // ... existing variants ...
    Device(winit::event::DeviceEvent),
}
```

In `src/run.rs`, implement `ApplicationHandler::device_event` and forward the winit enum into `EngineEvent::Device` via the event channel.

**Design consideration:** Should we expose the raw winit enum, or wrap it in our own type? v1 exposed raw winit (`pub use egui_winit::winit`). v2 keeps that re-export. So passing the raw winit type through is consistent.

**Files to touch:** `src/event.rs`, `src/run.rs`, `src/host.rs` (possibly forward through the event queue).

**Effort:** ~45 minutes.

**Acceptance:** Add a test engine that counts `DeviceEvent::MouseMotion` deltas. Moving the mouse while focused increments the counter.

---

### B7. `AppEvent` lifecycle events (Resumed / Suspended / LoopExiting etc.)

**What v1 did:**
- `Event::AppEvent { event: AppEvent }` forwarded winit's lifecycle callbacks: `NewEvents(StartCause)`, `Suspended`, `Resumed`, `AboutToWait`, `LoopExiting`, `MemoryWarning`.

**What v2 has today:**
- Internal handling in `run.rs` (the `resumed` callback creates `Host`). Not forwarded to the engine or UI closure.

**User impact:**
- **Android / mobile:** `Suspended` / `Resumed` control surface recreation. Without these, apps can't handle backgrounding correctly. (v2 doesn't claim mobile support today, so this is moot unless we want it in scope.)
- **Desktop:** `MemoryWarning` is the OS asking apps to release caches. `LoopExiting` is the last chance to save state (but v1 already hooks this via `App::save`; see B3).

**Restoration approach:**

Minimally, add `EngineEvent::Lifecycle(AppLifecycleEvent)` with the subset the user might actually care about:
```rust
pub enum AppLifecycleEvent {
    Suspended,
    Resumed,
    MemoryWarning,
    LoopExiting,
}
```

Forward from `run::ApplicationHandler::suspended` / `resumed` / `memory_warning` / `exiting` via the event channel.

**Note:** `Resumed` is *also* used internally by `run::resumed` to create `Host`. We don't need to forward that first Resumed (Host doesn't exist yet); only subsequent ones (e.g., Android app coming back from background).

**Files to touch:** `src/event.rs`, `src/run.rs`.

**Effort:** ~30 minutes.

**Acceptance:** Test engine logs `Lifecycle(Resumed)` on Android app foreground, `Suspended` on background.

---

### B8. AccessKit action request forwarding

**What v1 did:**
- `Event::AccessKitActionRequest(accesskit_winit::Event)` forwarded AccessKit action requests (screen reader commands, accessibility automation) to `App::handle_event`.
- Gated by `accesskit` feature; egui-winit delivers these via its event channel.

**What v2 has today:**
- `accesskit` feature flag still exists in `Cargo.toml`.
- No wiring. The feature compiles but no accessibility events reach the engine.

**User impact:** Screen reader users cannot interact with controls that rely on custom actions. This is an accessibility regression — real but affecting a small user population.

**Restoration approach:**

When feature is on, forward via an additional `EngineEvent::AccessKitAction(accesskit_winit::Event)` variant. Implement `ApplicationHandler::user_event` or use egui-winit's accessibility-adapter path.

**Complication:** AccessKit events in winit arrive via the `user_event` channel, not the normal window event. egui-winit has an adapter; we need to integrate it in `Host::new` (same place we construct `egui_winit::State`).

**Files to touch:** `src/event.rs`, `src/run.rs`, `src/host.rs`. Feature-gate all additions on `accesskit`.

**Effort:** ~1–2 hours (integrating AccessKit's event loop is the tricky part).

**Acceptance:** Feature-on compile; NVDA / VoiceOver sees and can invoke actions in a test example. (Harder to verify without the tooling; manual test acceptable.)

---

### B9. Multi-viewport egui UI support

**CAVEAT:** The earlier exhaustive audit claimed this was "KEPT" in v2. A direct grep of v2 source finds **zero** matches for `immediate_viewport`, `deferred_viewport`, `ViewportIdMap`, or `set_immediate_viewport_renderer`. This conflict needs to be resolved before execution — the "KEPT" claim may be wrong.

**What v1 did:**
- v1 called `egui_winit::State::new` with settings enabling multi-viewport.
- `set_immediate_viewport_renderer` installed a callback. When egui spawned an immediate viewport (typically for file dialogs, popups, tooltips rendered as separate OS windows), this callback created a new window, surface, swapchain.
- Deferred viewports (user-requested, `egui::Window::show_viewport`) emitted `ViewportEvent`; v1 routed these and called the user's `App::request_redraw(viewport_id)` per viewport.
- `examples/multi_viewports` demonstrated both.

**What v2 has today:**
- Verified: no multi-viewport code in `src/`. `Host` owns a single `window` and a single compositor swapchain.
- If user code calls `egui::Window::show_viewport(...)` inside the UI closure, it's silently ignored (egui expects the integration to spawn a window; v2 doesn't).

**User impact:**
- Egui file/color pickers that open in separate OS windows (immediate viewports) don't work.
- Deferred viewports (tool palettes that can be popped out to separate windows) don't work.
- This affects a meaningful slice of the user base; egui's multi-viewport feature is well-used.

**Restoration approach — requires design:**

This is the second-largest B-item after B5. At least:
1. Change `Host` to own a map of windows + compositors, not just one.
2. Install `set_immediate_viewport_renderer` on `egui::Context`.
3. Route `WindowEvent`s by `WindowId` to the correct viewport.
4. Handle viewport close / creation events.
5. Per-viewport clear color + present mode decisions.

Non-custom-rendering multi-viewport (just egui UI in each window) is the target. Custom rendering per viewport is intentionally out of scope (Category A4).

**Files to touch:** `src/host.rs` (major), `src/compositor.rs` (per-viewport compositors), `src/run.rs`, `src/types.rs` (maybe expose `PerViewportOptions`).

**Effort:** ~6–10 hours. **Worth its own plan document.**

**Recommendation before starting:** Verify whether the audit's "KEPT" claim was based on some v2 mechanism I haven't seen. Run a minimal test: example with `egui::Window::show_viewport(...)` inside the UI closure. Observe whether a new window appears. If yes, mechanism exists; find it. If no, full restoration needed.

**Acceptance:** Port v1 `multi_viewports` example's *egui UI* (not the custom triangle renderer). File dialog via `rfd` (which uses egui immediate viewports) opens in its own OS window.

---

## 6. Additional cleanup items (lesser)

Each of these is minor but worth tracking in this doc so nothing is forgotten.

### B10. `auto_save_interval` default value
Covered by B3; kept as a separate note because the default `Duration::from_secs(30)` should match v1 exactly.

### B11. `handle_event` was called for *every* event, not just engine events
v1's `App::handle_event` received everything (v1's `Event` enum was a superset). v2's `EngineRenderer::handle_event` only receives the filtered subset. Whether this is a regression depends on use case — most apps don't care about, say, pointer movement outside the engine viewport, so the filter is a feature. But some apps might. Document the narrowing in the migration guide.

### B12. `ExitCode` vs panic path in `run()`
v1 returned `ExitCode` cleanly; panics propagated. v2 same. No regression.

### B13. Default `clear_color`
v2 uses `[0.0, 0.0, 0.0, 1.0]` (black). v1 uses same. Verified identical.

### B14. Surface lifetime
v1 tied surface to the `Presenter`; v2 ties to `Compositor`. No user-visible difference.

---

## 7. Suggested execution order

Dependencies shown in parentheses.

1. **B1 — Persistence wiring.** Small, in-progress, gets the persistence module off "dead island" status. (None)
2. **B4 — ExitSignal equivalent on EngineHandle.** Quick win, unblocks any app that needs programmatic exit. (None)
3. **B6 — Raw DeviceEvent forwarding.** Quick win for 3D apps. (None)
4. **B7 — AppEvent lifecycle forwarding.** Quick win, pairs with B6. (None)
5. **B2 — User-level Storage API.** Modest but needs design decision first (Option B2a recommended). (Depends on B1 done first.)
6. **B3 — Periodic flush.** Trivial once B2 lands. (Depends on B2.)
7. **B9 — Multi-viewport egui UI.** Needs verification + design spike before execution. (Mostly independent; may interact with B5 if user textures cross viewport boundaries.)
8. **B5 — ImageRegistry.** Large. Worth its own plan doc. (Independent but do last because it touches compositor deeply.)
9. **B8 — AccessKit forwarding.** Optional for alpha if time-boxed; lowest user-impact-per-effort ratio. Could defer to 1.0.0-beta if needed, with the feature flag removed in Cargo.toml as a caveat.

**Estimated total effort:** 18–26 focused hours across 2–4 sessions. The big two (B5 + B9) dominate — each warrants a separate plan document before implementation.

---

## 8. Open design decisions needing input before coding

Each of these should be answered before the item is executed. Listed by severity.

1. **B2 Option A vs B vs C.** How does user code get access to Storage? Recommend A (closure 6th param with no-op stub). Confirm before implementing — signature change is user-visible.
2. **B5 handle obtain path.** Does `ImageRegistry` come from `EngineHandle`, from a new closure param, or from both UI closure and engine thread separately? Recommend: both. UI closure gets access via EngineHandle (consistent with B2c if we take that path), engine thread gets it in `EngineContext` so the engine can upload textures too.
3. **B9 scope.** Full multi-viewport-egui-UI restoration, or drop the feature explicitly and document as retired? If dropped, users who used `egui::Window::show_viewport` need a clear note in the migration guide.
4. **B8 or defer?** AccessKit forwarding is nontrivial; if we want alpha soon, deferring and removing the feature flag is a legitimate choice.
5. **New crate version:** Given the scale (Category A shifts) and the restoration work, is `1.0.0-alpha.1` the right version, or should we go `1.0.0-beta.1` once parity is restored? Alpha implies still-experimental; beta implies feature-complete but not stabilized. Probably alpha while B-work is landing, bump to beta when all B-items are done.

---

## 9. What this doc does NOT cover

- The code-quality polish tasks (crate-level `//!`, fatten `RunOption` field docs, README rewrite, CHANGELOG entry, version bump). Those stay in the main TaskList; they become meaningful only after Category B restoration is done.
- Tests / CI. Zero tests exist today. Worth separate planning; at minimum a compile-check on the examples in GitHub Actions before 1.0.0 stable.
- Migration guide writeup. The CHANGELOG 1.0.0-alpha entry will summarize, but a full `docs/migration-v1-to-v2.md` should be written in a later session — it should cover every Category A item with a code-level before/after example, and list every Category B item as "was X, is now Y".
- The `docs/known-limitations.md` cross-queue-family item. That one is explicitly deferred *past* 1.0.0 and tracked separately.

---

## 10. Closing notes

The distinction between Category A and Category B is a judgement call that should stay stable once accepted. If during execution one of the A items turns out to be wrong — that a user pattern genuinely needs restoration in some form — move it to B here and update the plan. Don't silently work around the split.

The alpha tag carries an implicit contract: `1.0.0-alpha.1` users accept that the API may still shift. But they *do not* accept silent feature loss relative to `0.4.0`. Category B restoration is about honoring that contract.

Commit the restoration work in small, verifiable PRs — one per B-item or logically grouped pair (e.g., B6 + B7 together). Each commit should be independently revertible if it breaks something; the examples are the integration test bed until we have real tests.
