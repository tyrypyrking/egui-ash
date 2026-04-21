# Change Log

## [1.0.0-alpha.1] - 2026-04-21

**This is a ground-up rewrite.** Upgrading from 0.4.x requires rewriting your
app against the new `EngineRenderer` trait + UI closure model. See the
"Migrating from 0.4" table in the README for a symbol-by-symbol mapping.

### Architectural shift

- Render code runs on a dedicated **engine thread**, isolated from the UI
  thread by `catch_unwind`. Engine panics no longer take down the app; the
  UI shows a black viewport and the user can restart in-place via
  `EngineHandle::restart(new_engine)`.
- UI and engine state flow through `UiState` / `EngineState` associated
  types on `EngineRenderer`, exchanged lock-free via `arc-swap`.
- Frame handoff uses timeline semaphores on double-buffered render targets.

### Added

- `trait EngineRenderer` — the new user-render entry point (`init`,
  `render`, `handle_event`, `destroy`).
- `struct VulkanContext` — user-constructed bundle of Vulkan handles,
  replaces `AshRenderState<A>`.
- `struct EngineContext` — passed to `EngineRenderer::init` (device, queue,
  queue family, initial extent, format, queue mutex, image registry).
- `struct EngineHandle<E>` — passed to the UI closure; supports
  `restart(engine)`, `exit(code)`, and `image_registry()`.
- `struct EngineStatus` / `enum EngineHealth` / `enum EngineRestartError` —
  engine state and restart diagnostics observable from the UI closure.
- `struct RenderTarget` / `struct CompletedFrame` — opaque handles for the
  engine's write-target / submit-result plumbing.
- `enum EngineEvent` — `Pointer`, `Key`, `Scroll`, `Resize`, `Focus`,
  `Device(winit::DeviceEvent)`, `Lifecycle(AppLifecycleEvent)`, `Shutdown`.
- `enum AppLifecycleEvent` — `Suspended`, `Resumed`, `MemoryWarning`,
  `LoopExiting`.
- `struct ImageRegistry` / `struct UserTextureHandle` — replaces v0.4's
  `ImageRegistry::register_user_texture`; obtainable via
  `EngineHandle::image_registry()` (UI thread) or `EngineContext::image_registry`
  (engine thread).
- `RunOption::auto_save_interval: Option<Duration>` — default 30 s periodic
  flush of persisted state for crash safety.
- User-level persistent storage via `Storage::set_value` / `get_value`
  (feature-gated on `persistence`), passed to the UI closure as its 6th
  argument.
- Crate-level `//!` documentation and architecture diagram on docs.rs.
- **Multi-viewport egui UI support (B9).**
  `egui::Context::show_viewport_immediate` and
  `egui::Context::show_viewport_deferred` now spawn real pop-out
  windows, each with its own Vulkan surface, swapchain, and
  egui-winit adapter. Engine viewport texture broadcasts to every
  compositor (decision #2) so pop-outs render the engine scene too.
  Viewport teardown runs through a fence-poll deferred-destruction
  queue (decision #3) — closing a pop-out no longer stalls the main
  thread. Immediate-viewport renderer uses a raw-pointer `HostPtr`
  wrapper (decision #1) rather than `Arc<Mutex<>>`.

### Changed

- `fn run` signature is entirely new. Old:
  `run(app_id, creator, options) -> ExitCode`.
  New:
  `run(app_id, vulkan, engine, options, |ctx, status, ui_state, engine_state, engine_handle, storage| { ... }) -> ExitCode`.
- `RunOption::persistent_windows` / `persistent_egui_memory` now actually
  wire through — in 0.4 they were documented but only active in v1 code,
  and broken during the rewrite until this release.
- Persistence serialises window geometry and `egui::Memory` to
  `<data_dir>/<app_id>/app.ron` (unchanged from v0.4 schema).

### Removed (intentional architectural shifts)

- `trait App` and `trait AppCreator<A>` — replaced by `EngineRenderer` +
  closure.
- `struct CreationContext` / `struct AshRenderState<A>` — the user now
  constructs `VulkanContext` themselves.
- `enum HandleRedraw` — v0.4's per-viewport `HandleRedraw::Handle(fn)` for
  custom Vulkan rendering is retired. Move your rendering into
  `EngineRenderer::render`. The engine thread is the only custom-Vulkan
  path in v1.
- `trait Allocator` / `trait Allocation` / `struct AllocationCreateInfo` /
  `enum MemoryLocation` and the `gpu-allocator` feature flag. The user
  owns `VkDevice` and decides how to allocate; egui-ash does not require
  the allocator trait bound.
- `struct ExitSignal` — use `EngineHandle::exit(code)` instead.
- `struct EguiCommand` / `struct SwapchainUpdateInfo` — no longer needed;
  the host owns swapchain management.
- Per-viewport `request_redraw` callback.

### Known limitations

- **Cross-queue-family image transfer not yet implemented.**
  `host_queue_family_index` must equal `engine_queue_family_index`. Asserted
  at `Host::new`. Single-queue-family hardware (including single-queue Intel
  / AMD RADV) fully supported via `VulkanContext::queue_mutex`. See
  `docs/known-limitations.md`.
- **Managed textures (fonts, user textures) in non-root viewports.**
  egui's `textures_delta` is only applied to the compositor whose
  `context.run` produced it, so newly-created child viewports can
  start with an empty managed-texture table — text and egui-managed
  images may render incorrectly in pop-outs until the next texture
  refresh. Same class of issue as `ImageRegistry` textures being
  ROOT-only (decision #8). Full broadcast-delta fix scheduled
  post-alpha; see `docs/known-limitations.md`.
- **Non-root viewports skipped by persistence.**
  Window geometry is saved for ROOT only; pop-out positions don't
  persist across launches. Decision #7 for this alpha; extends in a
  later release.
- **AccessKit wired on ROOT only.**
  Screen readers see ROOT's egui widgets but not those inside
  pop-outs. Decision #6 for this alpha.

### New required Vulkan 1.2 features

Callers must enable the following on the `VkDevice`:

- `timelineSemaphore`
- `descriptorBindingSampledImageUpdateAfterBind`
- `descriptorBindingUpdateUnusedWhilePending`

See `examples/common/vkutils.rs` for a reference device-creation flow.

### Examples

Removed (covered v0.4 patterns that are retired): `native_image`.

New / rewritten against v1 API: `v2_simple`, `egui_ash_simple`,
`egui_ash_vulkan` (triangle engine), `images`, `scene_view` (model engine),
`tiles`, `multi_viewports` (B9 pop-out demo).

## [0.4.0] - 2024-01-14
### Added
- `egui_cmd.swapchain_recreate_required()` for change scale factor etc.

### Changed
- exit signal now receive `std::process::ExitCode`.

### Fixed
- fix error for ui zoom.
- fix error by forgetting to destroy image and image view when zoom factor changes.
- fix now restore main window position and size when `persistent_windows` is `true`.
  - Note: there is currently a bug in egui itself in saving the scale factor and window position. https://github.com/emilk/egui/issues/3797

### Update
- update egui from 0.24.2 to 0.25.0.
- update egui-winit from 0.24.1 to 0.25.0.

## [0.3.0] - 2024-01-05
### Added
- add exit_signal API to close exit app in code.
- add ability to change `vk::PresentModeKHR`.

### Fixed
- fix error in unregister_user_texture.

### Update
- update gpu-allocator from 0.24.0 to 0.25.0.

## [0.2.0] - 2024-01-01
### Changed
- remove Arc from AshRenderState.
- control flow to poll and present mode to FIFO.
- update example code.

### Fixed
- fix unused import.
- remove unnecessary build() of ash method.
- fix error when import spirv binary in examples.

## [0.1.1] - 2023-12-14
### Fixed
- fix README.md documentation.

## [0.1.0] - 2023-12-14
### Added
- initial release
