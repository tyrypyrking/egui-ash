# Sub-Plan: B9 — Multi-Viewport egui UI Support

**Parent:** `2026-04-20-v1-parity-restoration.md` §5.B9
**Status:** Design approved 2026-04-21. Ready to execute in a dedicated session.
**Estimated effort:** 6–10 hours.

---

## Locked-in design decisions (2026-04-21)

Maintainer-approved decisions — do not re-open without cause:

1. **Immediate-viewport renderer state access: raw pointer in an `unsafe impl Send + Sync` wrapper.** egui's `set_immediate_viewport_renderer` requires `Fn + 'static` only (no `Send + Sync` bound), but a raw-pointer wrapper is still the cleanest. Matches the Round 1 restructure's aversion to uncontested `Arc<Mutex<>>`. Safety invariant: `Host` lives in a `Box` (address stable) and the renderer is only invoked synchronously from `context.run()` on the main thread. `Host::destroy` must reset the callback to a no-op before teardown.

2. **Engine viewport in non-root windows: YES — per-compositor descriptor set.** Each `Viewport`'s `Compositor` allocates its own `engine_viewport_descriptor_set` pointing at the engine's current image. When the engine produces a new frame, iterate all compositors and update each one's descriptor. Cost: one `vkUpdateDescriptorSets` per viewport per new engine frame — negligible.

3. **Deferred-viewport close: deferred-destruction queue, per-fence polling.** When a viewport closes (deferred ui_cb no longer invoked, or non-ROOT CloseRequested), move it to `Host::pending_destruction: Vec<PendingDestroy>`. Each frame, poll `vkGetFenceStatus` on each pending entry's collected in-flight fences — destroy only those where all fences have signalled. Zero-stall happy path.

4. **ROOT close teardown: single `device_wait_idle`, then reverse-order destroy.** One wait, one teardown loop over `self.viewports` in reverse insertion order. Skip per-viewport fence polling on exit — the app is terminating anyway, simplicity wins.

5. **Memory per viewport: full compositor each (no resource sharing for alpha).** ~30 MB per pop-out window accepted as the alpha cost. Revisit only if users report memory pressure.

6. **AccessKit on ROOT only for the alpha.** Non-root viewports do not get their own AccessKit adapter. Document as a limitation; per-viewport AccessKit is a 1.0.0-beta item.

7. **Persistence for deferred viewports: ROOT only for the alpha.** The saved HashMap still keys on `ViewportId`, but only ROOT is captured on flush. Document the policy; extend to stable user-chosen IDs in a later release.

8. **User textures (via `ImageRegistry`) in non-root viewports: ROOT only for the alpha.** Engine viewport works everywhere (per decision #2), but general user-registered textures require per-compositor descriptor propagation that's out of scope for the first multi-viewport cut. Document as a limitation; extend by broadcasting Register/Update/Unregister commands to all compositors in a later release.

---

---

## 1. Goal

Restore support for egui's multi-viewport feature: the integration must honour `egui::Window::show_viewport(...)` / `egui::Context::show_viewport_immediate(...)` by spawning additional OS windows with their own swapchains, each rendering the egui content assigned to its `ViewportId`.

**Out of scope:** per-viewport *custom* Vulkan rendering. That is Category A4 in the parent plan — an intentional architectural shift. v2's engine thread produces one output texture; that texture can appear in any egui panel in any viewport, but there is no per-viewport user render hook.

## 2. Why this matters

egui's multi-viewport machinery is used by:
- File dialogs (`rfd` with its `file_dialog` feature spawns a viewport)
- Color pickers that pop out into a separate window
- User-code `egui::Window::show_viewport()` for tool palettes
- Immediate-mode popups that prefer native windows over in-app floaters

Without this wiring, those features silently do nothing in v2 — the ViewportBuilder comes through the egui pipeline but egui-ash ignores it. Users upgrading from v0.4.0 lose these capabilities.

## 3. Required preliminary verification

**Before starting implementation**, verify the claim made by the earlier audit that multi-viewport is "KEPT" in v2. Direct-grep found zero references to `immediate_viewport` / `deferred_viewport` / `ViewportIdMap` in v2's `src/`, which contradicts that claim.

Verification steps:
1. Create a minimal test example that calls `egui::Window::new("...").show_viewport_immediate(...)` from the UI closure.
2. Run it and observe: does a new OS window appear? Does egui panic? Does rendering proceed silently?
3. Check whether `egui::Context::set_immediate_viewport_renderer` is being called anywhere in v2 — it was the lynchpin in v1 (`main:src/integration.rs:193`-ish).

If a new window appears and renders correctly, the audit was right and there's mystery wiring we haven't found — investigate before scope-changing. If no window appears (expected), proceed with full restoration below.

## 4. Current v2 architecture impact

v2's `Host<E>` owns exactly one `window`, one `compositor` (with one swapchain), and one `surface`. Multi-viewport support means all of these become per-viewport.

Key data structures that need to become maps:
- `Host::window` → `HashMap<ViewportId, egui_winit::winit::window::Window>`
- `Host::compositor` → `HashMap<ViewportId, Compositor>`
- `Host::surface` → owned by each Compositor instance (no change per se)
- `Host::egui_winit_state` → `HashMap<ViewportId, egui_winit::State>`

Additionally: `WindowId → ViewportId` mapping for routing `window_event` dispatches.

## 5. Proposed design

### 5.1 Viewport struct

```rust
struct Viewport {
    id: egui::ViewportId,
    ids: egui::ViewportIdPair,  // needed for info updates
    class: egui::ViewportClass,  // Root, Immediate, or Deferred
    builder: egui::ViewportBuilder,
    info: egui::ViewportInfo,
    is_first_frame: bool,
    window: winit::window::Window,
    state: egui_winit::State,
    compositor: Compositor,
    ui_cb: Option<Arc<DeferredViewportUiCallback>>,  // only for Deferred
}
```

Matches v1's `Viewport` struct at `main:src/integration.rs:37`. Keep the same shape — it's a known-good design.

### 5.2 Host struct changes

```rust
pub(crate) struct Host<E: EngineRenderer> {
    // Swap single fields for maps:
    viewports: HashMap<ViewportId, Viewport>,
    window_id_to_viewport: HashMap<WindowId, ViewportId>,
    focused_viewport: Option<ViewportId>,

    // Still singular:
    vulkan: VulkanContext,
    context: egui::Context,  // shared across all viewports
    render_target_pool: RenderTargetPool,  // engine still produces one output
    // ... engine thread / state / storage unchanged ...
}
```

Root viewport is `ViewportId::ROOT` — created in `Host::new` same as today, just stored in the map.

### 5.3 Immediate viewport renderer callback

Install via `egui::Context::set_immediate_viewport_renderer(...)` inside `Host::new`. The callback must:
1. Look up or create a Viewport for the requested `ViewportId`.
2. If new: create a winit Window, Surface, Compositor, egui-winit State.
3. Take egui input, run the egui immediate-viewport UI function, tessellate, record + submit, present.

The callback needs access to `&mut self`-ish state — impossible directly because `egui::Context` doesn't let you pass `&mut Host` through. v1 solved this by holding `Arc<Mutex<...>>` around the viewport map and associated state, so the closure captures `Arc` clones and can unlock inside.

v2 alternative: use raw pointers (the Round 1 restructure established that `Host` is stored in a `Box` so its address is stable). The immediate_viewport_renderer closure captures `*mut Host` and uses `&mut *ptr` inside, respecting the main-thread-only invariant (`Host` isn't `Send` anyway).

Recommended: **raw pointer**, matching the pattern already used elsewhere in the Round 1 restructure. This keeps the `Arc<Mutex<>>` count at zero — consistent with the design memo.

### 5.4 Deferred viewport handling

Deferred viewports (`egui::Context::show_viewport_deferred`) are user-opened and may live across many frames. They appear in each frame's `FullOutput::viewport_output` map.

In `Host::frame` after `context.run(...)` returns:
1. Iterate `viewport_output` map.
2. For each new viewport ID: create Viewport (window + compositor + state) and insert.
3. For each existing viewport: update its builder (size changes, title, etc.).
4. For each closed viewport: destroy and remove (careful with in-flight GPU work).
5. Run each deferred viewport's `ui_cb` to produce its own content, then paint.

v1 has this in `run_ui_and_record_paint_cmd` — reference when executing.

### 5.5 Window event routing

Current `window_event` always routes to the single host window. With many windows:
1. Look up `ViewportId` from `WindowId` via `window_id_to_viewport` map.
2. If not found: ignore (stale event after window closed).
3. Dispatch to that viewport's `egui_winit::State`.
4. Special-case CloseRequested: if it's the ROOT, exit the app; if a deferred/immediate viewport, close that viewport specifically.

### 5.6 Per-viewport compositor extent tracking

Current Compositor has `fallback_extent` for `currentExtent == u32::MAX` drivers. Each per-viewport compositor has its own. The plumbing is already correct at the single-compositor level; just needs to run per-viewport.

### 5.7 Engine viewport texture availability

The engine thread produces one output. When that texture appears in a panel inside a non-root window, the corresponding compositor needs to sample it. Each Compositor has its own descriptor set layout; they all need a descriptor set pointing at the engine's current image.

Two choices:
- **Per-compositor engine texture descriptor** — each compositor allocates a descriptor set, updated on each engine-frame reception. More memory, more descriptor-writes.
- **Shared engine texture descriptor** — one set, shared. Requires descriptor-indexing features we've already enabled, but cross-pool descriptor sharing is awkward in Vulkan.

Recommended: **per-compositor**, consistent with v1 and simpler to reason about. Each compositor handles its own user-texture slots identically.

## 6. Interaction with other B-items

### 6.1 B1 (persistence)
v1 persisted a `HashMap<ViewportId, WindowSettings>` — not just the root. Restoration needs to:
- On save: iterate `self.viewports`, capture each window's settings keyed by ViewportId.
- On load: apply each saved setting to the corresponding ViewportBuilder before creation.

B1's current implementation only handles ROOT — extend it after B9 lands.

### 6.2 B4 (programmatic exit)
Exit is already per-Host (not per-viewport). No change needed — closing a non-root window should close that viewport but not exit the app; closing ROOT exits as today.

### 6.3 B5 (ImageRegistry)
User textures are global — same `TextureId` should render in any viewport. Shared state between compositors is already part of the "per-compositor descriptor" plan in §5.7. User texture slot map needs sharing (one map, many compositor descriptor sets).

Alternative: keep one compositor hold the canonical user-texture state and others reference it. Cleaner than duplicating. Requires re-thinking Compositor ownership.

**Recommendation:** start without B5 (implement B9 first with engine-viewport-only rendering), then fold B5 into the multi-compositor design once B5's own shape is settled.

### 6.4 B6, B7 (events)
`EngineEvent::Device` and `EngineEvent::Lifecycle` are app-level — not per-viewport. No change.

### 6.5 B2, B3 (Storage)
UI closure has one Storage, regardless of viewport count. No change.

## 7. Implementation steps (execution order)

1. Verify (§3) — confirm multi-viewport is actually broken as suspected.
2. Introduce `Viewport` struct in a new `src/viewport.rs` (crate-internal). Move per-window state into it.
3. Refactor `Host` to hold `viewports: HashMap<ViewportId, Viewport>` + `window_id_to_viewport` map. Single-viewport works unchanged at this point (map size 1).
4. Refactor `handle_window_event` to route by WindowId → ViewportId → Viewport.
5. Install `egui::Context::set_immediate_viewport_renderer` in `Host::new` using raw-pointer-to-Host pattern. Handle the single-immediate-viewport case end-to-end.
6. Extend to multiple immediate viewports — map insert on new ID, reuse on existing.
7. Handle deferred viewports in `Host::frame` after `context.run(...)` based on `FullOutput::viewport_output`.
8. Close-viewport lifecycle: deferred viewport's UI closure drops → remove from map → destroy compositor → destroy window.
9. Extend B1 to persist/restore all viewports, not just ROOT.
10. Port v1's `multi_viewports` example (the egui-UI part, not the custom triangle rendering).

## 8. Residual risks (after design lock-in)

The design decisions at the top of this doc resolved the four originally-open questions. Remaining risks to watch during implementation:

1. **Deferred-viewport close race.** When a deferred viewport closes mid-frame, its `egui_winit::State` is still being used downstream in `handle_platform_output`. Queue destruction *after* the current frame completes, not immediately — implementation detail of decision #3 that still needs care.
2. **Fence-polling correctness in the pending-destruction queue.** Each closed viewport carries its compositor's full set of in-flight fences. Must wait on **all** of them (not just one) before tearing down — a single unsignalled fence means some command buffer still references the viewport's resources.
3. **egui-winit adapter drop ordering on ROOT close.** The AccessKit adapter holds a reference to the winit `Window`. Destroy the adapter before destroying the window to avoid dangling references in the OS-facing AccessKit tree.
4. **Raw-pointer safety invariant documentation.** The raw pointer pattern requires a clear SAFETY comment on the `unsafe impl Send + Sync` wrapper citing: (a) Host lives in a Box, (b) the renderer is only called synchronously from context.run, (c) Host::destroy resets the callback before teardown.

## 9. Non-goals for this sub-plan

- Per-viewport custom Vulkan rendering (Category A4; intentionally retired).
- Per-viewport present_mode / clear_color configuration (use global `RunOption` values for all).
- Android multi-display scenarios.
- AccessKit adapter per viewport (root-only for alpha — sufficient for screen readers).

## 10. Acceptance

1. `cargo clippy --workspace --all-targets` clean under default, `persistence`, `accesskit`, and combined features.
2. A v2 port of v1's `multi_viewports` example that demonstrates egui-UI multi-viewport — drop the custom triangle rendering (Category A4), keep the viewport-spawn UI, verify second OS windows appear with independent egui panels.
3. `rfd` file dialog (or equivalent multi-viewport-dependent crate) opens a native file-chooser window when invoked from inside egui-ash.
4. Window positions persist per-viewport when `persistent_windows: true` — extends B1.
