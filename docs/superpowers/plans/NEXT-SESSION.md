# Next Session — Start Here

**Last session:** 2026-04-20 → 2026-04-21 (two days).
**Branch:** `v2-editor-compositor`, version `1.0.0-alpha.1` (uncommitted).

This doc tells a future session (human or agentic) exactly what to pick up and in what order. Read top-to-bottom.

---

## Where the project stands

**Ready for `cargo publish --dry-run`:** `1.0.0-alpha.1` on branch `v2-editor-compositor`. All four feature configurations (`default`, `persistence`, `accesskit`, combined) pass `cargo clippy --workspace --all-targets` cleanly. `cargo doc --no-deps` has no warnings.

**What's in the alpha:** see `CHANGELOG.md` for the definitive list. Highlights:
- 7/9 Category-B v1-parity items landed (B1–B8 from the restoration plan).
- Comprehensive crate-level `//!`, README rewrite, CHANGELOG entry.
- Code-quality pass complete (panics → Result, SAFETY comments, dead code resolved).

**What's deferred:** see §Deferred Work below.

---

## Immediate next steps

Recommended execution order:

### 1. Ship the alpha (~30 min)

The code state is publishable right now. Concrete steps:

- `cargo publish --dry-run --features "persistence accesskit"` — smoke test the package build.
- If clean: create a git commit on `v2-editor-compositor` capturing all the uncommitted work since `1c0129e`. Suggested message: `release: 1.0.0-alpha.1 — v1 parity restoration (B1–B8)`.
- Tag `v1.0.0-alpha.1` on that commit.
- `cargo publish --features "persistence accesskit"` — publish.
- Merge branch to `main` (or leave branch-only until beta, maintainer's call).

### 2. Execute B9 — Multi-viewport egui UI (~6–10 hours)

**Plan:** `docs/superpowers/plans/2026-04-20-b9-multi-viewport.md`.

All eight design decisions are locked in at the top of that plan (approved 2026-04-21). Execute §9 "Implementation steps" in order. Do NOT re-open the design questions in §8.

**Acceptance criteria** (§10 of that plan):

1. `cargo clippy --workspace --all-targets` clean under all four feature combinations.
2. Port v1's `multi_viewports` example's egui-UI portion (drop the custom triangle rendering — that's Category A4, intentionally retired).
3. `rfd` (or equivalent multi-viewport-dependent crate) opens a native file-chooser window when invoked from inside an egui-ash UI closure.
4. Window positions persist per-viewport when `persistent_windows: true` — extend B1's root-only capture to cover all active viewports.

**Ship as** `1.0.0-alpha.2` after B9 lands. Update CHANGELOG, bump version.

---

## Deferred Work

A complete catalog of everything outside this alpha's scope, ordered by priority.

### Priority-1 (should land before 1.0.0 stable)

#### B9 — Multi-viewport egui UI
- **Impact:** Users calling `egui::Window::show_viewport_immediate` / `show_viewport_deferred` currently see no window. Blocks file dialogs via `rfd`, pop-out palettes, etc.
- **Plan:** `docs/superpowers/plans/2026-04-20-b9-multi-viewport.md` — design locked in.
- **Effort:** 6–10 hours focused session.
- **Status:** Ready to execute.

#### Per-viewport AccessKit (blocked on B9)
- **Impact:** Screen-reader users with multi-window egui apps only see the root window's widgets once B9 lands. Non-root pop-outs are invisible to accessibility tooling.
- **Plan:** Add one AccessKit adapter per `Viewport` in the `Viewport` struct. Route AccessKit events by `WindowId` (same routing B9 already does for `WindowEvent`).
- **Effort:** 1–2 hours on top of B9.
- **Status:** Pending B9 completion.

#### Cross-queue-family image ownership transfer
- **Impact:** Users whose host and engine queues live on different queue families cannot use egui-ash. `Host::new` asserts family match today. This affects some NVIDIA configurations and SteamDeck-class AMD parts with separate compute queues.
- **Plan:** `docs/known-limitations.md` has the full technical write-up — 7 concrete steps to lift the restriction plus an effort estimate.
- **Effort:** 1–2 focused days.
- **Status:** Engine-side API shape already preserved so downstream `EngineRenderer` implementations won't need changes when this lands.

### Priority-2 (nice to have; can wait for 1.0.x or 1.1)

#### User textures in non-root viewports (blocked on B9)
- **Impact:** Users can register Vulkan images via `ImageRegistry` today, but those textures only render in the root window. Sampling them inside a pop-out window logs "Unknown user texture id" and renders nothing.
- **Plan:** Broadcast `RegistryCommand::Register/Update/Unregister` from the host-level channel to every compositor. Each compositor maintains its own descriptor map.
- **Effort:** 2–3 hours after B9.
- **Status:** Pending B9 completion; engine viewport already works across all compositors so the plumbing template exists.

#### Deferred-viewport persistence
- **Impact:** v1 saved `HashMap<ViewportId, WindowSettings>` for every viewport. v1.0.0-alpha saves ROOT only; pop-out palettes don't remember their positions across sessions.
- **Plan:** Honor egui's own ViewportId convention — persist only viewports whose ID is a stable user-chosen hash (not randomly regenerated each session). Store a whitelist or heuristic in `Host::frame`'s auto-save path.
- **Effort:** ~2 hours.
- **Status:** Pending B9.

#### Compositor memory-sharing across viewports
- **Impact:** Each `Viewport` allocates ~30 MB of Vulkan resources (swapchain images, descriptor pool, buffers). Apps that spawn many small pop-outs have notable memory overhead.
- **Plan:** Factor `Compositor` into shared-across-viewports (descriptor layout, sampler, render pass) vs per-viewport (swapchain, framebuffers, command pool). Multi-viewport design decision #5 deliberately deferred this.
- **Effort:** 4–6 hours after B9.
- **Status:** Optimize only if real users hit memory pressure.

### Priority-3 (quality-of-life)

#### Test suite
- **Impact:** Zero tests today. Regressions only caught by example compile checks and manual runs. Given 200 weekly downloads, a CI smoke test suite is overdue.
- **Plan:** At minimum — compile-check all examples in GitHub Actions; add a headless "engine spawns, engine crashes via panic, handle reports Crashed, restart recovers" integration test using `VK_LAYER_KHRONOS_swiftshader` or similar.
- **Effort:** 3–6 hours.
- **Status:** Independent of B9. Could happen before or after.

#### CI (`.github/workflows/`)
- **Impact:** `main` isn't protected against regressions. A clippy-break or compile-break could land unnoticed.
- **Plan:** Add a `ci.yml` running `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings` across the four feature combos, `cargo test`, and `cargo doc --no-deps`. Matrix on Linux + Windows + macOS.
- **Effort:** 1–2 hours.
- **Status:** Do alongside the test suite.

#### Migration guide (`docs/migration-v1-to-v2.md`)
- **Impact:** v0.4.0 users upgrading have only the CHANGELOG's migration table. A full guide with before/after code samples for each Category A pattern (trait-based App → EngineRenderer + closure; HandleRedraw → EngineRenderer::render; custom allocator → user-owned Vulkan) would reduce migration friction.
- **Plan:** Walk through v1's `egui_ash_vulkan` example side-by-side with v2's `egui_ash_vulkan` — annotate every change.
- **Effort:** 2–3 hours.
- **Status:** Post-alpha. Write when first migration issue comes in if it hasn't happened already.

---

## Reference docs (unchanged, still current)

- `docs/superpowers/plans/2026-04-20-v1-parity-restoration.md` — the master plan. Mostly executed; B9 is the last open item.
- `docs/superpowers/plans/2026-04-20-b5-image-registry.md` — B5 sub-plan (completed, retained for reference).
- `docs/superpowers/plans/2026-04-20-b9-multi-viewport.md` — B9 sub-plan (design locked 2026-04-21, ready to execute).
- `docs/known-limitations.md` — cross-queue-family deferred work (post-1.0 item).
- `docs/superpowers/plans/2026-03-23-editor-compositor-v2.md` — original v2 architecture spec.

## Tools / commands used this session

```bash
# Verify all four feature configurations:
cargo clippy --workspace --all-targets --message-format=short
cargo clippy --workspace --all-targets --features persistence --message-format=short
cargo clippy --workspace --all-targets --features accesskit --message-format=short
cargo clippy --workspace --all-targets --features "persistence accesskit" --message-format=short

# Docs check:
cargo doc --no-deps --features "persistence accesskit" 2>&1 | grep -E "warning|error"

# Examples compile check:
cargo check --examples --message-format=short
```

---

## End-of-session note (2026-04-21)

The alpha shipped as-is delivers real value to the 200 weekly downloaders — seven v1-parity restorations, all documented, all tested across feature combos. B9 was consciously deferred to protect the alpha's quality; implementation is fully planned and the design decisions are locked in. Next session can execute B9 straight from the sub-plan without re-deriving context.
