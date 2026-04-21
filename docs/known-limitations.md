# Known Limitations (1.0.0-alpha)

This document tracks intentional gaps in the 1.0.0-alpha surface that need
to close before 1.0.0 stable.

---

## 1. Single queue family required (cross-family ownership transfer NYI)

### Status
Enforced at runtime: `Host::new` asserts
`vulkan.host_queue_family_index == vulkan.engine_queue_family_index`.

### What's missing
The engine viewport image is shared between two queues:

- **host queue** — the compositor samples it as a texture in the egui render pass.
- **engine queue** — the user's `EngineRenderer` writes to it as a color attachment.

When these queues belong to **different queue families**, Vulkan requires a
**queue-family ownership transfer** to keep the image contents defined across
the handoff. Each transfer is two barriers that must be executed in lockstep:

- a **release** barrier on the source queue (ending write access, transferring
  ownership),
- a matching **acquire** barrier on the destination queue (claiming ownership,
  preparing for next-use access).

The engine side is wired up: `RenderTarget::acquire_barrier()` and
`RenderTarget::release_barrier()` are part of the public `EngineRenderer`
contract, and the example engines correctly emit both barriers in their
command buffers.

The **host side is not**. The compositor currently uses
`QUEUE_FAMILY_IGNORED` in every pipeline barrier it records, so the engine's
release is never matched by a host acquire (and the host never emits the
pre-render acquire that would pair with the engine's release from the previous
frame). On hardware where the two queues are in the same family, this is a
no-op — the current guard. On hardware where they're in separate families,
sampling the engine texture after its handoff would be undefined per the
Vulkan spec.

### Impact of the current workaround
On hardware exposing a single graphics queue family (Intel integrated, AMD
RADV on most generations), callers pick any queue from that family for both
`host_queue` and `engine_queue`. If the two handles end up being the same
physical `VkQueue`, they supply `VulkanContext::queue_mutex: Some(...)` to
serialise submits — that path is already tested.

On hardware with distinct graphics-capable queue families (some NVIDIA
configurations, SteamDeck-class AMD parts with separate transfer/compute
queues), users currently **cannot** use separate queue families for host and
engine. They must pick the same family for both, even if a second family
exists and would in principle be faster.

### What needs to happen to lift the restriction

1. In `render_targets.rs::RenderTargetPool::new`, reinstate cross-family
   detection: re-add the `cross_family: bool` field and the two
   `host_queue_family` / `engine_queue_family` parameters dropped when
   the restriction was introduced.

2. In `render_targets.rs::make_target`, reinstate the conditional
   construction of `acquire_barrier` / `release_barrier` — the deleted
   code is preserved in git history at the commit that introduced this
   limitation.

3. In `compositor.rs`, thread the most recently received `CompletedFrame`
   through to the command-buffer-recording path in `render_frame`. Right
   before the egui render pass samples the engine viewport texture, emit
   a `cmd_pipeline_barrier2` with a barrier that mirrors the engine's
   release (swap src/dst, matching layout/queue fields). This is the
   host-side acquire.

4. Additionally, on the *first* frame and after every swapchain rebuild,
   emit a host-side release barrier to hand the image back to the engine
   queue after the compositor's initial UNDEFINED → SHADER_READ_ONLY
   layout transition. This pairs with the engine's `acquire_barrier()` on
   its next submit.

5. Restore the deleted field on `CompletedFrame`: the compositor needs to
   know which release barrier to match against. Either plumb the full
   barrier through `CompletedFrame` or, equivalently, look it up from the
   `RenderTargetPool` by image index.

6. Remove the assertion in `Host::new`. Relax the `VulkanContext`
   documentation to reflect that distinct queue families are now permitted.

7. Add an example that exercises the cross-family path — ideally with a
   Vulkan validation-layer test harness to catch missing-barrier regressions.

### Effort estimate
Roughly 1–2 focused days. The engine-side API shape was preserved
specifically so that lifting the restriction does **not** require any
changes in downstream `EngineRenderer` implementations — the
always-`None` barriers will simply start returning `Some` again.

### Why this was deferred
Cross-family correctness requires validation-layer testing to trust, and
none of the current examples exercise the path. Shipping the alpha with an
enforced-single-family restriction is honest and narrow; silently producing
UB on a minority of hardware configurations would not be.

---

## 2. Managed textures (fonts, user textures) in non-root viewports

### Status
Multi-viewport rendering works (B9), but non-root viewports share the
egui `Context` with ROOT while each compositor maintains its **own**
managed-texture table. egui's `FullOutput::textures_delta` is consumed
once per `context.run` call — by whichever compositor rendered that
pass. That leaves other compositors' managed-texture tables empty
for any `TextureId` they didn't personally handle.

### What's missing
Two symptoms users will see:

- **Text / fonts may not render in pop-out windows.** On the first
  frame where the user calls `show_viewport_immediate` /
  `show_viewport_deferred`, ROOT's `context.run` has already
  consumed the font-atlas `textures_delta`; the child compositor
  never sees it. Subsequent frames' deltas are typically empty, so
  the atlas never reaches the child compositor until
  `Context::forget_all_images()` is called or fonts are changed.
- **`ImageRegistry` user textures only render in ROOT.** The
  `RegistryCommand` channel is consumed by ROOT's compositor; child
  compositors receive a pre-closed dummy receiver per B9 decision
  #8. Sampling a `UserTextureHandle::id()` inside a pop-out
  produces "unknown user texture id" — renders as nothing.

### What needs to happen to lift the restriction
The underlying design question (tracked in the B9 sub-plan) is
**per-compositor textures vs shared textures**. Two paths:

1. **Broadcast all `textures_delta` to every compositor.** Each
   compositor uploads its own copy of every texture. Easiest to
   implement; wastes GPU memory (e.g. one font atlas per pop-out).
   Requires extracting `Compositor::apply_texture_delta` as a public
   method and calling it on every compositor for each `context.run`
   delta, cloning the delta per call.
2. **Shared-image / per-compositor-descriptor split.** Textures live
   on `Host` as `vk::Image`/`vk::ImageView` handles; every compositor
   allocates a local descriptor set pointing at those shared views.
   Memory-optimal but refactors `Compositor::managed_textures` and
   `Compositor::user_textures` into two pieces.

### Impact of the current workaround
- Pop-outs that don't use text or `ImageRegistry`-registered textures
  render correctly today. The built-in `multi_viewports` example
  deliberately stays in this box — every pop-out just shows the
  engine viewport texture, which is fine because the engine viewport
  uses a reserved descriptor slot per compositor (decision #2) that
  IS populated.
- Pop-outs that use text will show missing glyphs until a font
  rebuild forces a full re-emit.

### Effort estimate
Path 1 (broadcast): ~2–3 hours. Path 2 (shared images): ~1 day.
Revisit based on user reports; the broadcast approach is enough
unless memory pressure surfaces.

### Why this was deferred
The B9 sub-plan explicitly punted this (decision #8). Shipping
multi-viewport support alongside a narrow, documented texture
limitation is more valuable than blocking B9 on the texture-sharing
design question. File a bug with a specific use case if this hits
you.
