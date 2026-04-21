//! egui integration for [ash](https://github.com/ash-rs/ash) (Vulkan).
//!
//! egui-ash manages the winit event loop, Vulkan swapchain, egui texture
//! uploads, and frame composition so that consumers only implement an
//! [`EngineRenderer`] and a UI closure.
//!
//! # Architecture (1.0.0-alpha)
//!
//! The v2 design runs user render code on a **dedicated engine thread**,
//! isolated from the UI thread by `catch_unwind`. The engine produces one
//! output image per frame; the host thread composites egui over it and
//! presents to the swapchain. UI and engine state are exchanged lock-free
//! via [`arc-swap`](https://crates.io/crates/arc-swap), keeping both sides
//! wait-free on the hot path.
//!
//! ```text
//!  ┌──────────────── main thread ────────────────┐   ┌─── engine thread ───┐
//!  │  winit event loop                           │   │                     │
//!  │    │                                        │   │                     │
//!  │    ├─► egui::Context::run(ui_closure)  ◄────┼───┼── engine_state  ──► │
//!  │    │    └──► user draws panels, reads       │   │  (reader side)      │
//!  │    │         engine_state, writes ui_state ─┼───┼──►  ui_state ──►    │
//!  │    │                                        │   │   (writer side)     │
//!  │    ├─► Compositor composites + presents  ◄──┼───┼── RenderTarget ──►  │
//!  │    │                                        │   │   (engine renders)  │
//!  └────┴────────────────────────────────────────┘   └─────────────────────┘
//! ```
//!
//! # Quick start
//!
//! Implement [`EngineRenderer`] for your renderer, construct a
//! [`VulkanContext`] (all Vulkan objects you own), and call [`run`]:
//!
//! ```ignore
//! let vulkan: egui_ash::VulkanContext = build_vulkan_context(...);
//! egui_ash::run(
//!     "my-app",
//!     vulkan,
//!     MyEngine::default(),
//!     egui_ash::RunOption::default(),
//!     |ctx, status, ui_state, engine_state, engine, storage| {
//!         egui::CentralPanel::default().show(ctx, |ui| {
//!             ui.image(status.viewport_texture_id);
//!         });
//!     },
//! );
//! ```
//!
//! # Required Vulkan 1.2 features
//!
//! Callers **must** enable the following on the `VkDevice` before handing
//! it to [`VulkanContext`] (all core in Vulkan 1.2):
//!
//! - `timelineSemaphore` — frame handoff between engine and host.
//! - `descriptorBindingSampledImageUpdateAfterBind`
//! - `descriptorBindingUpdateUnusedWhilePending`
//!
//! The last two let the compositor rebind user textures while previous
//! frames are still in flight — used by egui's managed texture deltas and
//! by [`ImageRegistry`] for user-owned images.
//!
//! # Feature flags
//!
//! | Feature       | Effect                                                                     |
//! |---------------|----------------------------------------------------------------------------|
//! | `persistence` | Enables [`Storage`] disk I/O, window geometry + egui memory auto-save.     |
//! | `accesskit`   | Wires egui-winit's AccessKit adapter so screen readers see egui widgets.   |
//! | `wayland`     | Passes through to egui-winit (Linux Wayland backend).                      |
//! | `x11`         | Passes through to egui-winit (Linux X11 backend).                          |
//!
//! # Known limitations (1.0.0-alpha)
//!
//! These are documented in `docs/known-limitations.md`:
//!
//! - **Single queue family only.** `host_queue_family_index` must equal
//!   `engine_queue_family_index` until cross-family image ownership
//!   transfer is wired up host-side.
//! - **Managed textures (fonts / egui icons) in non-root viewports.**
//!   egui's textures_delta is only applied to the compositor whose
//!   `context.run` produced it, so pop-out windows created via
//!   `show_viewport_immediate` / `show_viewport_deferred` can start
//!   with an empty managed-texture table and render text incorrectly
//!   until the atlas next refreshes. `ImageRegistry`-registered user
//!   textures are similarly ROOT-only for the alpha. Broadcast-delta
//!   fix scheduled post-alpha.
//! - **Per-viewport persistence / AccessKit.** Window geometry and the
//!   AccessKit adapter are wired on ROOT only; pop-out positions
//!   don't persist across launches and screen readers don't see
//!   pop-out widgets. Both are post-alpha items.
//! - **Per-viewport custom rendering retired.** v1's `HandleRedraw::Handle`
//!   pattern is intentionally gone — use [`EngineRenderer`] instead.

mod compositor;
mod engine;
mod engine_thread;
pub mod event;
mod host;
mod image_registry;
mod mailbox;
mod render_targets;
mod run;
mod state_exchange;
pub mod types;
mod viewport;

pub mod storage;

pub use egui_winit::winit;
pub use raw_window_handle;

pub use engine::*;
pub use event::{AppLifecycleEvent, EngineEvent, PointerButtons};
pub use host::EngineHandle;
pub use image_registry::{ImageRegistry, UserTextureHandle};
pub use run::*;
pub use storage::Storage;
pub use types::*;
