mod compositor;
mod engine;
mod engine_thread;
pub mod event;
mod mailbox;
mod render_targets;
mod state_exchange;
pub mod types;

// These v1 modules still exist but will be removed in Task 9.
// Keep them for now to avoid breaking the build more than necessary.
mod allocator;
mod app;
mod integration;
mod renderer;
mod run;
mod utils;
mod viewport_context;

#[cfg(feature = "persistence")]
pub mod storage;

pub use egui_winit::winit;
pub use raw_window_handle;

// v2 public API
pub use engine::*;
pub use event::{EngineEvent, PointerButtons};
pub use types::*;

// v1 re-exports (will be removed in Task 9)
pub use allocator::*;
pub use app::*;
pub use renderer::*;
pub use run::*;

#[cfg(feature = "gpu-allocator")]
mod gpu_allocator;
