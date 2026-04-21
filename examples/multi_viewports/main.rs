//! Multi-viewport egui-ash example — exercises B9's
//! `show_viewport_immediate` + `show_viewport_deferred` wiring.
//!
//! Demonstrates:
//! - ROOT window with a sidebar controlling two pop-out toggles.
//! - Immediate viewport spawned via `ctx.show_viewport_immediate` —
//!   renders inline while `ctx.run` is active, disappears when the
//!   toggle flips off.
//! - Deferred viewport spawned via `ctx.show_viewport_deferred` — its
//!   `ui_cb` is captured and invoked every frame until egui removes it
//!   (close button, or user unchecks the toggle).
//! - Engine viewport texture rendered in all three windows (root,
//!   immediate, deferred) — verifies B9 decision #2's per-compositor
//!   descriptor broadcast.
//!
//! v1 had a `multi_viewports` example combining this with custom
//! Vulkan rendering in every viewport (Category A4). v2 retires the
//! per-viewport custom-render hook; the engine texture substitutes.

use ash::vk;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use egui_ash::{EngineHandle, EngineStatus, RunOption};

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/color_engine.rs"]
mod color_engine;
use color_engine::{ColorEngine, ColorEngineState, ColorUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("multi_viewports");

    let engine = ColorEngine::new();

    let options = RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash multi_viewports")
                .with_inner_size(egui::vec2(900.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: egui_ash::winit::window::Theme::Dark,
        ..Default::default()
    };

    // Toggles are AtomicBool-behind-Arc because the deferred
    // viewport's ui_cb requires `Fn + Send + Sync + 'static`. Using
    // the same pattern for the immediate toggle keeps the code
    // symmetric.
    let show_immediate = Arc::new(AtomicBool::new(false));
    let show_deferred = Arc::new(AtomicBool::new(false));
    let mut clear_color = [0.2_f32, 0.3, 0.8];

    egui_ash::run(
        "multi_viewports",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &EngineStatus,
              ui_state: &mut ColorUiState,
              _engine_state: &ColorEngineState,
              _handle: &EngineHandle<ColorEngine>,
              _storage: &mut egui_ash::Storage| {
            // ─── ROOT sidebar ───────────────────────────────────────
            egui::SidePanel::left("controls")
                .default_width(260.0)
                .show(ctx, |ui| {
                    ui.heading("Multi-viewport demo");
                    ui.separator();

                    ui.label("Engine clear color:");
                    ui.color_edit_button_rgb(&mut clear_color);

                    ui.separator();

                    // Immediate toggle
                    let mut imm = show_immediate.load(Ordering::Relaxed);
                    if ui.checkbox(&mut imm, "Show immediate viewport").changed() {
                        show_immediate.store(imm, Ordering::Relaxed);
                    }

                    // Deferred toggle
                    let mut def = show_deferred.load(Ordering::Relaxed);
                    if ui.checkbox(&mut def, "Show deferred viewport").changed() {
                        show_deferred.store(def, Ordering::Relaxed);
                    }

                    ui.separator();

                    ui.label(format!("Engine frames: {}", status.frames_delivered));
                    if let Some(ft) = status.last_frame_time {
                        ui.label(format!("Last frame: {:.2} ms", ft.as_secs_f64() * 1000.0));
                    }
                });

            // ─── ROOT central panel: engine texture ─────────────────
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Root viewport");
                ui.label(
                    "The engine texture is rendered in every viewport. \
                     Open the pop-outs from the sidebar to see.",
                );
                let size = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    size,
                ));
            });

            ui_state.clear_color = clear_color;

            // ─── Immediate viewport ────────────────────────────────
            // Runs inline during ctx.run — we set up the window here
            // if the toggle is on, then the closure is invoked
            // synchronously by egui's `show_viewport_immediate`.
            if show_immediate.load(Ordering::Relaxed) {
                let engine_tex = status.viewport_texture_id;
                let show_immediate = Arc::clone(&show_immediate);
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("immediate-viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("Immediate viewport")
                        .with_inner_size(egui::vec2(500.0, 400.0)),
                    move |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            show_immediate.store(false, Ordering::Relaxed);
                        }
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.heading("Immediate viewport");
                            ui.label("Same engine texture, different window.");
                            let sz = ui.available_size();
                            ui.image(egui::load::SizedTexture::new(engine_tex, sz));
                        });
                    },
                );
            }

            // ─── Deferred viewport ─────────────────────────────────
            // Registers a callback that egui invokes each frame until
            // we stop calling `show_viewport_deferred` (or the user
            // closes the window, which we detect via close_requested).
            if show_deferred.load(Ordering::Relaxed) {
                let engine_tex = status.viewport_texture_id;
                let show_deferred = Arc::clone(&show_deferred);
                ctx.show_viewport_deferred(
                    egui::ViewportId::from_hash_of("deferred-viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("Deferred viewport")
                        .with_inner_size(egui::vec2(500.0, 400.0)),
                    move |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            show_deferred.store(false, Ordering::Relaxed);
                        }
                        egui::CentralPanel::default().show(ctx, |ui| {
                            ui.heading("Deferred viewport");
                            ui.label("Persistent pop-out driven by a stored callback.");
                            let sz = ui.available_size();
                            ui.image(egui::load::SizedTexture::new(engine_tex, sz));
                        });
                    },
                );
            }
        },
    )
}
