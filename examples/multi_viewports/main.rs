//! Multi-viewport egui-ash example — ports v1's `multi_viewports`
//! demo onto the v2 API. Drops v1's per-viewport custom Vulkan
//! rendering (Category A4, retired) and substitutes the host-
//! broadcast engine texture so every pop-out still shows live engine
//! output — in this demo a rotating Suzanne from `ModelEngine`.
//!
//! Exercises:
//! - `egui::Context::show_viewport_immediate` spawning a pop-out window.
//! - `egui::Context::show_viewport_deferred` with a persistent callback.
//! - Per-compositor engine-viewport descriptor broadcast (B9 decision #2).
//! - Per-compositor managed-texture (font atlas) broadcast so text
//!   renders in every pop-out, not just ROOT.
//! - Pop-out windows marked as xdg_toplevel children of ROOT so tiling
//!   Wayland compositors (niri, Hyprland) float them.
//!
//! Usage: `cargo run --release --example multi_viewports`

use ash::vk;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use egui_ash::winit::window::Theme;
use egui_ash::{EngineHandle, EngineStatus, RunOption};

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/model_engine.rs"]
mod model_engine;
use model_engine::{ModelEngine, ModelEngineState, ModelUiState};

/// Shared state read / written by every viewport's UI callback. The
/// deferred-viewport callback is stored `Fn + Send + Sync + 'static`, so
/// anything it touches lives behind an `Arc<Mutex>`. Same shape used in
/// ROOT and the immediate pop-out for symmetry.
struct Shared {
    theme: Theme,
    text: String,
    rotate_y: f32,
    auto_rotate: bool,
    bg_color: [f32; 3],
    show_immediate: bool,
    show_deferred: bool,
}

impl Shared {
    fn new() -> Self {
        Self {
            theme: Theme::Dark,
            text: "edit me — shared across all viewports".to_string(),
            rotate_y: 0.0,
            auto_rotate: true,
            bg_color: [0.0, 0.2, 0.4],
            show_immediate: false,
            show_deferred: false,
        }
    }
}

/// Mirrors v1's recurring content block: theme combo → hyperlink →
/// text_edit. Drawn in the root sidebar, the root floating window, and
/// the immediate pop-out so every viewport shows the same widgets.
fn content_block(ui: &mut egui::Ui, s: &mut Shared, combo_salt: &str) {
    ui.horizontal(|ui| {
        ui.label("Theme");
        let id = ui.make_persistent_id(combo_salt);
        egui::ComboBox::from_id_salt(id)
            .selected_text(format!("{:?}", s.theme))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut s.theme, Theme::Dark, "Dark");
                ui.selectable_value(&mut s.theme, Theme::Light, "Light");
            });
    });
    ui.separator();
    ui.hyperlink("https://github.com/emilk/egui");
    ui.separator();
    ui.text_edit_singleline(&mut s.text);
}

/// The engine-controls block: rotate slider, auto-rotate checkbox,
/// background color picker. Lives in the root sidebar and the deferred
/// pop-out's SidePanel — mirrors v1's rotate slider.
fn engine_block(ui: &mut egui::Ui, s: &mut Shared) {
    ui.label("Rotate Y");
    ui.add(egui::Slider::new(&mut s.rotate_y, -180.0..=180.0));
    ui.checkbox(&mut s.auto_rotate, "Auto rotate");
    ui.separator();
    ui.label("Background color");
    ui.color_edit_button_rgb(&mut s.bg_color);
}

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("multi_viewports");

    let engine = ModelEngine::new(&vulkan.instance, vulkan.physical_device);

    let options = RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.0, 0.0, 0.0, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash multi_viewports")
                .with_inner_size(egui::vec2(900.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: Theme::Dark,
        ..Default::default()
    };

    let shared = Arc::new(Mutex::new(Shared::new()));

    egui_ash::run(
        "multi_viewports",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &EngineStatus,
              ui_state: &mut ModelUiState,
              _engine_state: &ModelEngineState,
              _handle: &EngineHandle<ModelEngine>,
              _storage: &mut egui_ash::Storage| {
            let engine_tex = status.viewport_texture_id;

            // Apply the currently-selected theme at Context level so
            // every viewport reflects it.
            {
                let s = shared.lock().unwrap();
                match s.theme {
                    Theme::Dark => ctx.set_visuals(egui::Visuals::dark()),
                    Theme::Light => ctx.set_visuals(egui::Visuals::light()),
                }
            }

            // ─── ROOT sidebar ─────────────────────────────────────
            egui::SidePanel::left("root_side_panel")
                .default_width(260.0)
                .show(ctx, |ui| {
                    let mut s = shared.lock().unwrap();
                    ui.heading("Multi viewports");
                    ui.label("Hello egui multi viewports!");
                    ui.separator();
                    content_block(ui, &mut s, "theme_combo_root_side");
                    ui.separator();
                    engine_block(ui, &mut s);
                    ui.separator();
                    ui.checkbox(&mut s.show_immediate, "show immediate viewport");
                    ui.checkbox(&mut s.show_deferred, "show deferred viewport");
                    ui.separator();
                    ui.label(format!("Engine frames: {}", status.frames_delivered));
                    if let Some(ft) = status.last_frame_time {
                        ui.label(format!("Last frame: {:.2} ms", ft.as_secs_f64() * 1000.0));
                    }
                });

            // ─── ROOT floating window ─────────────────────────────
            egui::Window::new("My Window")
                .id(egui::Id::new("root_window"))
                .resizable(true)
                .default_width(280.0)
                .scroll([true, true])
                .show(ctx, |ui| {
                    let mut s = shared.lock().unwrap();
                    ui.heading("Multi viewports");
                    ui.label("Hello egui multi viewports!");
                    ui.separator();
                    content_block(ui, &mut s, "theme_combo_root_window");
                });

            // ─── ROOT central panel: engine texture ───────────────
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Root viewport");
                ui.label("Suzanne rotates in every open viewport.");
                let size = ui.available_size();
                ui.image(egui::load::SizedTexture::new(engine_tex, size));
            });

            // Publish current shared state to the engine thread.
            {
                let s = shared.lock().unwrap();
                ui_state.rotate_y = s.rotate_y;
                ui_state.auto_rotate = s.auto_rotate;
                ui_state.bg_color = s.bg_color;
            }

            // Pull toggle state once per frame.
            let (show_immediate, show_deferred) = {
                let s = shared.lock().unwrap();
                (s.show_immediate, s.show_deferred)
            };

            // ─── Immediate viewport ───────────────────────────────
            if show_immediate {
                let shared = Arc::clone(&shared);
                ctx.show_viewport_immediate(
                    egui::ViewportId::from_hash_of("immediate-viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("immediate-viewport")
                        .with_inner_size(egui::vec2(500.0, 400.0)),
                    move |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            shared.lock().unwrap().show_immediate = false;
                        }
                        egui::CentralPanel::default().show(ctx, |ui| {
                            let mut s = shared.lock().unwrap();
                            ui.heading("Immediate Viewport");
                            ui.label("immediate viewport!");
                            ui.separator();
                            content_block(ui, &mut s, "theme_combo_immediate");
                            ui.separator();
                            ui.checkbox(&mut s.show_immediate, "show immediate viewport");
                            ui.checkbox(&mut s.show_deferred, "show deferred viewport");
                            ui.separator();
                            let sz = ui.available_size();
                            ui.image(egui::load::SizedTexture::new(engine_tex, sz));
                        });
                    },
                );
            }

            // ─── Deferred viewport ────────────────────────────────
            if show_deferred {
                let shared_outer = Arc::clone(&shared);
                ctx.show_viewport_deferred(
                    egui::ViewportId::from_hash_of("deferred-viewport"),
                    egui::ViewportBuilder::default()
                        .with_title("deferred-viewport")
                        .with_inner_size(egui::vec2(600.0, 400.0)),
                    move |ctx, _class| {
                        if ctx.input(|i| i.viewport().close_requested()) {
                            shared_outer.lock().unwrap().show_deferred = false;
                        }
                        egui::SidePanel::left("deferred_side_panel").show(ctx, |ui| {
                            let mut s = shared_outer.lock().unwrap();
                            ui.heading("Deferred Viewport");
                            ui.label("deferred viewport!");
                            ui.separator();
                            engine_block(ui, &mut s);
                        });
                        egui::Window::new("My Window")
                            .id(egui::Id::new("deferred_window"))
                            .resizable(true)
                            .default_width(260.0)
                            .scroll([true, true])
                            .show(ctx, |ui| {
                                let mut s = shared_outer.lock().unwrap();
                                ui.heading("Deferred Viewport");
                                ui.label("deferred viewport!");
                                ui.separator();
                                content_block(ui, &mut s, "theme_combo_deferred");
                            });
                        egui::CentralPanel::default().show(ctx, |ui| {
                            let available = ui.available_size();
                            ui.image(egui::load::SizedTexture::new(engine_tex, available));
                        });
                    },
                );
            }
        },
    )
}
