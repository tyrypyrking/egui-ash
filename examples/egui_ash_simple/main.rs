//! Simple egui widget demo using the v2 `ColorEngine`.
//!
//! Demonstrates:
//! - Theme switching (Dark/Light)
//! - Text fields, drag values, sliders
//! - Japanese font loading
//! - Floating egui::Window
//! - Engine viewport in a side panel

use std::process::ExitCode;
use std::sync::Arc;

use ash::vk;
use egui::FontData;
use egui_ash::winit::window::Theme;

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/color_engine.rs"]
mod color_engine;
use color_engine::{ColorEngine, ColorEngineState, ColorUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("egui_ash_simple");
    let engine = ColorEngine::new();

    let options = egui_ash::RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash simple")
                .with_inner_size(egui::vec2(900.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: Theme::Dark,
        ..Default::default()
    };

    let mut theme = Theme::Dark;
    let mut text = String::from("Hello text!");
    let mut value = 0.0_f32;
    let mut fonts_loaded = false;

    egui_ash::run(
        "egui-ash-simple",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &egui_ash::EngineStatus,
              _ui_state: &mut ColorUiState,
              _engine_state: &ColorEngineState,
              _handle: &egui_ash::EngineHandle<ColorEngine>| {
            // Load Japanese font once
            if !fonts_loaded {
                let mut fonts = egui::FontDefinitions::default();
                fonts.font_data.insert(
                    "NotoSansJP".to_owned(),
                    Arc::new(FontData::from_static(include_bytes!(
                        "../common/fonts/NotoSansJP-VariableFont_wght.ttf"
                    ))),
                );
                fonts
                    .families
                    .get_mut(&egui::FontFamily::Proportional)
                    .unwrap()
                    .insert(0, "NotoSansJP".to_owned());
                ctx.set_fonts(fonts);
                fonts_loaded = true;
            }

            // Apply theme
            match theme {
                Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
                Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
            }

            // Left panel with controls
            egui::SidePanel::left("controls")
                .default_width(300.0)
                .show(ctx, |ui| {
                    ui.heading("Hello");
                    ui.label("Hello egui!");
                    ui.separator();

                    ui.label("You can use non-Latin characters by setting your font.");
                    ui.label("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{3001}egui\u{ff01}");
                    ui.label("\u{30d5}\u{30a9}\u{30f3}\u{30c8}\u{3092}\u{8a2d}\u{5b9a}\u{3059}\u{308b}\u{3053}\u{3068}\u{3067}\u{65e5}\u{672c}\u{8a9e}\u{3082}\u{8868}\u{793a}\u{3067}\u{304d}\u{307e}\u{3059}\u{3002}");
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("Theme");
                        let id = ui.make_persistent_id("theme_combo_box_side");
                        egui::ComboBox::from_id_salt(id)
                            .selected_text(format!("{:?}", theme))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut theme, Theme::Dark, "Dark");
                                ui.selectable_value(&mut theme, Theme::Light, "Light");
                            });
                    });
                    ui.separator();

                    ui.hyperlink("https://github.com/emilk/egui");
                    ui.separator();

                    ui.label("Value");
                    ui.add(egui::widgets::DragValue::new(&mut value));
                    ui.add(egui::widgets::Slider::new(&mut value, -10.0..=10.0));
                    ui.separator();

                    ui.text_edit_singleline(&mut text);
                    ui.separator();

                    ui.label("Press ESC to close the application.");
                });

            // Central panel with engine viewport
            egui::CentralPanel::default().show(ctx, |ui| {
                let available = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    available,
                ));
            });

            // Floating window
            egui::Window::new("My Window")
                .id(egui::Id::new("my_window"))
                .resizable(true)
                .scroll([true, true])
                .show(ctx, |ui| {
                    ui.heading("Hello");
                    ui.label("Hello egui!");
                    ui.separator();
                    ui.label("You can use non-Latin characters by setting your font.");
                    ui.label("\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{3001}egui\u{ff01}");
                    ui.label("\u{30d5}\u{30a9}\u{30f3}\u{30c8}\u{3092}\u{8a2d}\u{5b9a}\u{3059}\u{308b}\u{3053}\u{3068}\u{3067}\u{65e5}\u{672c}\u{8a9e}\u{3082}\u{8868}\u{793a}\u{3067}\u{304d}\u{307e}\u{3059}\u{3002}");
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Theme");
                        let id = ui.make_persistent_id("theme_combo_box_window");
                        egui::ComboBox::from_id_salt(id)
                            .selected_text(format!("{:?}", theme))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut theme, Theme::Dark, "Dark");
                                ui.selectable_value(&mut theme, Theme::Light, "Light");
                            });
                    });
                    ui.separator();
                    ui.hyperlink("https://github.com/emilk/egui");
                    ui.separator();
                    ui.label("Value");
                    ui.add(egui::widgets::DragValue::new(&mut value));
                    ui.add(egui::widgets::Slider::new(&mut value, -10.0..=10.0));
                    ui.separator();
                    ui.text_edit_singleline(&mut text);
                });
        },
    )
}
