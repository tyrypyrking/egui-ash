//! Rotating triangle with egui controls using the v2 `TriangleEngine`.
//!
//! Demonstrates:
//! - `TriangleEngine` with a real Vulkan graphics pipeline (dynamic rendering)
//! - egui sidebar + floating window with rotation/theme controls
//! - Engine viewport displayed as an egui User texture

use std::process::ExitCode;

use ash::vk;
use egui_ash::winit::window::Theme;

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/triangle_engine.rs"]
mod triangle_engine;
use triangle_engine::{TriangleEngine, TriangleEngineState, TriangleUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("egui_ash_vulkan");

    let engine = TriangleEngine::new(vulkan.instance.clone(), vulkan.physical_device);

    let options = egui_ash::RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash vulkan")
                .with_inner_size(egui::vec2(1000.0, 800.0)),
        ),
        follow_system_theme: false,
        default_theme: Theme::Dark,
        ..Default::default()
    };

    let mut theme = Theme::Dark;
    let mut text = String::from("Hello text!");

    egui_ash::run(
        "egui-ash-vulkan",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &egui_ash::EngineStatus,
              ui_state: &mut TriangleUiState,
              engine_state: &TriangleEngineState,
              _handle: &egui_ash::EngineHandle<TriangleEngine>| {
            egui::SidePanel::left("controls")
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("Hello");
                    ui.label("Hello egui!");
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

                    ui.text_edit_singleline(&mut text);
                    ui.separator();

                    ui.label("Rotate");
                    ui.add(egui::Slider::new(
                        &mut ui_state.rotate_y,
                        -180.0..=180.0,
                    ));
                    ui.checkbox(&mut ui_state.auto_rotate, "Auto-rotate");
                    ui.separator();

                    ui.label(format!("Frames: {}", engine_state.frame_count));
                    ui.label(format!("Angle: {:.1}\u{00b0}", engine_state.current_angle));
                });

            egui::Window::new("My Window")
                .id(egui::Id::new("my_window"))
                .resizable(true)
                .scroll([true, true])
                .show(ctx, |ui| {
                    ui.heading("Hello");
                    ui.label("Hello egui!");
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

                    ui.text_edit_singleline(&mut text);
                    ui.separator();

                    ui.label("Rotate");
                    ui.add(egui::Slider::new(
                        &mut ui_state.rotate_y,
                        -180.0..=180.0,
                    ));
                    ui.checkbox(&mut ui_state.auto_rotate, "Auto-rotate");
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                let available = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    available,
                ));
            });

            match theme {
                Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
                Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
            }
        },
    )
}
