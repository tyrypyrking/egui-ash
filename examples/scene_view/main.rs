//! Scene view example: engine viewport in a floating window.
//!
//! Demonstrates:
//! - `TriangleEngine` displayed in a resizable egui::Window
//! - Controls for rotation, auto-rotate, and background color

use std::process::ExitCode;

use ash::vk;

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/triangle_engine.rs"]
mod triangle_engine;
use triangle_engine::{TriangleEngine, TriangleEngineState, TriangleUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("egui_ash_scene_view");

    let engine = TriangleEngine::new(vulkan.instance.clone(), vulkan.physical_device);

    let options = egui_ash::RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash scene view")
                .with_inner_size(egui::vec2(800.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: egui_ash::winit::window::Theme::Dark,
        ..Default::default()
    };

    let mut show_scene_view = true;

    egui_ash::run(
        "egui-ash-scene-view",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &egui_ash::EngineStatus,
              ui_state: &mut TriangleUiState,
              engine_state: &TriangleEngineState,
              _handle: &egui_ash::EngineHandle<TriangleEngine>| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.heading("Scene View");
                ui.label("The engine viewport is shown in a floating window.");
                ui.separator();

                ui.hyperlink("https://github.com/emilk/egui");
                ui.separator();

                ui.checkbox(&mut show_scene_view, "Show scene view");
                ui.separator();

                ui.add(
                    egui::Slider::new(&mut ui_state.rotate_y, -180.0..=180.0)
                        .text("Rotation"),
                );
                ui.checkbox(&mut ui_state.auto_rotate, "Auto-rotate");
                ui.color_edit_button_rgb(&mut ui_state.bg_color);
                ui.separator();

                ui.label(format!("Frames: {}", engine_state.frame_count));
                ui.label(format!("Angle: {:.1}\u{00b0}", engine_state.current_angle));
            });

            egui::Window::new("Scene View")
                .open(&mut show_scene_view)
                .resizable(true)
                .default_size(egui::vec2(600.0, 400.0))
                .show(ctx, |ui| {
                    ui.label("Drag the rotation slider to control the triangle.");
                    let size = ui.available_size();
                    ui.image(egui::load::SizedTexture::new(
                        status.viewport_texture_id,
                        size,
                    ));
                });
        },
    )
}
