//! Image loading example using the v2 `ColorEngine` + `egui_extras` image loaders.
//!
//! Demonstrates:
//! - Loading images from URLs and embedded SVGs via egui_extras
//! - Engine viewport in a side panel

use std::process::ExitCode;

use ash::vk;

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/color_engine.rs"]
mod color_engine;
use color_engine::{ColorEngine, ColorEngineState, ColorUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("egui_ash_images");
    let engine = ColorEngine::new();

    let options = egui_ash::RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash images")
                .with_inner_size(egui::vec2(800.0, 600.0)),
        ),
        follow_system_theme: true,
        ..Default::default()
    };

    let mut loaders_installed = false;

    egui_ash::run(
        "egui-ash-images",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &egui_ash::EngineStatus,
              _ui_state: &mut ColorUiState,
              _engine_state: &ColorEngineState,
              _handle: &egui_ash::EngineHandle<ColorEngine>,
              _storage: &mut egui_ash::Storage| {
            if !loaders_installed {
                egui_extras::install_image_loaders(ctx);
                loaders_installed = true;
            }

            // Left panel with engine viewport
            egui::SidePanel::left("viewport")
                .default_width(200.0)
                .show(ctx, |ui| {
                    ui.heading("Engine Viewport");
                    let available = ui.available_size();
                    ui.image(egui::load::SizedTexture::new(
                        status.viewport_texture_id,
                        available,
                    ));
                });

            // Central panel with images
            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.heading("Images");
                    ui.separator();

                    ui.label("Image loaded from URL:");
                    ui.add(
                        egui::Image::new("https://picsum.photos/seed/1.759706314/1024")
                            .corner_radius(10.0),
                    );

                    ui.separator();
                    ui.label("Embedded SVG:");
                    ui.image(egui::include_image!("./ferris.svg"));
                });
            });
        },
    )
}
