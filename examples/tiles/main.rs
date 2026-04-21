//! Tiled layout example using `egui_tiles` + `ModelEngine`.
//!
//! Demonstrates:
//! - Docked layout with Viewport, Properties, and Hello panes
//! - Engine viewport rendered in one of the panes
//! - Controls accessible from the Properties pane

use std::process::ExitCode;

use ash::vk;

mod tree_behavior;
use tree_behavior::TreeBehavior;

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/model_engine.rs"]
mod model_engine;
use model_engine::{ModelEngine, ModelEngineState, ModelUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("egui_ash_tiles");

    let engine = ModelEngine::new(&vulkan.instance, vulkan.physical_device);

    let options = egui_ash::RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash tiles")
                .with_inner_size(egui::vec2(800.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: egui_ash::winit::window::Theme::Dark,
        ..Default::default()
    };

    let mut tree = tree_behavior::create_tree();

    egui_ash::run(
        "egui-ash-tiles",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &egui_ash::EngineStatus,
              ui_state: &mut ModelUiState,
              engine_state: &ModelEngineState,
              _handle: &egui_ash::EngineHandle<ModelEngine>,
              _storage: &mut egui_ash::Storage| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let mut behavior = TreeBehavior {
                    viewport_texture_id: status.viewport_texture_id,
                    ui_state,
                    engine_state,
                };
                tree.ui(&mut behavior, ui);
            });
        },
    )
}
