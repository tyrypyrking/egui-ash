//! Minimal v2 egui-ash example: a `ColorEngine` that clears render targets to
//! a solid color controlled by an egui sidebar.
//!
//! Demonstrates:
//! - `EngineRenderer` trait implementation (via shared ColorEngine)
//! - egui sidebar with color picker + engine status
//! - Engine viewport displayed as an egui User texture
//! - Crash test button + restart capability

use ash::vk;
use std::process::ExitCode;

use egui_ash::{EngineHandle, EngineStatus, RunOption};

#[path = "../common/mod.rs"]
mod common;
use common::vkutils;

#[path = "../common/color_engine.rs"]
mod color_engine;
use color_engine::{ColorEngine, ColorEngineState, ColorUiState};

fn main() -> ExitCode {
    let (vulkan, _resources) = vkutils::create_vulkan_context("v2_simple");

    let engine = ColorEngine::new();

    let options = RunOption {
        present_mode: vk::PresentModeKHR::FIFO,
        clear_color: [0.1, 0.1, 0.1, 1.0],
        viewport_builder: Some(
            egui::ViewportBuilder::default()
                .with_title("egui-ash v2 simple")
                .with_inner_size(egui::vec2(900.0, 600.0)),
        ),
        follow_system_theme: false,
        default_theme: egui_ash::winit::window::Theme::Dark,
        ..Default::default()
    };

    let mut clear_color = [0.2_f32, 0.3, 0.8];

    egui_ash::run(
        "v2_simple",
        vulkan,
        engine,
        options,
        move |ctx: &egui::Context,
              status: &EngineStatus,
              ui_state: &mut ColorUiState,
              engine_state: &ColorEngineState,
              handle: &EngineHandle<ColorEngine>| {
            egui::SidePanel::left("controls")
                .default_width(250.0)
                .show(ctx, |ui| {
                    ui.heading("Color Engine");
                    ui.separator();

                    ui.label("Clear Color");
                    ui.color_edit_button_rgb(&mut clear_color);
                    ui.add(
                        egui::Slider::new(&mut clear_color[0], 0.0..=1.0)
                            .text("R"),
                    );
                    ui.add(
                        egui::Slider::new(&mut clear_color[1], 0.0..=1.0)
                            .text("G"),
                    );
                    ui.add(
                        egui::Slider::new(&mut clear_color[2], 0.0..=1.0)
                            .text("B"),
                    );

                    ui.separator();
                    ui.heading("Engine Status");

                    let health_text = match &status.health {
                        egui_ash::EngineHealth::Starting => "Starting...".to_string(),
                        egui_ash::EngineHealth::Running => "Running".to_string(),
                        egui_ash::EngineHealth::Stopped => "Stopped".to_string(),
                        egui_ash::EngineHealth::Crashed { message } => {
                            format!("CRASHED: {}", message)
                        }
                    };
                    ui.label(format!("Health: {}", health_text));
                    ui.label(format!("Frames delivered: {}", status.frames_delivered));

                    if let Some(ft) = status.last_frame_time {
                        ui.label(format!("Last frame: {:.2}ms", ft.as_secs_f64() * 1000.0));
                    }

                    ui.label(format!(
                        "Engine frame count: {}",
                        engine_state.frame_count
                    ));

                    ui.separator();

                    // Crash test button
                    if status.health.is_alive()
                        && ui.button("Crash Engine").clicked()
                    {
                        ui_state.should_crash = true;
                    }

                    // Restart button (only if crashed/stopped)
                    let can_restart = status.health.is_crashed()
                        || matches!(status.health, egui_ash::EngineHealth::Stopped);
                    if can_restart && ui.button("Restart Engine").clicked() {
                        let _ = handle.restart(ColorEngine::new());
                    }
                });

            egui::CentralPanel::default().show(ctx, |ui| {
                let available = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    status.viewport_texture_id,
                    available,
                ));
            });

            // Publish current color to engine
            ui_state.clear_color = clear_color;
        },
    )
}
