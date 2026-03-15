use std::sync::{Arc, Mutex};

use crate::common::scene::Scene;
use crate::common::scene_view_full::SceneView;

pub enum Pane {
    Hello,
    SceneView(SceneView),
    Properties(Arc<Mutex<Scene>>),
}
impl Pane {
    pub fn create_tree(scene: Arc<Mutex<Scene>>, scene_view: SceneView) -> egui_tiles::Tree<Pane> {
        let mut tiles = egui_tiles::Tiles::default();

        let tabs = vec![
            tiles.insert_pane(Pane::SceneView(scene_view)),
            tiles.insert_pane(Pane::Hello),
        ];
        let left = tiles.insert_vertical_tile(tabs);

        let tabs = vec![left, tiles.insert_pane(Pane::Properties(scene))];
        let root = tiles.insert_horizontal_tile(tabs);

        egui_tiles::Tree::new("root", root, tiles)
    }

    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
    ) -> egui_tiles::UiResponse {
        match self {
            Pane::Hello => {
                ui.heading("Hello");
                ui.label("Hello egui_tiles!");
                ui.hyperlink("https://github.com/rerun-io/egui_tiles");
            }
            Pane::SceneView(scene_view) => {
                ui.add(scene_view);
            }
            Pane::Properties(scene) => {
                let mut scene = scene.lock().unwrap();

                ui.with_layout(egui::Layout::top_down_justified(egui::Align::TOP), |ui| {
                    ui.heading("Properties");
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.style_mut().spacing.item_spacing.y = 8.0;
                        ui.group(|ui| {
                            ui.heading("Background");
                            ui.label("Color");
                            ui.color_edit_button_rgb(&mut scene.background.color);
                        });
                        ui.group(|ui| {
                            ui.heading("Suzanne");
                            ui.label("Rotation");
                            ui.horizontal(|ui| {
                                ui.label("X");
                                ui.add(egui::DragValue::new(&mut scene.suzanne.rotation_x));
                                ui.label("Y");
                                ui.add(egui::DragValue::new(&mut scene.suzanne.rotation_y));
                                ui.label("Z");
                                ui.add(egui::DragValue::new(&mut scene.suzanne.rotation_z));
                            });
                            ui.label("Diffuse Color");
                            ui.color_edit_button_rgb(&mut scene.suzanne.diffuse_color);
                            ui.label("Specular Color");
                            ui.color_edit_button_rgb(&mut scene.suzanne.specular_color);
                            ui.label("Shininess");
                            ui.add(egui::Slider::new(&mut scene.suzanne.shininess, 0.0..=100.0));
                        });
                        ui.group(|ui| {
                            ui.heading("Light");
                            ui.label("Position");
                            ui.horizontal(|ui| {
                                ui.label("X");
                                ui.add(egui::DragValue::new(&mut scene.light.position[0]));
                                ui.label("Y");
                                ui.add(egui::DragValue::new(&mut scene.light.position[1]));
                                ui.label("Z");
                                ui.add(egui::DragValue::new(&mut scene.light.position[2]));
                            });
                            ui.label("Intensity");
                            ui.add(egui::Slider::new(&mut scene.light.intensity, 0.0..=10.0));
                            ui.label("Color");
                            ui.color_edit_button_rgb(&mut scene.light.color);
                        });
                    });
                });
            }
        }
        Default::default()
    }

    pub fn title(&self) -> egui::WidgetText {
        match self {
            Pane::Hello => "Hello".into(),
            Pane::SceneView(_) => "Scene View".into(),
            Pane::Properties { .. } => "Properties".into(),
        }
    }
}
