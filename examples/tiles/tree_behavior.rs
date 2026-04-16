use crate::model_engine::{ModelEngineState, ModelUiState};

// ─────────────────────────────────────────────────────────────────────────────
// Pane types
// ─────────────────────────────────────────────────────────────────────────────

pub enum Pane {
    Viewport,
    Properties,
    Hello,
}

// ─────────────────────────────────────────────────────────────────────────────
// TreeBehavior — carries per-frame state so pane_ui can access it
// ─────────────────────────────────────────────────────────────────────────────

pub struct TreeBehavior<'a> {
    pub viewport_texture_id: egui::TextureId,
    pub ui_state: &'a mut ModelUiState,
    pub engine_state: &'a ModelEngineState,
}

impl<'a> egui_tiles::Behavior<Pane> for TreeBehavior<'a> {
    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        match pane {
            Pane::Viewport => {
                let size = ui.available_size();
                ui.image(egui::load::SizedTexture::new(
                    self.viewport_texture_id,
                    size,
                ));
            }
            Pane::Properties => {
                ui.heading("Properties");
                ui.separator();

                ui.label("Rotate");
                ui.add(egui::Slider::new(
                    &mut self.ui_state.rotate_y,
                    -180.0..=180.0,
                ));
                ui.checkbox(&mut self.ui_state.auto_rotate, "Auto-rotate");
                ui.separator();

                ui.label("Background Color");
                ui.color_edit_button_rgb(&mut self.ui_state.bg_color);
                ui.separator();

                ui.label(format!("Frames: {}", self.engine_state.frame_count));
                ui.label(format!(
                    "Angle: {:.1}\u{00b0}",
                    self.engine_state.current_angle
                ));
            }
            Pane::Hello => {
                ui.heading("Hello");
                ui.label("Hello egui!");
                ui.separator();
                ui.hyperlink("https://github.com/emilk/egui");
            }
        }
        egui_tiles::UiResponse::None
    }

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::Viewport => "Viewport".into(),
            Pane::Properties => "Properties".into(),
            Pane::Hello => "Hello".into(),
        }
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        24.0
    }

    fn gap_width(&self, _style: &egui::Style) -> f32 {
        8.0
    }

    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tree construction helper
// ─────────────────────────────────────────────────────────────────────────────

pub fn create_tree() -> egui_tiles::Tree<Pane> {
    let mut tiles = egui_tiles::Tiles::default();

    let viewport = tiles.insert_pane(Pane::Viewport);
    let properties = tiles.insert_pane(Pane::Properties);
    let hello = tiles.insert_pane(Pane::Hello);

    // Right side: Properties and Hello stacked vertically
    let right = tiles.insert_vertical_tile(vec![properties, hello]);

    // Root: Viewport on left (larger), right panel on right
    let root = tiles.insert_horizontal_tile(vec![viewport, right]);

    // Set the split fraction so viewport gets ~70% of width
    if let Some(egui_tiles::Tile::Container(container)) = tiles.get_mut(root) {
        container.set_kind(egui_tiles::ContainerKind::Horizontal);
    }

    egui_tiles::Tree::new("tiles_tree", root, tiles)
}
