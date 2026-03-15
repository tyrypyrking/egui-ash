use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use egui_ash::{
    winit::window::Theme, App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption,
};
use gpu_allocator::vulkan::*;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

#[path = "../common/mod.rs"]
mod common;
use common::model_renderer;
use common::vkutils::*;

use model_renderer::{Renderer, RendererInnerCreationInfo};

struct MyApp {
    entry: Entry,
    instance: Instance,
    device: Device,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    renderer: Renderer,

    theme: Theme,
    text: String,
    rotate_y: f32,
}
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("my_side_panel").show(ctx, |ui| {
            ui.heading("Hello");
            ui.label("Hello egui!");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Theme");
                let id = ui.make_persistent_id("theme_combo_box_side");
                egui::ComboBox::from_id_salt(id)
                    .selected_text(format!("{:?}", self.theme))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.theme, Theme::Dark, "Dark");
                        ui.selectable_value(&mut self.theme, Theme::Light, "Light");
                    });
            });
            ui.separator();
            ui.hyperlink("https://github.com/emilk/egui");
            ui.separator();
            ui.text_edit_singleline(&mut self.text);
            ui.separator();
            ui.label("Rotate");
            ui.add(egui::widgets::Slider::new(
                &mut self.rotate_y,
                -180.0..=180.0,
            ));
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
                        .selected_text(format!("{:?}", self.theme))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.theme, Theme::Dark, "Dark");
                            ui.selectable_value(&mut self.theme, Theme::Light, "Light");
                        });
                });
                ui.separator();
                ui.hyperlink("https://github.com/emilk/egui");
                ui.separator();
                ui.text_edit_singleline(&mut self.text);
                ui.separator();
                ui.label("Rotate");
                ui.add(egui::widgets::Slider::new(
                    &mut self.rotate_y,
                    -180.0..=180.0,
                ));
            });

        match self.theme {
            Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
            Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
        }
    }

    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        HandleRedraw::Handle(Box::new({
            let renderer = self.renderer.clone();
            let rotate_y = self.rotate_y;
            move |size, egui_cmd| renderer.render(size.width, size.height, egui_cmd, rotate_y)
        }))
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.renderer.destroy();

            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader.destroy_surface(self.surface, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if self.debug_messenger != vk::DebugUtilsMessengerEXT::null() {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct MyAppCreator;
impl AppCreator<Arc<Mutex<Allocator>>> for MyAppCreator {
    type App = MyApp;

    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<Arc<Mutex<Allocator>>>) {
        // create vk objects
        let entry = create_entry();
        let (instance, debug_utils_loader, debug_messenger) =
            create_instance(&cc.required_instance_extensions, &entry);
        let surface_loader = create_surface_loader(&entry, &instance);
        let surface = create_surface(&entry, &instance, cc.main_window);
        let (physical_device, _physical_device_memory_properties, queue_family_index) =
            create_physical_device(
                &instance,
                &surface_loader,
                surface,
                &cc.required_device_extensions,
            );
        let (device, queue) = create_device(
            &instance,
            physical_device,
            queue_family_index,
            &cc.required_device_extensions,
        );
        let swapchain_loader = create_swapchain_loader(&instance, &device);
        let command_pool = create_command_pool(&device, queue_family_index);

        // create allocator
        let allocator = {
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .expect("Failed to create allocator")
        };
        let allocator = Arc::new(Mutex::new(allocator));

        // setup context
        cc.context.set_visuals(egui::style::Visuals::dark());

        let renderer_info = RendererInnerCreationInfo {
            physical_device,
            device: device.clone(),
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            allocator: allocator.clone(),
            surface,
            queue_family_index,
            queue,
            command_pool,
            width: 1000,
            height: 800,
        };

        let renderer = Renderer::new(renderer_info);

        let app = MyApp {
            entry,
            instance,
            device,
            debug_utils_loader,
            debug_messenger,
            physical_device,
            surface_loader,
            swapchain_loader,
            surface,
            queue,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            renderer,

            theme: if cc.context.style().visuals.dark_mode {
                Theme::Dark
            } else {
                Theme::Light
            },
            text: String::from("Hello text!"),
            rotate_y: 0.0,
        };
        let ash_render_state = AshRenderState {
            entry: app.entry.clone(),
            instance: app.instance.clone(),
            physical_device: app.physical_device,
            device: app.device.clone(),
            surface_loader: app.surface_loader.clone(),
            swapchain_loader: app.swapchain_loader.clone(),
            queue: app.queue,
            queue_family_index,
            command_pool: app.command_pool,
            allocator: allocator.clone(),
        };

        (app, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "egui-ash-vulkan",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("egui-winit-ash")
                    .with_inner_size(egui::vec2(1000.0, 800.0)),
            ),
            follow_system_theme: false,
            default_theme: Theme::Dark,
            ..Default::default()
        },
    )
}
