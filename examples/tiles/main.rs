use ash::{ext::debug_utils, khr::surface, vk, Device, Entry, Instance};
use egui_ash::{App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption};
use gpu_allocator::vulkan::*;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

mod pane;
mod scene;
mod scene_view;
mod tree_behavior;
mod vkutils;

use pane::Pane;
use scene::Scene;
use scene_view::SceneView;
use tree_behavior::TreeBehavior;
use vkutils::*;

struct MyApp {
    _entry: Arc<Entry>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    surface_loader: Arc<surface::Instance>,
    surface: vk::SurfaceKHR,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    scene_view: SceneView,
    tree: egui_tiles::Tree<Pane>,
    tree_behavior: TreeBehavior,
}
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let behavior = &mut self.tree_behavior;
            self.tree.ui(behavior, ui);
        });
    }

    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        self.scene_view.render();
        HandleRedraw::Auto
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.scene_view.destroy();

            self.device.destroy_command_pool(self.command_pool, None);
            ManuallyDrop::drop(&mut self.allocator);
            self.surface_loader.destroy_surface(self.surface, None);
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

        let ash_render_state = AshRenderState {
            entry: entry.clone(),
            instance: instance.clone(),
            physical_device,
            device: device.clone(),
            surface_loader: surface_loader.clone(),
            swapchain_loader,
            queue,
            queue_family_index,
            command_pool,
            allocator: allocator.clone(),
        };

        let device = Arc::new(device);
        let surface_loader = Arc::new(surface_loader);
        let scene = Arc::new(Mutex::new(Scene::new()));
        let scene_view = SceneView::new(
            device.clone(),
            allocator.clone(),
            queue,
            queue_family_index,
            command_pool,
            cc.image_registry,
            scene.clone(),
        );
        let app = MyApp {
            _entry: Arc::new(entry),
            instance: Arc::new(instance),
            device: device.clone(),
            debug_utils_loader,
            debug_messenger,
            surface_loader: surface_loader.clone(),
            surface,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            scene_view: scene_view.clone(),
            tree: pane::Pane::create_tree(scene.clone(), scene_view.clone()),
            tree_behavior: TreeBehavior,
        };

        (app, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "egui-ash-tiles",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("egui-ash")
                    .with_inner_size(egui::vec2(800.0, 600.0)),
            ),
            ..Default::default()
        },
    )
}
