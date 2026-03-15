use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use egui_ash::{App, AppCreator, AshRenderState, CreationContext, RunOption};
use gpu_allocator::vulkan::*;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

#[path = "../common/mod.rs"]
mod common;
use common::vkutils::*;

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
}
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui| {
                ui.add(
                    egui::Image::new("https://picsum.photos/seed/1.759706314/1024").corner_radius(10.0),
                );

                ui.image(egui::include_image!("./ferris.svg"));
            });
        });
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
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
        egui_extras::install_image_loaders(&cc.context);

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
        "egui-ash-images",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("egui-ash")
                    .with_inner_size(egui::vec2(600.0, 800.0)),
            ),
            follow_system_theme: true,
            ..Default::default()
        },
    )
}
