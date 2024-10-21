use ash::{ext::debug_utils, vk, Device, Entry, Instance};
use egui_ash::{
    event, App, AppCreator, AshRenderState, CreationContext, HandleRedraw, RunOption, Theme,
};
use gpu_allocator::vulkan::*;
use std::{
    mem::ManuallyDrop,
    sync::{Arc, Mutex},
};

mod model_renderer;
mod triangle_renderer;
mod vkutils;
use model_renderer::ModelRenderer;
use triangle_renderer::TriangleRenderer;
use vkutils::*;

struct MyApp {
    entry: Arc<Entry>,
    instance: Arc<Instance>,
    device: Arc<Device>,
    debug_utils_loader: debug_utils::Instance,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface_loader: Arc<ash::khr::surface::Instance>,
    swapchain_loader: Arc<ash::khr::swapchain::Device>,

    triangle_surface: vk::SurfaceKHR,
    model_surface: Option<vk::SurfaceKHR>,

    queue_family_index: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,

    triangle_renderer: TriangleRenderer,
    model_renderer: Option<ModelRenderer>,

    theme: Theme,
    text: String,
    show_immediate_viewport: bool,
    show_deferred_viewport: Arc<Mutex<bool>>,
    rotate_y: Arc<Mutex<f32>>,
}
impl App for MyApp {
    fn ui(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("my_side_panel").show(&ctx, |ui| {
            ui.heading("Multi viewports");
            ui.label("Hello egui multi viewports!");
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
            ui.checkbox(&mut self.show_immediate_viewport, "show immediate viewport");
            let mut show_deferred_viewport = self.show_deferred_viewport.lock().unwrap();
            ui.checkbox(&mut show_deferred_viewport, "show deferred viewport");
        });
        egui::Window::new("My Window")
            .id(egui::Id::new("my_window"))
            .resizable(true)
            .scroll([true, true])
            .show(&ctx, |ui| {
                ui.heading("Multi viewports");
                ui.label("Hello egui multi viewports!");
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
            });

        if self.show_immediate_viewport {
            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("immediate-viewport"),
                egui::ViewportBuilder::default()
                    .with_title("immediate-viewport")
                    .with_inner_size(egui::vec2(400.0, 300.0)),
                |ctx, _| {
                    // check close requested
                    if ctx.input(|i| i.viewport().close_requested()) {
                        self.show_immediate_viewport = false;
                    }

                    egui::CentralPanel::default().show(&ctx, |ui| {
                        ui.heading("Immediate Viewport");
                        ui.label("immediate viewport!");
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
                        ui.checkbox(&mut self.show_immediate_viewport, "show immediate viewport");
                        let mut show_deferred_viewport =
                            self.show_deferred_viewport.lock().unwrap();
                        ui.checkbox(&mut show_deferred_viewport, "show deferred viewport");
                    });
                },
            );
        }

        if *self.show_deferred_viewport.lock().unwrap() {
            ctx.show_viewport_deferred(
                egui::ViewportId::from_hash_of("deferred-viewport"),
                egui::ViewportBuilder::default().with_title("deferred-viewport"),
                {
                    let rotate_y = self.rotate_y.clone();
                    let show_deferred_viewport = self.show_deferred_viewport.clone();
                    move |ctx, _| {
                        // check close requested
                        if ctx.input(|i| i.viewport().close_requested()) {
                            let mut show_deferred_viewport = show_deferred_viewport.lock().unwrap();
                            *show_deferred_viewport = false;
                        }

                        let mut rotate_y = rotate_y.lock().unwrap();
                        let rotate_y = &mut *rotate_y;
                        egui::SidePanel::left("my_deferred_side_panel").show(&ctx, |ui| {
                            ui.heading("Deferred Viewport");
                            ui.label("deferred viewport!");
                            ui.separator();
                            ui.label("Rotate");
                            ui.add(egui::widgets::Slider::new(rotate_y, -180.0..=180.0));
                        });
                        egui::Window::new("My Window")
                            .id(egui::Id::new("deferred-viewport-window"))
                            .resizable(true)
                            .scroll([true, true])
                            .show(&ctx, |ui| {
                                ui.heading("Deferred Viewport");
                                ui.label("deferred viewport!");
                                ui.separator();
                                ui.label("Rotate");
                                ui.add(egui::widgets::Slider::new(rotate_y, -180.0..=180.0));
                            });
                    }
                },
            );
        }

        match self.theme {
            Theme::Dark => ctx.set_visuals(egui::style::Visuals::dark()),
            Theme::Light => ctx.set_visuals(egui::style::Visuals::light()),
        }
    }

    fn handle_event(&mut self, event: event::Event) {
        match event {
            event::Event::DeferredViewportCreated {
                viewport_id,
                window,
            } if viewport_id == egui::ViewportId(egui::Id::new("deferred-viewport")) => {
                if let Some(model_renderer) = &mut self.model_renderer {
                    model_renderer.destroy();
                }
                if let Some(model_surface) = self.model_surface {
                    unsafe {
                        self.surface_loader.destroy_surface(model_surface, None);
                    }
                }
                let surface = create_surface(&self.entry, &self.instance, &window);
                self.model_surface = Some(surface);
                self.model_renderer = Some(ModelRenderer::new(
                    self.physical_device,
                    self.device.clone(),
                    self.surface_loader.clone(),
                    self.swapchain_loader.clone(),
                    self.allocator.clone(),
                    surface,
                    self.queue_family_index,
                    self.queue,
                    self.command_pool,
                    window.inner_size().width,
                    window.inner_size().height,
                ));
            }
            _ => (),
        }
    }

    fn request_redraw(&mut self, viewport_id: egui::ViewportId) -> HandleRedraw {
        match viewport_id {
            egui::ViewportId::ROOT => HandleRedraw::Handle(Box::new({
                let renderer = self.triangle_renderer.clone();
                move |size, egui_cmd| renderer.render(size.width, size.height, egui_cmd)
            })),
            id if id == egui::ViewportId::from_hash_of("deferred-viewport") => {
                if let Some(model_renderer) = &mut self.model_renderer {
                    HandleRedraw::Handle(Box::new({
                        let renderer = model_renderer.clone();
                        let rotate_y = *self.rotate_y.lock().unwrap();
                        move |size, egui_cmd| {
                            renderer.render(size.width, size.height, egui_cmd, rotate_y)
                        }
                    }))
                } else {
                    HandleRedraw::Auto
                }
            }
            _ => HandleRedraw::Auto,
        }
    }
}
impl Drop for MyApp {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.triangle_renderer.destroy();
            if let Some(model_renderer) = &mut self.model_renderer {
                model_renderer.destroy();
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.surface_loader
                .destroy_surface(self.triangle_surface, None);
            if let Some(model_surface) = self.model_surface {
                self.surface_loader.destroy_surface(model_surface, None);
            }
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

        let ash_render_state = AshRenderState {
            entry: entry.clone(),
            instance: instance.clone(),
            physical_device,
            device: device.clone(),
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),
            queue,
            queue_family_index,
            command_pool,
            allocator: allocator.clone(),
        };

        let device = Arc::new(device);
        let surface_loader = Arc::new(surface_loader);
        let swapchain_loader = Arc::new(swapchain_loader);
        let app = MyApp {
            entry: Arc::new(entry),
            instance: Arc::new(instance),
            device: device.clone(),
            debug_utils_loader,
            debug_messenger,
            physical_device,
            surface_loader: surface_loader.clone(),
            swapchain_loader: swapchain_loader.clone(),

            queue_family_index,
            queue,
            command_pool,
            allocator: ManuallyDrop::new(allocator.clone()),

            triangle_surface: surface,
            model_surface: None,

            triangle_renderer: TriangleRenderer::new(
                physical_device,
                device.clone(),
                surface_loader.clone(),
                swapchain_loader.clone(),
                ManuallyDrop::new(allocator.clone()),
                surface,
                queue_family_index,
                queue,
                command_pool,
                800,
                600,
            ),
            model_renderer: None,

            theme: if cc.context.style().visuals.dark_mode {
                Theme::Dark
            } else {
                Theme::Light
            },
            text: String::from("Hello text!"),
            rotate_y: Arc::new(Mutex::new(0.0)),
            show_immediate_viewport: false,
            show_deferred_viewport: Arc::new(Mutex::new(false)),
        };

        (app, ash_render_state)
    }
}

fn main() -> std::process::ExitCode {
    egui_ash::run(
        "egui-ash-multi-viewports",
        MyAppCreator,
        RunOption {
            viewport_builder: Some(
                egui::ViewportBuilder::default()
                    .with_title("egui-ash")
                    .with_inner_size(egui::vec2(800.0, 600.0)),
            ),
            follow_system_theme: false,
            default_theme: Theme::Dark,
            ..Default::default()
        },
    )
}
