use ash::{vk, Device, Entry, Instance};
use egui_winit::winit;
use std::ffi::CString;

#[cfg(feature = "persistence")]
use crate::storage;
use crate::{
    event,
    renderer::{EguiCommand, ImageRegistry},
    Allocator, ExitSignal,
};

/// egui theme type.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Theme {
    Light,
    Dark,
}

/// redraw handler type.
pub type RedrawHandler = Box<dyn FnOnce(winit::dpi::PhysicalSize<u32>, EguiCommand) + Send>;

/// return type of [`App::request_redraw`].
pub enum HandleRedraw {
    Auto,
    Handle(RedrawHandler),
}

/// main egui-ash app trait.
pub trait App {
    /// egui entry point
    fn ui(&mut self, ctx: &egui::Context);

    /// handle events of the app.
    fn handle_event(&mut self, _event: event::Event) {}

    /// redraw the app.
    ///
    /// If you want to draw only egui, return [`HandleRedraw::Auto`].
    ///
    /// If you want to do your own Vulkan drawing in ash,
    /// return [`HandleRedraw::Handle(RedrawHandle)`] with FnOnce of drawing.
    /// NOTE: You must call `egui_cmd.update_swapchain` inside render function
    /// when you first render and when you recreate the swapchain.
    fn request_redraw(&mut self, _viewport_id: egui::ViewportId) -> HandleRedraw {
        HandleRedraw::Auto
    }

    /// time interval for call [`Self::save`].
    #[cfg(feature = "persistence")]
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(30)
    }

    /// save app state.
    #[cfg(feature = "persistence")]
    fn save(&mut self, _storage: &mut storage::Storage) {}
}

/// passed to [`AppCreator::create()`] for creating egui-ash app.
pub struct CreationContext<'a> {
    /// root window
    pub main_window: &'a winit::window::Window,

    /// egui context
    pub context: egui::Context,

    /// required instance extensions for ash vulkan
    pub required_instance_extensions: Vec<CString>,

    /// required device extensions for ash vulkan
    pub required_device_extensions: Vec<CString>,

    /// user texture image registry for egui-ash
    pub image_registry: ImageRegistry,

    /// exit signal sender
    pub exit_signal: ExitSignal,
}

/// Vulkan objects required for drawing ash.
/// You should return this struct from [`AppCreator::create()`].
pub struct AshRenderState<A: Allocator + 'static> {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub surface_loader: ash::khr::surface::Instance,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub allocator: A,
}

/// egui-ash app creator trait.
pub trait AppCreator<A: Allocator + 'static> {
    type App: App;

    /// create egui-ash app.
    fn create(&self, cc: CreationContext) -> (Self::App, AshRenderState<A>);
}
