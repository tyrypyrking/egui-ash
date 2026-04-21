//! Per-egui-viewport state.
//!
//! Under egui's multi-viewport API (§B9), each active [`egui::ViewportId`]
//! corresponds to one OS window, one Vulkan surface, one swapchain
//! (inside [`Compositor`]), and one `egui-winit` input adapter. This
//! module bundles those resources so [`crate::host::Host`] can key them
//! by `ViewportId` in a `HashMap`.

use std::sync::Arc;

use ash::vk;
use egui_winit::winit;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};

use crate::compositor::Compositor;
use crate::image_registry::RegistryCommand;
use crate::types::VulkanContext;

/// Per-viewport state: window, egui-winit adapter, compositor, surface,
/// and the egui-facing metadata egui needs to track viewport lifecycle.
///
/// All fields are `pub(crate)` — external access goes through
/// [`crate::host::Host`], which owns the `HashMap<ViewportId, Viewport>`.
///
/// The metadata fields (`id`, `ids`, `class`, `builder`, `info`,
/// `is_first_frame`, `ui_cb`) come online in subsequent B9 steps as
/// multi-viewport routing, immediate/deferred viewport construction, and
/// per-viewport lifecycle tracking are wired in. Allow `dead_code` on the
/// struct for that interim; it'll be removed once the fields are active.
#[allow(dead_code)]
pub(crate) struct Viewport {
    pub(crate) id: egui::ViewportId,
    pub(crate) ids: egui::ViewportIdPair,
    pub(crate) class: egui::ViewportClass,
    /// Builder used to create this viewport's window. Retained so egui's
    /// per-frame delta updates (size/title/decorations) can be applied
    /// without re-creating the window.
    pub(crate) builder: egui::ViewportBuilder,
    /// Live viewport info (focus, inner_size, ...) populated each frame
    /// from winit before passing to egui via `raw_input.viewports`.
    pub(crate) info: egui::ViewportInfo,
    /// True only during the first frame — lets egui emit one-shot
    /// "created" events on viewport birth.
    pub(crate) is_first_frame: bool,

    pub(crate) window: winit::window::Window,
    pub(crate) state: egui_winit::State,
    pub(crate) compositor: Compositor,
    /// Vulkan surface for this viewport's window. Owned here because
    /// surfaces are tied 1:1 to windows; [`Compositor`] borrows it.
    pub(crate) surface: vk::SurfaceKHR,

    /// Deferred viewports hold the user-supplied UI callback egui invokes
    /// each frame to populate this viewport's content. `None` for Root
    /// and Immediate (Immediate callbacks are invoked inside egui's own
    /// `set_immediate_viewport_renderer`).
    pub(crate) ui_cb: Option<Arc<egui::DeferredViewportUiCallback>>,
}

impl Viewport {
    /// Build the ROOT viewport from a pre-created winit window. Called
    /// once from [`crate::host::Host::new`]. The caller owns window
    /// creation so it can apply persisted window settings + macOS outer
    /// position fixups before handing the window over.
    ///
    /// # Safety
    /// - All Vulkan handles in `vulkan` must be valid.
    /// - Must run on the main thread (the thread that created `window`
    ///   and drives the winit event loop).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new_root(
        vulkan: &VulkanContext,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window: winit::window::Window,
        builder: egui::ViewportBuilder,
        context: &egui::Context,
        theme: Option<winit::window::Theme>,
        present_mode: vk::PresentModeKHR,
        registry_rx: std::sync::mpsc::Receiver<RegistryCommand>,
    ) -> Self {
        let surface = ash_window::create_surface(
            &vulkan.entry,
            &vulkan.instance,
            window
                .display_handle()
                .expect("failed to get display handle")
                .as_raw(),
            window
                .window_handle()
                .expect("failed to get window handle")
                .as_raw(),
            None,
        )
        .expect("failed to create surface");

        let initial = window.inner_size();
        let compositor = Compositor::new(
            &vulkan.entry,
            &vulkan.instance,
            &vulkan.device,
            vulkan.physical_device,
            vulkan.host_queue,
            vulkan.host_queue_family_index,
            registry_rx,
            surface,
            present_mode,
            vulkan.queue_mutex.clone(),
            vk::Extent2D {
                width: initial.width,
                height: initial.height,
            },
        );

        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            event_loop,
            Some(window.scale_factor() as f32),
            theme,
            None,
        );

        Self {
            id: egui::ViewportId::ROOT,
            ids: egui::ViewportIdPair::ROOT,
            class: egui::ViewportClass::Root,
            builder,
            info: egui::ViewportInfo::default(),
            is_first_frame: true,
            window,
            state,
            compositor,
            surface,
            ui_cb: None,
        }
    }

    /// Destroy this viewport's Vulkan resources: compositor first (its
    /// internal `device_wait_idle` drains pending submits against the
    /// swapchain), then the surface.
    ///
    /// # Safety
    /// `vulkan` must be the same context used to build this viewport.
    /// Caller is responsible for ensuring no other thread references any
    /// of these resources at the time of the call.
    pub(crate) unsafe fn destroy(&mut self, vulkan: &VulkanContext) {
        self.compositor.destroy();
        let surface_loader = ash::khr::surface::Instance::new(&vulkan.entry, &vulkan.instance);
        surface_loader.destroy_surface(self.surface, None);
    }
}
