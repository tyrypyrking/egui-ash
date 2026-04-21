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

    /// Build a non-root viewport (Immediate or Deferred). Called from
    /// [`crate::host::Host`] when egui reports a new viewport in the
    /// immediate-viewport renderer or in a deferred `viewport_output`
    /// map.
    ///
    /// Differs from `new_root` in three respects:
    ///   1. The winit window is created here from the supplied
    ///      [`egui::ViewportBuilder`] — the ROOT path creates the window
    ///      earlier in `run.rs` so persisted settings can be applied.
    ///   2. A pre-closed `RegistryCommand` receiver is passed to the
    ///      Compositor — per B9 decision #8, user-registered textures
    ///      are ROOT-only in the alpha; child compositors never see any
    ///      Register/Update/Unregister commands.
    ///   3. Starts with `is_first_frame: true` and the caller-supplied
    ///      `ViewportClass` + `ui_cb` (None for Immediate, Some for
    ///      Deferred).
    ///
    /// # Safety
    /// - All Vulkan handles in `vulkan` must be valid.
    /// - Must run on the main thread (owns the new winit Window).
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new_child(
        vulkan: &VulkanContext,
        event_loop: &winit::event_loop::ActiveEventLoop,
        id: egui::ViewportId,
        parent: egui::ViewportId,
        class: egui::ViewportClass,
        builder: egui::ViewportBuilder,
        context: &egui::Context,
        present_mode: vk::PresentModeKHR,
        ui_cb: Option<Arc<egui::DeferredViewportUiCallback>>,
    ) -> Self {
        // Translate the egui ViewportBuilder into winit WindowAttributes.
        // Cover every field winit honours at creation time; egui patches
        // (size/title changes after creation) are applied per-frame in a
        // later step.
        let mut attrs = winit::window::WindowAttributes::default().with_visible(true);
        if let Some(title) = &builder.title {
            attrs = attrs.with_title(title.clone());
        }
        if let Some(pos) = builder.position {
            attrs =
                attrs.with_position(winit::dpi::LogicalPosition::new(pos.x as f64, pos.y as f64));
        }
        if let Some(size) = builder.inner_size {
            attrs =
                attrs.with_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(size) = builder.min_inner_size {
            attrs = attrs
                .with_min_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(size) = builder.max_inner_size {
            attrs = attrs
                .with_max_inner_size(winit::dpi::LogicalSize::new(size.x as f64, size.y as f64));
        }
        if let Some(decorations) = builder.decorations {
            attrs = attrs.with_decorations(decorations);
        }
        if let Some(resizable) = builder.resizable {
            attrs = attrs.with_resizable(resizable);
        }
        if let Some(transparent) = builder.transparent {
            attrs = attrs.with_transparent(transparent);
        }
        if let Some(maximized) = builder.maximized {
            attrs = attrs.with_maximized(maximized);
        }

        let window = event_loop
            .create_window(attrs)
            .expect("failed to create child viewport window");

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

        // Dummy channel — drop the sender so the compositor's per-frame
        // drain sees `Err(Disconnected)` immediately and exits. Per B9
        // decision #8 non-root compositors don't receive user textures
        // in the alpha.
        let (tx, registry_rx) = std::sync::mpsc::channel();
        drop(tx);

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
            id,
            event_loop,
            Some(window.scale_factor() as f32),
            // Non-root viewports default to the platform theme; explicit
            // theme propagation is a post-alpha nicety.
            None,
            None,
        );

        Self {
            id,
            ids: egui::ViewportIdPair { this: id, parent },
            class,
            builder,
            info: egui::ViewportInfo::default(),
            is_first_frame: true,
            window,
            state,
            compositor,
            surface,
            ui_cb,
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
