use egui::{ahash::HashMapExt, DeferredViewportUiCallback, ViewportIdMap};
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit::Event as AccessKitEvent;
use egui_winit::winit::{self, event_loop::ActiveEventLoop};
use std::time::Instant;
use std::collections::HashMap;

use crate::allocator::Allocator;
use crate::renderer::{EguiCommand, ImageRegistryReceiver, Renderer};
#[cfg(feature = "persistence")]
use crate::storage::Storage;
#[cfg(feature = "persistence")]
use crate::utils;
use crate::AshRenderState;

#[derive(Debug)]
pub(crate) struct IntegrationEvent {
    #[cfg(feature = "accesskit")]
    pub(crate) accesskit: AccessKitEvent,
}

pub(crate) type ViewportUiCallback = std::sync::Arc<dyn Fn(&egui::Context) + Send + Sync>;

#[cfg(feature = "accesskit")]
impl From<AccessKitEvent> for IntegrationEvent {
    fn from(event: AccessKitEvent) -> Self {
        Self {
            #[cfg(feature = "accesskit")]
            accesskit: event,
        }
    }
}

struct Viewport {
    ids: egui::ViewportIdPair,
    class: egui::ViewportClass,
    builder: egui::ViewportBuilder,
    info: egui::ViewportInfo,
    is_first_frame: bool,
    window: winit::window::Window,
    state: egui_winit::State,
    ui_cb: Option<std::sync::Arc<DeferredViewportUiCallback>>,
}
impl Viewport {
    fn update_viewport_info(&mut self, ctx: &egui::Context) {
        egui_winit::update_viewport_info(&mut self.info, ctx, &self.window, false);
    }
}

pub enum PaintResult {
    Exit,
    Wait,
}

pub(crate) struct Integration<A: Allocator + 'static> {
    _app_id: String,
    beginning: Instant,

    pub(crate) renderer: Renderer<A>,

    pub(crate) context: egui::Context,
    window_id_to_viewport_id: HashMap<winit::window::WindowId, egui::ViewportId>,
    viewports: ViewportIdMap<Viewport>,
    focused_viewport: Option<egui::ViewportId>,
    max_texture_side: usize,
    theme: Option<winit::window::Theme>,

    #[cfg(feature = "persistence")]
    pub(crate) storage: Storage,
    #[cfg(feature = "persistence")]
    persistent_windows: bool,
    #[cfg(feature = "persistence")]
    persistent_egui_memory: bool,
    #[cfg(feature = "persistence")]
    last_auto_save: Instant,
}

impl<A: Allocator + 'static> Integration<A> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        app_id: &str,
        event_loop: &ActiveEventLoop,
        context: egui::Context,
        main_window: winit::window::Window,
        render_state: AshRenderState<A>,
        present_mode: ash::vk::PresentModeKHR,
        receiver: ImageRegistryReceiver,
        theme: Option<winit::window::Theme>,
        #[cfg(feature = "accesskit")] event_loop_proxy: &winit::event_loop::EventLoopProxy<IntegrationEvent>,
        #[cfg(feature = "persistence")] storage: Storage,
        #[cfg(feature = "persistence")] persistent_windows: bool,
        #[cfg(feature = "persistence")] persistent_egui_memory: bool,
    ) -> Self {
        let renderer = Renderer::new(
            render_state.entry.clone(),
            render_state.instance.clone(),
            render_state.physical_device,
            render_state.device.clone(),
            render_state.surface_loader.clone(),
            render_state.swapchain_loader.clone(),
            render_state.queue,
            render_state.queue_family_index,
            render_state.command_pool,
            present_mode,
            render_state.allocator,
            receiver,
        );

        let main_window_id = main_window.id();

        #[cfg(feature = "persistence")]
        restore_main_window(
            event_loop,
            &context,
            &main_window,
            &storage,
            persistent_windows,
        );

        // use native window viewports
        context.set_embed_viewports(false);

        let limits = unsafe {
            let properties = render_state
                .instance
                .get_physical_device_properties(render_state.physical_device);
            properties.limits
        };
        let max_texture_side = limits.max_image_dimension2_d as usize;

        let root_state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            &event_loop,
            Some(main_window.scale_factor() as f32),
            theme,
            Some(max_texture_side),
        );

        let mut window_id_to_viewport_id = HashMap::new();
        window_id_to_viewport_id.insert(main_window_id, egui::ViewportId::ROOT);

        let mut viewports = ViewportIdMap::new();
        #[allow(unused_mut)] // for accesskit
        let mut root_viewport = Viewport {
            ids: egui::ViewportIdPair::ROOT,
            class: egui::ViewportClass::Root,
            builder: egui::ViewportBuilder::default(),
            info: egui::ViewportInfo::default(),
            is_first_frame: true,
            window: main_window,
            state: root_state,
            ui_cb: None,
        };

        #[cfg(feature = "accesskit")]
        {
            root_viewport.state.init_accesskit(
                event_loop,
                &root_viewport.window,
                event_loop_proxy.clone(),
            );
        }
        viewports.insert(egui::ViewportId::ROOT, root_viewport);

        Self {
            _app_id: app_id.to_owned(),
            beginning: Instant::now(),

            renderer,

            context,
            window_id_to_viewport_id,
            viewports,
            focused_viewport: None,
            max_texture_side,
            theme,

            #[cfg(feature = "persistence")]
            storage,
            #[cfg(feature = "persistence")]
            persistent_windows,
            #[cfg(feature = "persistence")]
            persistent_egui_memory,
            #[cfg(feature = "persistence")]
            last_auto_save: Instant::now(),
        }
    }

    /// Register the immediate viewport renderer callback.
    /// SAFETY: self must be stored at a stable heap address (e.g. Box<Integration<A>>)
    /// when this is called. The callback is cleared in destroy() before self drops.
    pub(crate) fn register_immediate_viewport_renderer(&mut self, event_loop: &ActiveEventLoop) {
        let ptr = self as *mut Integration<A>;
        // SAFETY: event loop lives at least as long as this callback
        let event_loop_ptr = std::ptr::from_ref(event_loop);

        egui::Context::set_immediate_viewport_renderer(move |ctx, immediate_viewport| {
            // SAFETY: ptr is valid; callback cleared before Integration drops in destroy().
            let integration = unsafe { &mut *ptr };
            let event_loop = unsafe { &*event_loop_ptr };
            integration.immediate_viewport_render(ctx, immediate_viewport, event_loop);
        });
    }

    fn immediate_viewport_render(
        &mut self,
        ctx: &egui::Context,
        mut immediate_viewport: egui::ImmediateViewport<'_>,
        event_loop: &ActiveEventLoop,
    ) {
        let raw_input = {
            let mut window_initialized = false;
            let viewport = initialize_or_update_viewport(
                ctx,
                event_loop,
                &mut self.window_id_to_viewport_id,
                self.max_texture_side,
                &mut self.viewports,
                self.focused_viewport,
                immediate_viewport.ids,
                egui::ViewportClass::Immediate,
                immediate_viewport.builder,
                None,
                &mut window_initialized,
                self.theme,
                #[cfg(feature = "persistence")]
                &self.storage,
                #[cfg(feature = "persistence")]
                self.persistent_windows,
            );
            if window_initialized {
                self.renderer
                    .ensure_viewport_context(viewport.ids.this, &viewport.window);
            }
            egui_winit::apply_viewport_builder_to_window(ctx, &viewport.window, &viewport.builder);

            let mut raw_input = viewport.state.take_egui_input(&viewport.window);
            raw_input.viewports = self
                .viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();

            raw_input
        };

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = ctx.run(raw_input, |ctx| {
            (immediate_viewport.viewport_ui_cb)(ctx);
        });

        let viewport_id = immediate_viewport.ids.this;
        let viewport = self.viewports.get_mut(&viewport_id).unwrap();
        viewport.info.events.clear();
        viewport
            .state
            .handle_platform_output(&viewport.window, platform_output);

        let clipped_primitives = ctx.tessellate(shapes, pixels_per_point);
        let scale = ctx.zoom_factor();
        let size = viewport.window.inner_size();
        let window_ref = &viewport.window;

        self.renderer.present_viewport_auto(
            viewport_id,
            window_ref,
            clipped_primitives,
            textures_delta,
            scale,
            size,
        );

        let viewport = self.viewports.get_mut(&viewport_id).unwrap();
        if viewport.is_first_frame {
            viewport.is_first_frame = false;
        } else {
            viewport.window.set_visible(true);
        }

        // handle viewport output
        for (&vp_id, output) in &viewport_output {
            let ids = egui::ViewportIdPair::from_self_and_parent(vp_id, output.parent);

            let mut window_initialized = false;
            let viewport = initialize_or_update_viewport(
                ctx,
                event_loop,
                &mut self.window_id_to_viewport_id,
                self.max_texture_side,
                &mut self.viewports,
                self.focused_viewport,
                ids,
                output.class,
                output.builder.clone(),
                output.viewport_ui_cb.clone(),
                &mut window_initialized,
                self.theme,
                #[cfg(feature = "persistence")]
                &self.storage,
                #[cfg(feature = "persistence")]
                self.persistent_windows,
            );
            if window_initialized {
                self.renderer
                    .ensure_viewport_context(viewport.ids.this, &viewport.window);
            }

            viewport.info.focused = Some(self.focused_viewport == Some(vp_id));
            let mut _actions = Vec::new();
            egui_winit::process_viewport_commands(
                ctx,
                &mut viewport.info,
                output.commands.clone(),
                &viewport.window,
                &mut _actions,
            );
        }

        // Prune dead viewports
        let active_viewports_ids: egui::ViewportIdSet = viewport_output.keys().copied().collect();
        self.viewports.retain(|id, _| active_viewports_ids.contains(id));
        self.renderer.destroy_viewports(&active_viewports_ids);
        self.window_id_to_viewport_id
            .retain(|_, id| active_viewports_ids.contains(id));
    }

    pub(crate) fn viewport_id_from_window_id(
        &self,
        window_id: winit::window::WindowId,
    ) -> Option<egui::ViewportId> {
        self.window_id_to_viewport_id.get(&window_id).copied()
    }

    pub(crate) fn get_viewport_size(
        &self,
        viewport_id: egui::ViewportId,
    ) -> Option<winit::dpi::PhysicalSize<u32>> {
        let viewport = self.viewports.get(&viewport_id)?;
        Some(viewport.window.inner_size())
    }

    pub(crate) fn handle_window_event(
        &mut self,
        window_id: winit::window::WindowId,
        window_event: &winit::event::WindowEvent,
        event_loop: &ActiveEventLoop,
        follow_system_theme: bool,
        app: &mut impl crate::App,
    ) -> bool {
        let event_response = {
            let Some(&viewport_id) = self.window_id_to_viewport_id.get(&window_id) else {
                return false;
            };

            let Some(viewport) = self.viewports.get_mut(&viewport_id) else {
                return false;
            };

            match window_event {
                winit::event::WindowEvent::ThemeChanged(theme) => {
                    if follow_system_theme {
                        viewport.window.set_theme(Some(*theme));
                    }
                }
                winit::event::WindowEvent::Focused(focused) => {
                    if *focused {
                        self.focused_viewport = Some(viewport_id);
                    } else {
                        self.focused_viewport = None;
                    }
                }
                winit::event::WindowEvent::Resized(_) => {
                    self.renderer.mark_viewport_dirty(viewport_id);
                }
                winit::event::WindowEvent::ScaleFactorChanged { .. } => {
                    self.renderer.mark_viewport_dirty(viewport_id);
                }
                winit::event::WindowEvent::CloseRequested => {
                    if viewport_id == egui::ViewportId::ROOT {
                        event_loop.exit();
                    }
                    viewport.info.events.push(egui::ViewportEvent::Close);
                    self.context.request_repaint_of(viewport.ids.parent);
                }
                _ => {}
            }

            let event_response = viewport
                .state
                .on_window_event(&viewport.window, window_event);

            if event_response.repaint {
                viewport.window.request_redraw();
            }

            event_response
        };

        if window_event == &winit::event::WindowEvent::RedrawRequested {
            self.paint(event_loop, window_id, app);
        }

        event_response.consumed
    }

    #[cfg(feature = "accesskit")]
    pub(crate) fn handle_accesskit_event(
        &mut self,
        event: &AccessKitEvent,
        event_loop: &ActiveEventLoop,
        app: &mut impl crate::App,
    ) {
        let AccessKitEvent {
            window_id,
            window_event,
        } = event;
        {
            use egui_winit::accesskit_winit::WindowEvent;
            let Some(viewport_id) = self.viewport_id_from_window_id(*window_id) else {
                return;
            };
            let viewport = self.viewports.get_mut(&viewport_id).unwrap();
            if let WindowEvent::ActionRequested(request) = window_event {
                viewport.state.on_accesskit_action_request(request.clone());
            }
        }
        self.paint(event_loop, *window_id, app);
    }

    pub(crate) fn run_ui_and_record_paint_cmd(
        &mut self,
        event_loop: &ActiveEventLoop,
        app: &mut impl crate::App,
        window_id: winit::window::WindowId,
        create_swapchain_internal: bool,
    ) -> (Option<EguiCommand>, PaintResult) {
        let (viewport_id, viewport_ui_cb, raw_input, window_inner_size, window_scale_factor) = {
            let Some(viewport_id) = self.window_id_to_viewport_id.get(&window_id).copied() else {
                log::error!("window_id not found");
                return (None, PaintResult::Wait);
            };

            if viewport_id != egui::ViewportId::ROOT {
                let Some(viewport) = self.viewports.get(&viewport_id) else {
                    log::error!("viewport not found");
                    return (None, PaintResult::Wait);
                };

                if viewport.ui_cb.is_none() {
                    // This only happens in an immediate viewport.
                    // need to repaint with parent viewport.
                    if self.viewports.get(&viewport.ids.parent).is_some() {
                        self.context.request_repaint_of(viewport.ids.parent);
                        return (None, PaintResult::Wait);
                    }
                    return (None, PaintResult::Wait);
                }
            }

            let Some(viewport) = self.viewports.get_mut(&viewport_id) else {
                log::error!("viewport not found");
                return (None, PaintResult::Wait);
            };
            viewport.update_viewport_info(&self.context);

            let viewport_ui_cb = viewport.ui_cb.clone();
            let window_inner_size = viewport.window.inner_size();
            let window_scale_factor = viewport.window.scale_factor() as f32;

            if create_swapchain_internal {
                self.renderer
                    .ensure_viewport_context(viewport_id, &viewport.window);
            }
            // For handle path (create_swapchain_internal=false), no swapchain management here.

            let mut raw_input = viewport.state.take_egui_input(&viewport.window);
            raw_input.time = Some(self.beginning.elapsed().as_secs_f64());
            raw_input.viewports = self
                .viewports
                .iter()
                .map(|(id, vp)| (*id, vp.info.clone()))
                .collect();

            (viewport_id, viewport_ui_cb, raw_input, window_inner_size, window_scale_factor)
        };

        let egui::FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = {
            let close_requested = raw_input.viewport().close_requested();

            let full_output = self.context.run(raw_input, |ctx| {
                if let Some(viewport_ui_cb) = viewport_ui_cb.clone() {
                    // child viewport
                    viewport_ui_cb(ctx);
                } else {
                    // ROOT viewport
                    app.ui(ctx);
                }
            });

            let is_root_viewport = viewport_ui_cb.is_none();
            if is_root_viewport && close_requested {
                let canceled = full_output.viewport_output[&egui::ViewportId::ROOT]
                    .commands
                    .contains(&egui::ViewportCommand::CancelClose);
                if !canceled {
                    return (None, PaintResult::Exit);
                }
            }

            full_output
        };

        let egui_cmd_or_auto = {
            let egui_cmd = if let Some(viewport) = self.viewports.get_mut(&viewport_id) {
                viewport.info.events.clear();
                viewport
                    .state
                    .handle_platform_output(&viewport.window, platform_output);

                let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
                let scale = window_scale_factor * self.context.zoom_factor();

                if create_swapchain_internal {
                    // Auto path: present immediately, return None
                    self.renderer.present_viewport_auto(
                        viewport_id,
                        &viewport.window,
                        clipped_primitives,
                        textures_delta,
                        scale,
                        window_inner_size,
                    );
                    None
                } else {
                    // Handle path: create EguiCommand for user
                    let cmd = self.renderer.create_egui_cmd(
                        viewport_id,
                        clipped_primitives,
                        textures_delta,
                        scale,
                        window_inner_size,
                    );
                    Some(cmd)
                }
            } else {
                return (None, PaintResult::Wait);
            };

            for (&vp_id, output) in &viewport_output {
                let ids = egui::ViewportIdPair::from_self_and_parent(vp_id, output.parent);
                let focused_viewport = self.focused_viewport;

                let mut window_initialized = false;
                let viewport = initialize_or_update_viewport(
                    &self.context,
                    event_loop,
                    &mut self.window_id_to_viewport_id,
                    self.max_texture_side,
                    &mut self.viewports,
                    focused_viewport,
                    ids,
                    output.class,
                    output.builder.clone(),
                    output.viewport_ui_cb.clone(),
                    &mut window_initialized,
                    self.theme,
                    #[cfg(feature = "persistence")]
                    &self.storage,
                    #[cfg(feature = "persistence")]
                    self.persistent_windows,
                );
                if window_initialized {
                    app.handle_event(crate::event::Event::DeferredViewportCreated {
                        viewport_id: ids.this,
                        window: &viewport.window,
                    });
                }

                viewport.info.focused = Some(self.focused_viewport == Some(vp_id));
                let mut _actions = Vec::new();
                egui_winit::process_viewport_commands(
                    &self.context,
                    &mut viewport.info,
                    output.commands.clone(),
                    &viewport.window,
                    &mut _actions,
                );
            }

            if let Some(viewport) = self.viewports.get_mut(&viewport_id) {
                if viewport.window.is_minimized() == Some(true) {
                    // On Mac, a minimized Window uses up all CPU
                }
            }

            // Prune dead viewports
            let active_viewports_ids: egui::ViewportIdSet =
                viewport_output.keys().copied().collect();
            self.viewports.retain(|id, _| active_viewports_ids.contains(id));
            self.renderer.destroy_viewports(&active_viewports_ids);
            self.window_id_to_viewport_id
                .retain(|_, id| active_viewports_ids.contains(id));

            egui_cmd
        };

        // autosave
        self.maybe_autosave(app);

        (egui_cmd_or_auto, PaintResult::Wait)
    }

    pub(crate) fn paint(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        app: &mut impl crate::App,
    ) {
        let Some(viewport_id) = self.viewport_id_from_window_id(window_id) else {
            return;
        };

        let handle_redraw = app.request_redraw(viewport_id);
        let paint_result = match handle_redraw {
            crate::HandleRedraw::Auto => {
                let (_, paint_result) =
                    self.run_ui_and_record_paint_cmd(event_loop, app, window_id, true);
                // auto path presents internally
                paint_result
            }
            crate::HandleRedraw::Handle(handler) => {
                let (egui_cmd, paint_result) =
                    self.run_ui_and_record_paint_cmd(event_loop, app, window_id, false);
                if let Some(size) = self.get_viewport_size(viewport_id) {
                    if let Some(egui_cmd) = egui_cmd {
                        handler(size, egui_cmd);
                    }
                }
                paint_result
            }
        };

        if let Some(viewport) = self.viewports.get_mut(&viewport_id) {
            if viewport.is_first_frame {
                viewport.is_first_frame = false;
            } else {
                viewport.window.set_visible(true);
            }
        }

        match paint_result {
            PaintResult::Wait => (),
            PaintResult::Exit => event_loop.exit(),
        }
    }

    pub(crate) fn paint_all(&mut self, event_loop: &ActiveEventLoop, app: &mut impl crate::App) {
        let window_ids = self.window_id_to_viewport_id.keys().copied().collect::<Vec<_>>();
        for window_id in window_ids {
            self.paint(event_loop, window_id, app);
        }
    }

    fn maybe_autosave(&mut self, _app: &mut impl crate::App) {
        #[cfg(feature = "persistence")]
        {
            if self.last_auto_save.elapsed() < _app.auto_save_interval() {
                return;
            }
            self.save(_app);
            self.last_auto_save = Instant::now();
        }
    }

    #[cfg(feature = "persistence")]
    pub(crate) fn save(&mut self, app: &mut impl crate::App) {
        let storage = &mut self.storage;
        if self.persistent_windows {
            let mut windows = HashMap::new();
            for (&id, viewport) in self.viewports.iter() {
                let settings = egui_winit::WindowSettings::from_window(
                    self.context.zoom_factor(),
                    &viewport.window,
                );
                windows.insert(id, settings);
            }
            storage.set_windows(&windows);
        }
        if self.persistent_egui_memory {
            storage.set_egui_memory(&self.context.memory(|m| m.clone()));
        }
        app.save(storage);
        storage.flush();
    }

    pub fn destroy(&mut self) {
        // Clear the immediate viewport renderer callback first
        egui::Context::set_immediate_viewport_renderer(|_, _| {});
        self.renderer.destroy();
    }
}

#[allow(clippy::too_many_arguments)]
fn initialize_or_update_viewport<'vp>(
    context: &egui::Context,
    event_loop: &ActiveEventLoop,
    window_id_to_viewport_id: &mut HashMap<winit::window::WindowId, egui::ViewportId>,
    max_texture_side: usize,
    viewports: &'vp mut ViewportIdMap<Viewport>,
    focused_viewport: Option<egui::ViewportId>,
    ids: egui::ViewportIdPair,
    class: egui::ViewportClass,
    mut builder: egui::ViewportBuilder,
    viewport_ui_cb: Option<ViewportUiCallback>,
    window_initialized: &mut bool,
    theme: Option<winit::window::Theme>,
    #[cfg(feature = "persistence")] storage: &Storage,
    #[cfg(feature = "persistence")] persistent_windows: bool,
) -> &'vp mut Viewport {
    if builder.icon.is_none() {
        // Inherit icon from parent
        builder.icon = viewports
            .get_mut(&ids.parent)
            .and_then(|vp| vp.builder.icon.clone());
    }
    *window_initialized = false;

    match viewports.entry(ids.this) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            *window_initialized = true;
            let window = create_viewport_window(
                event_loop,
                context,
                window_id_to_viewport_id,
                ids.this,
                builder.clone(),
                #[cfg(feature = "persistence")]
                storage,
                #[cfg(feature = "persistence")]
                persistent_windows,
            );
            let state = egui_winit::State::new(
                context.clone(),
                ids.this,
                event_loop,
                Some(window.scale_factor() as f32),
                theme,
                Some(max_texture_side),
            );
            entry.insert(Viewport {
                ids,
                class,
                builder,
                info: egui::ViewportInfo {
                    maximized: Some(window.is_maximized()),
                    minimized: window.is_minimized(),
                    ..Default::default()
                },
                is_first_frame: true,
                window,
                state,
                ui_cb: viewport_ui_cb,
            })
        }

        std::collections::hash_map::Entry::Occupied(mut entry) => {
            // Patch an existing viewport:
            let viewport = entry.get_mut();

            viewport.class = class;
            viewport.ids.parent = ids.parent;
            viewport.ui_cb = viewport_ui_cb;

            let (delta_commands, recreate) = viewport.builder.patch(builder.clone());

            if recreate {
                *window_initialized = true;
                viewport.window = create_viewport_window(
                    event_loop,
                    context,
                    window_id_to_viewport_id,
                    ids.this,
                    builder.clone(),
                    #[cfg(feature = "persistence")]
                    storage,
                    #[cfg(feature = "persistence")]
                    persistent_windows,
                );
                viewport.state = egui_winit::State::new(
                    context.clone(),
                    ids.this,
                    event_loop,
                    Some(viewport.window.scale_factor() as f32),
                    theme,
                    Some(max_texture_side),
                );
                viewport.is_first_frame = true;
            } else {
                viewport.info.focused = Some(focused_viewport == Some(ids.this));
                let mut _actions = Vec::new();
                egui_winit::process_viewport_commands(
                    context,
                    &mut viewport.info,
                    delta_commands,
                    &viewport.window,
                    &mut _actions,
                );
            }

            entry.into_mut()
        }
    }
}

fn create_viewport_window(
    event_loop: &ActiveEventLoop,
    context: &egui::Context,
    window_id_to_viewport_id: &mut HashMap<winit::window::WindowId, egui::ViewportId>,
    viewport_id: egui::ViewportId,
    #[allow(unused_mut)] // for persistence
    mut builder: egui::ViewportBuilder,
    #[cfg(feature = "persistence")] storage: &Storage,
    #[cfg(feature = "persistence")] persistent_windows: bool,
) -> winit::window::Window {
    #[cfg(feature = "persistence")]
    if persistent_windows {
        let egui_zoom_factor = context.zoom_factor();
        let window_settings = storage
            .get_windows()
            .and_then(|windows| windows.get(&viewport_id).map(|s| s.to_owned()))
            .map(|mut settings| {
                settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                    egui_zoom_factor,
                    event_loop,
                ));
                settings.clamp_position_to_monitors(egui_zoom_factor, event_loop);
                settings.to_owned()
            });

        if let Some(window_settings) = window_settings {
            builder =
                window_settings.initialize_viewport_builder(egui_zoom_factor, event_loop, builder);
        }
    }

    builder = builder.with_visible(false);
    let window = egui_winit::create_window(context, event_loop, &builder).unwrap();

    egui_winit::apply_viewport_builder_to_window(context, &window, &builder);

    window_id_to_viewport_id.insert(window.id(), viewport_id);

    window
}

#[cfg(feature = "persistence")]
fn restore_main_window(
    event_loop: &ActiveEventLoop,
    context: &egui::Context,
    main_window: &winit::window::Window,
    storage: &Storage,
    persistent_windows: bool,
) {
    if persistent_windows {
        let window_settings = storage
            .get_windows()
            .and_then(|windows| windows.get(&egui::ViewportId::ROOT).map(|s| s.to_owned()))
            .map(|mut settings| {
                let egui_zoom_factor = context.zoom_factor();
                settings.clamp_size_to_sane_values(utils::largest_monitor_point_size(
                    egui_zoom_factor,
                    event_loop,
                ));
                settings.clamp_position_to_monitors(egui_zoom_factor, event_loop);
                settings.to_owned()
            });

        if let Some(window_settings) = window_settings {
            window_settings.initialize_window(main_window);
        }
    }
}
