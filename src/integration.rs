use egui::{
    ahash::{HashMapExt, HashSet},
    DeferredViewportUiCallback, ViewportIdMap,
};
#[cfg(feature = "accesskit")]
use egui_winit::accesskit_winit::ActionRequestEvent;
use egui_winit::winit::{
    self,
    event_loop::{EventLoop, EventLoopWindowTarget},
};
use std::time::Instant;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::allocator::Allocator;
use crate::presenters::Presenters;
use crate::renderer::{EguiCommand, ImageRegistryReceiver, Renderer};
#[cfg(feature = "persistence")]
use crate::storage::Storage;
#[cfg(feature = "persistence")]
use crate::utils;
use crate::AshRenderState;

#[derive(Debug)]
pub(crate) struct IntegrationEvent {
    #[cfg(feature = "accesskit")]
    pub(crate) accesskit: egui_winit::accesskit_winit::ActionRequestEvent,
}
#[cfg(feature = "accesskit")]
impl From<ActionRequestEvent> for IntegrationEvent {
    fn from(event: ActionRequestEvent) -> Self {
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
    ui_cb: Option<Arc<DeferredViewportUiCallback>>,
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

    presenters: Arc<Mutex<Presenters>>,
    renderer: Arc<Mutex<Renderer<A>>>,

    context: egui::Context,
    window_id_to_viewport_id: Arc<Mutex<HashMap<winit::window::WindowId, egui::ViewportId>>>,
    viewports: Arc<Mutex<ViewportIdMap<Viewport>>>,
    focused_viewport: Arc<Mutex<Option<egui::ViewportId>>>,
    max_texture_side: usize,

    #[cfg(feature = "persistence")]
    storage: Storage,
    #[cfg(feature = "persistence")]
    persistent_windows: bool,
    #[cfg(feature = "persistence")]
    persistent_egui_memory: bool,
    #[cfg(feature = "persistence")]
    last_auto_save: Instant,
}
impl<A: Allocator + 'static> Integration<A> {
    pub(crate) fn new(
        app_id: &str,
        event_loop: &EventLoop<IntegrationEvent>,
        context: egui::Context,
        main_window: winit::window::Window,
        render_state: AshRenderState<A>,
        clear_color: [f32; 4],
        present_mode: ash::vk::PresentModeKHR,
        receiver: ImageRegistryReceiver,
        #[cfg(feature = "persistence")] storage: Storage,
        #[cfg(feature = "persistence")] persistent_windows: bool,
        #[cfg(feature = "persistence")] persistent_egui_memory: bool,
    ) -> Self {
        let presenters = Arc::new(Mutex::new(Presenters::new(
            render_state.entry.clone(),
            render_state.instance.clone(),
            render_state.physical_device,
            render_state.device.clone(),
            render_state.surface_loader.clone(),
            render_state.swapchain_loader.clone(),
            render_state.queue,
            render_state.command_pool,
            clear_color,
            present_mode,
        )));
        let renderer = Renderer::new(
            render_state.device.clone(),
            render_state.queue,
            render_state.queue_family_index,
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
            Some(max_texture_side),
        );

        let mut window_id_to_viewport_id = HashMap::new();
        window_id_to_viewport_id.insert(main_window_id, egui::ViewportId::ROOT);
        let window_id_to_viewport_id = Arc::new(Mutex::new(window_id_to_viewport_id));

        let viewports = Arc::new(Mutex::new(ViewportIdMap::new()));
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
            let ctx = context.clone();
            let event_loop_proxy = event_loop.create_proxy();
            root_viewport.state.init_accesskit(
                &root_viewport.window,
                event_loop_proxy,
                move || {
                    ctx.enable_accesskit();
                    ctx.request_repaint();
                    ctx.accesskit_placeholder_tree_update()
                },
            );
        }

        {
            let mut viewports = viewports.lock().unwrap();
            viewports.insert(egui::ViewportId::ROOT, root_viewport);
        }

        let focused_viewport = Arc::new(Mutex::new(None));

        egui::Context::set_immediate_viewport_renderer(immediate_viewport_renderer(
            &presenters,
            &renderer,
            &viewports,
            &window_id_to_viewport_id,
            &focused_viewport,
            max_texture_side,
            #[cfg(feature = "persistence")]
            &storage,
            #[cfg(feature = "persistence")]
            persistent_windows,
            event_loop,
        ));

        Self {
            _app_id: app_id.to_owned(),
            beginning: Instant::now(),

            presenters,
            renderer,

            context,
            window_id_to_viewport_id,
            viewports,
            focused_viewport,
            max_texture_side,

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

    pub(crate) fn viewport_id_from_window_id(
        &self,
        window_id: winit::window::WindowId,
    ) -> Option<egui::ViewportId> {
        let window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();
        window_id_to_viewport_id.get(&window_id).map(|v| *v)
    }

    pub(crate) fn get_viewport_size(
        &self,
        viewport_id: egui::ViewportId,
    ) -> Option<winit::dpi::PhysicalSize<u32>> {
        let viewports = self.viewports.lock().unwrap();
        let Some(viewport) = viewports.get(&viewport_id) else {
            return None;
        };
        Some(viewport.window.inner_size())
    }

    pub(crate) fn handle_window_event(
        &mut self,
        window_id: winit::window::WindowId,
        window_event: &winit::event::WindowEvent,
        event_loop: &EventLoopWindowTarget<IntegrationEvent>,
        follow_system_theme: bool,
        app: &mut impl crate::App,
    ) -> bool {
        let event_response = {
            let window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();
            let Some(&viewport_id) = window_id_to_viewport_id.get(&window_id) else {
                return false;
            };

            let mut viewports = self.viewports.lock().unwrap();
            let Some(viewport) = viewports.get_mut(&viewport_id) else {
                return false;
            };

            match window_event {
                winit::event::WindowEvent::ThemeChanged(theme) => {
                    if follow_system_theme {
                        viewport.window.set_theme(Some(theme.clone()));
                    }
                }
                winit::event::WindowEvent::Focused(focused) => {
                    if *focused {
                        *self.focused_viewport.lock().unwrap() = Some(viewport_id);
                    } else {
                        *self.focused_viewport.lock().unwrap() = None;
                    }
                }
                winit::event::WindowEvent::Resized(_) => {
                    let mut presenters = self.presenters.lock().unwrap();
                    presenters.dirty_swapchain(viewport_id);
                }
                winit::event::WindowEvent::ScaleFactorChanged { .. } => {
                    let mut presenters = self.presenters.lock().unwrap();
                    presenters.dirty_swapchain(viewport_id);
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
                .on_window_event(&viewport.window, &window_event);

            if event_response.repaint {
                viewport.window.request_redraw();
            }

            event_response
        };

        match window_event {
            winit::event::WindowEvent::RedrawRequested => {
                self.paint(event_loop, window_id, app);
            }
            _ => {}
        }

        event_response.consumed
    }

    #[cfg(feature = "accesskit")]
    pub(crate) fn handle_accesskit_event(
        &mut self,
        event: &ActionRequestEvent,
        event_loop: &EventLoopWindowTarget<IntegrationEvent>,
        control_flow: &mut winit::event_loop::ControlFlow,
        app: &mut impl crate::App,
    ) {
        let ActionRequestEvent { window_id, request } = event;
        {
            let Some(viewport_id) = self.viewport_id_from_window_id(*window_id) else {
                return;
            };
            let mut viewports = self.viewports.lock().unwrap();
            let viewport = viewports.get_mut(&viewport_id).unwrap();
            viewport.state.on_accesskit_action_request(request.clone());
        }
        self.paint(event_loop, control_flow, *window_id, app);
    }

    pub(crate) fn run_ui_and_record_paint_cmd(
        &mut self,
        event_loop: &EventLoopWindowTarget<IntegrationEvent>,
        app: &mut impl crate::App,
        window_id: winit::window::WindowId,
        create_swapchain_internal: bool,
    ) -> (Option<EguiCommand>, PaintResult) {
        let (viewport_id, viewport_ui_cb, raw_input) = {
            let window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();
            let Some(viewport_id) = window_id_to_viewport_id.get(&window_id).copied() else {
                log::error!("window_id not found");
                return (None, PaintResult::Wait);
            };

            if viewport_id != egui::ViewportId::ROOT {
                let viewports = self.viewports.lock().unwrap();
                let Some(viewport) = viewports.get(&viewport_id) else {
                    log::error!("viewport not found");
                    return (None, PaintResult::Wait);
                };

                if viewport.ui_cb.is_none() {
                    // This only happens in an immediate viewport.
                    // need to repaint with parent viewport.
                    if viewports.get(&viewport.ids.parent).is_some() {
                        self.context.request_repaint_of(viewport.ids.parent);
                        return (None, PaintResult::Wait);
                    }
                    return (None, PaintResult::Wait);
                }
            }

            let mut viewports = self.viewports.lock().unwrap();
            let Some(viewport) = viewports.get_mut(&viewport_id) else {
                log::error!("viewport not found");
                return (None, PaintResult::Wait);
            };
            viewport.update_viewport_info(&self.context);

            let viewport_ui_cb = viewport.ui_cb.clone();

            let mut presenters = self.presenters.lock().unwrap();
            if create_swapchain_internal {
                presenters.recreate_swapchain_if_needed(viewport_id, &viewport.window);
            } else {
                presenters.destroy_swapchain_if_needed(viewport_id);
            }

            let mut raw_input = viewport.state.take_egui_input(&viewport.window);

            raw_input.time = Some(self.beginning.elapsed().as_secs_f64());
            raw_input.viewports = viewports
                .iter()
                .map(|(id, viewport)| (*id, viewport.info.clone()))
                .collect();

            (viewport_id, viewport_ui_cb, raw_input)
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

        let egui_cmd = {
            let mut viewports = self.viewports.lock().unwrap();
            let egui_cmd = if let Some(viewport) = viewports.get_mut(&viewport_id) {
                viewport.info.events.clear();

                viewport
                    .state
                    .handle_platform_output(&viewport.window, platform_output);

                let mut renderer = self.renderer.lock().unwrap();

                let clipped_primitives = self.context.tessellate(shapes, pixels_per_point);
                let egui_cmd = renderer.create_egui_cmd(
                    viewport.ids.this,
                    clipped_primitives,
                    textures_delta,
                    self.context.zoom_factor(),
                    viewport.window.inner_size(),
                );

                egui_cmd
            } else {
                return (None, PaintResult::Wait);
            };

            for (&viewport_id, output) in &viewport_output {
                let mut window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();

                let ids = egui::ViewportIdPair::from_self_and_parent(viewport_id, output.parent);

                let focused_viewport = self.focused_viewport.lock().unwrap();
                let mut window_initialized = false;
                let viewport = initialize_or_update_viewport(
                    &self.context,
                    event_loop,
                    &mut window_id_to_viewport_id,
                    self.max_texture_side,
                    &mut *viewports,
                    *focused_viewport,
                    ids,
                    output.class,
                    output.builder.clone(),
                    output.viewport_ui_cb.clone(),
                    &mut window_initialized,
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

                viewport.info.focused = Some(*focused_viewport == Some(viewport_id));
                let mut _actions = HashSet::default();
                egui_winit::process_viewport_commands(
                    &self.context,
                    &mut viewport.info,
                    output.commands.clone(),
                    &viewport.window,
                    &mut _actions,
                )
            }

            if let Some(viewport) = viewports.get_mut(&viewport_id) {
                if viewport.window.is_minimized() == Some(true) {
                    // On Mac, a minimized Window uses up all CPU:
                    // https://github.com/emilk/egui/issues/325
                    // std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }

            // Prune dead viewports
            let active_viewports_ids: egui::ViewportIdSet =
                viewport_output.keys().copied().collect();
            viewports.retain(|id, _| active_viewports_ids.contains(id));
            {
                let mut renderer = self.renderer.lock().unwrap();
                let mut presenters = self.presenters.lock().unwrap();
                let mut window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();
                presenters.destroy_viewports(&active_viewports_ids);
                renderer.destroy_viewports(&active_viewports_ids);
                window_id_to_viewport_id.retain(|_, id| active_viewports_ids.contains(id));
            }

            egui_cmd
        };

        // autosave
        self.maybe_autosave(app);

        (Some(egui_cmd), PaintResult::Wait)
    }

    pub(crate) fn present_egui(&mut self, viewport_id: egui::ViewportId, egui_cmd: EguiCommand) {
        let mut presenters = self.presenters.lock().unwrap();
        presenters.present_egui(viewport_id, egui_cmd);
    }

    pub(crate) fn paint(
        &mut self,
        event_loop: &EventLoopWindowTarget<IntegrationEvent>,
        window_id: winit::window::WindowId,
        app: &mut impl crate::App,
    ) {
        let Some(viewport_id) = self.viewport_id_from_window_id(window_id) else {
            return;
        };

        let handle_redraw = app.request_redraw(viewport_id);
        let paint_result = match handle_redraw {
            crate::HandleRedraw::Auto => {
                let (egui_cmd, paint_result) =
                    self.run_ui_and_record_paint_cmd(event_loop, app, window_id, true);
                if let Some(egui_cmd) = egui_cmd {
                    self.present_egui(viewport_id, egui_cmd);
                };
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

        let mut viewports = self.viewports.lock().unwrap();
        if let Some(viewport) = viewports.get_mut(&viewport_id) {
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

    pub(crate) fn paint_all(
        &mut self,
        event_loop: &EventLoopWindowTarget<IntegrationEvent>,
        app: &mut impl crate::App,
    ) {
        let window_ids = {
            let window_id_to_viewport_id = self.window_id_to_viewport_id.lock().unwrap();
            window_id_to_viewport_id.keys().copied().collect::<Vec<_>>()
        };
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
            let viewports = self.viewports.lock().unwrap();
            let mut windows = HashMap::new();
            for (&id, viewport) in viewports.iter() {
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
        let mut presenters = self.presenters.lock().unwrap();
        let mut renderer = self.renderer.lock().unwrap();
        presenters.destroy_root();
        renderer.destroy_root();
        egui::Context::set_immediate_viewport_renderer(|_, _| {});
    }
}

fn initialize_or_update_viewport<'vp>(
    context: &egui::Context,
    event_loop: &EventLoopWindowTarget<IntegrationEvent>,
    window_id_to_viewport_id: &mut HashMap<winit::window::WindowId, egui::ViewportId>,
    max_texture_side: usize,
    viewports: &'vp mut ViewportIdMap<Viewport>,
    focused_viewport: Option<egui::ViewportId>,
    ids: egui::ViewportIdPair,
    class: egui::ViewportClass,
    mut builder: egui::ViewportBuilder,
    viewport_ui_cb: Option<Arc<dyn Fn(&egui::Context) + Send + Sync>>,
    window_initialized: &mut bool,
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
                &context,
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
                    &context,
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
                    Some(max_texture_side),
                );
                viewport.is_first_frame = true;
            } else {
                viewport.info.focused = Some(focused_viewport == Some(ids.this));
                let mut _actions = HashSet::default();
                egui_winit::process_viewport_commands(
                    &context,
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
    event_loop: &EventLoopWindowTarget<IntegrationEvent>,
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
                window_settings.initialize_viewport_builder(egui_zoom_factor, &event_loop, builder);
        }
    }

    let window = egui_winit::create_winit_window_builder(context, event_loop, builder.clone())
        .with_visible(false)
        .build(event_loop)
        .expect("Failed to create window");

    egui_winit::apply_viewport_builder_to_window(context, &window, &builder);

    window_id_to_viewport_id.insert(window.id(), viewport_id);

    window
}

#[cfg(feature = "persistence")]
fn restore_main_window(
    event_loop: &EventLoopWindowTarget<IntegrationEvent>,
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

fn immediate_viewport_renderer(
    presenters: &Arc<Mutex<Presenters>>,
    renderer: &Arc<Mutex<Renderer<impl Allocator>>>,
    viewports: &Arc<Mutex<ViewportIdMap<Viewport>>>,
    window_id_to_viewport_id: &Arc<Mutex<HashMap<winit::window::WindowId, egui::ViewportId>>>,
    focused_viewport: &Arc<Mutex<Option<egui::ViewportId>>>,
    max_texture_side: usize,
    #[cfg(feature = "persistence")] storage: &Storage,
    #[cfg(feature = "persistence")] persistent_windows: bool,
    event_loop: &EventLoop<IntegrationEvent>,
) -> impl for<'b, 'a> Fn(&'b egui::Context, egui::ImmediateViewport<'a>) {
    let presenters = presenters.clone();
    let renderer = renderer.clone();
    let viewports = viewports.clone();
    let window_id_to_viewport_id = window_id_to_viewport_id.clone();
    let focused_viewport = focused_viewport.clone();
    #[cfg(feature = "persistence")]
    let storage = storage.clone();
    #[cfg(feature = "persistence")]
    let persistent_windows = persistent_windows;

    let event_loop: *const EventLoop<IntegrationEvent> = event_loop;

    move |ctx, immediate_viewport| {
        // SAFETY: the event loop lives longer than this callback
        #[allow(unsafe_code)]
        let event_loop = unsafe { event_loop.as_ref().unwrap() };

        let mut renderer = renderer.lock().unwrap();
        let mut presenters = presenters.lock().unwrap();
        let mut viewports = viewports.lock().unwrap();
        let mut window_id_to_viewport_id = window_id_to_viewport_id.lock().unwrap();
        let focused_viewport = focused_viewport.lock().unwrap();

        let raw_input = {
            let mut window_initialized = false;
            let viewport = initialize_or_update_viewport(
                ctx,
                event_loop,
                &mut window_id_to_viewport_id,
                max_texture_side,
                &mut viewports,
                *focused_viewport,
                immediate_viewport.ids,
                egui::ViewportClass::Immediate,
                immediate_viewport.builder,
                None,
                &mut window_initialized,
                #[cfg(feature = "persistence")]
                &storage,
                #[cfg(feature = "persistence")]
                persistent_windows,
            );
            if window_initialized {
                presenters.recreate_swapchain_if_needed(viewport.ids.this, &viewport.window)
            }
            egui_winit::apply_viewport_builder_to_window(ctx, &viewport.window, &viewport.builder);

            let mut raw_input = viewport.state.take_egui_input(&viewport.window);
            raw_input.viewports = viewports
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
        } = {
            let full_output = ctx.run(raw_input, |ctx| {
                (immediate_viewport.viewport_ui_cb)(ctx);
            });

            full_output
        };

        let viewport = viewports.get_mut(&immediate_viewport.ids.this).unwrap();
        viewport.info.events.clear();

        viewport
            .state
            .handle_platform_output(&viewport.window, platform_output);

        let clipped_primitives = ctx.tessellate(shapes, pixels_per_point);
        let egui_cmd = renderer.create_egui_cmd(
            viewport.ids.this,
            clipped_primitives,
            textures_delta,
            ctx.zoom_factor(),
            viewport.window.inner_size(),
        );

        presenters.present_egui(viewport.ids.this, egui_cmd);
        if viewport.is_first_frame {
            viewport.is_first_frame = false;
        } else {
            viewport.window.set_visible(true);
        }

        // handle viewport output
        for (&viewport_id, output) in &viewport_output {
            let ids = egui::ViewportIdPair::from_self_and_parent(viewport_id, output.parent);

            let mut window_initialized = false;
            let viewport = initialize_or_update_viewport(
                &ctx,
                event_loop,
                &mut window_id_to_viewport_id,
                max_texture_side,
                &mut *viewports,
                *focused_viewport,
                ids,
                output.class,
                output.builder.clone(),
                output.viewport_ui_cb.clone(),
                &mut window_initialized,
                #[cfg(feature = "persistence")]
                &storage,
                #[cfg(feature = "persistence")]
                persistent_windows,
            );
            if window_initialized {
                presenters.recreate_swapchain_if_needed(viewport.ids.this, &viewport.window)
            }

            viewport.info.focused = Some(*focused_viewport == Some(viewport_id));
            let mut _actions = HashSet::default();
            egui_winit::process_viewport_commands(
                &ctx,
                &mut viewport.info,
                output.commands.clone(),
                &viewport.window,
                &mut _actions,
            )
        }

        // Prune dead viewports
        let active_viewports_ids: egui::ViewportIdSet = viewport_output.keys().copied().collect();
        viewports.retain(|id, _| active_viewports_ids.contains(id));
        presenters.destroy_viewports(&active_viewports_ids);
        renderer.destroy_viewports(&active_viewports_ids);
        window_id_to_viewport_id.retain(|_, id| active_viewports_ids.contains(id));
    }
}
