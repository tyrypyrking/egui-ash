use anyhow::Result;
use ash::{vk, Device, Entry, Instance};
use egui_winit::winit;
use raw_window_handle::{HasDisplayHandle as _, HasWindowHandle as _};
use std::collections::HashMap;

use crate::{
    renderer::{EguiCommand, SwapchainUpdateInfo},
    utils,
};

struct Presenter {
    width: u32,
    height: u32,

    device: Device,
    surface: vk::SurfaceKHR,
    clear_color: [f32; 4],
    present_mode: vk::PresentModeKHR,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,

    render_command_buffers: Vec<vk::CommandBuffer>,

    in_flight_fences: Vec<vk::Fence>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    current_frame: usize,

    dirty_flag: bool,
}
impl Presenter {
    fn create_swapchain(
        width: u32,
        height: u32,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        present_mode: vk::PresentModeKHR,
    ) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        let surface_present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?
        };

        // select swapchain format
        let surface_format = surface_formats
            .iter()
            .find(|surface_format| {
                surface_format.format == vk::Format::B8G8R8A8_UNORM
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&surface_formats[0])
            .clone();

        // select surface present mode
        let surface_present_mode = surface_present_modes
            .iter()
            .find(|&&mode| mode == present_mode)
            .unwrap_or(&vk::PresentModeKHR::FIFO);

        // calculate extent
        let surface_extent = if surface_capabilities.current_extent.width != u32::MAX {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            }
        };

        // get image count
        let image_count = surface_capabilities.min_image_count + 1;
        let image_count = if surface_capabilities.max_image_count != 0 {
            image_count.min(surface_capabilities.max_image_count)
        } else {
            image_count
        };

        // create swapchain
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(*surface_present_mode)
            .image_array_layers(1)
            .clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        // get swapchain images
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        Ok((
            swapchain,
            swapchain_images,
            surface_format.format,
            surface_extent,
        ))
    }

    fn create_render_command_buffers(
        device: &Device,
        command_pool: vk::CommandPool,
        len: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(len);
        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?;
        Ok(command_buffers)
    }

    fn create_sync_objects(
        device: &Device,
        len: u32,
    ) -> Result<(Vec<vk::Fence>, Vec<vk::Semaphore>, Vec<vk::Semaphore>)> {
        // create fences
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let mut in_flight_fences = vec![];
        for _ in 0..len {
            let fence = unsafe { device.create_fence(&fence_create_info, None)? };
            in_flight_fences.push(fence);
        }

        // create semaphores
        let mut image_available_semaphores = vec![];
        for _ in 0..len {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };
            image_available_semaphores.push(semaphore);
        }
        let mut render_finished_semaphores = vec![];
        for _ in 0..len {
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();
            let semaphore = unsafe { device.create_semaphore(&semaphore_create_info, None)? };
            render_finished_semaphores.push(semaphore);
        }

        Ok((
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
        ))
    }

    fn create(
        entry: &Entry,
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        command_pool: vk::CommandPool,
        window: &winit::window::Window,
        clear_color: [f32; 4],
        present_mode: vk::PresentModeKHR,
    ) -> Option<Self> {
        let width = window.inner_size().width;
        let height = window.inner_size().height;

        // if window is minimized, return empty presenter
        if width == 0 || height == 0 {
            return None;
        }

        // create surface
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window
                    .display_handle()
                    .expect("Unable to retrieve a display handle")
                    .as_raw(),
                window
                    .window_handle()
                    .expect("Unable to retrieve a window handle")
                    .as_raw(),
                None,
            )
            .expect("Failed to create surface")
        };

        // create swapchain
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain(
                width,
                height,
                physical_device,
                surface,
                surface_loader,
                swapchain_loader,
                present_mode,
            )
            .expect("Failed to create swapchain");

        // create render command buffers
        let render_command_buffers = Self::create_render_command_buffers(
            &device,
            command_pool,
            swapchain_images.len() as u32,
        )
        .expect("Failed to create render command buffers");

        // create sync objects
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(&device, swapchain_images.len() as u32)
                .expect("Failed to create sync objects");

        Some(Self {
            width,
            height,

            device,
            surface,
            clear_color,
            present_mode,

            swapchain,
            swapchain_images,
            swapchain_format,
            swapchain_extent,

            render_command_buffers,

            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
            current_frame: 0,

            dirty_flag: true,
        })
    }

    fn recreate(
        &mut self,
        physical_device: vk::PhysicalDevice,
        device: &Device,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        command_pool: vk::CommandPool,
        window: &winit::window::Window,
    ) {
        let width = window.inner_size().width;
        let height = window.inner_size().height;

        // if window is minimized, do nothing
        if width == 0 || height == 0 {
            return;
        }

        // wait device idle
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        // cleanup old swapchain and sync objects
        unsafe {
            for &fence in self.in_flight_fences.iter() {
                device.destroy_fence(fence, None);
            }
            for &semaphore in self.image_available_semaphores.iter() {
                device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                device.destroy_semaphore(semaphore, None);
            }
            for cmd in self.render_command_buffers.iter() {
                device.free_command_buffers(command_pool, std::slice::from_ref(cmd));
            }
            swapchain_loader.destroy_swapchain(self.swapchain, None);
        }

        // create swapchain
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) =
            Self::create_swapchain(
                width,
                height,
                physical_device,
                self.surface,
                surface_loader,
                swapchain_loader,
                self.present_mode,
            )
            .expect("Failed to create swapchain");

        // create render command buffers
        let render_command_buffers = Self::create_render_command_buffers(
            device,
            command_pool,
            swapchain_images.len() as u32,
        )
        .expect("Failed to create render command buffers");

        // create sync objects
        let (in_flight_fences, image_available_semaphores, render_finished_semaphores) =
            Self::create_sync_objects(device, swapchain_images.len() as u32)
                .expect("Failed to create sync objects");

        // update self
        self.width = width;
        self.height = height;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        self.render_command_buffers = render_command_buffers;
        self.in_flight_fences = in_flight_fences;
        self.image_available_semaphores = image_available_semaphores;
        self.render_finished_semaphores = render_finished_semaphores;
        self.current_frame = 0;
    }

    fn present(
        &mut self,
        mut egui_cmd: EguiCommand,
        device: &Device,
        swapchain_loader: &ash::khr::swapchain::Device,
        queue: vk::Queue,
    ) -> anyhow::Result<()> {
        // Wait for the resources at this index to be completed on the GPU before requesting an available image.
        // Otherwise, the `image_available_semaphores` below may not be ready for reuse.
        unsafe {
            device.wait_for_fences(
                std::slice::from_ref(&self.in_flight_fences[self.current_frame]),
                true,
                u64::MAX,
            )
        }?;

        // acquire next image
        let result = unsafe {
            swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        };
        let index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.dirty_flag = true;
                return Ok(());
            }
            Err(error) => return Err(anyhow::anyhow!(error)),
        };

        // clear command buffer
        unsafe {
            device.reset_command_buffer(
                self.render_command_buffers[self.current_frame],
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
        }

        // begin command buffer
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(
                self.render_command_buffers[self.current_frame],
                &command_buffer_begin_info,
            )?
        }

        // update swapchain
        if self.dirty_flag || egui_cmd.swapchain_recreate_required() {
            let swapchain_update_info = SwapchainUpdateInfo {
                width: self.width,
                height: self.height,
                swapchain_images: self.swapchain_images.clone(),
                surface_format: self.swapchain_format,
            };
            egui_cmd.update_swapchain(swapchain_update_info);
            self.dirty_flag = false;
        }

        // Ensure that the image layout is correct for egui (i.e., `COLOR_ATTACHMENT_OPTIMAL`).
        // This includes converting `UNDEFINED` or `PRESENT_SRC_KHR` layouts to `COLOR_ATTACHMENT_OPTIMAL`, as well as a no-op when it is already in the correct layout.
        utils::insert_image_memory_barrier(
            device,
            self.render_command_buffers[self.current_frame],
            self.swapchain_images[index],
            vk::QUEUE_FAMILY_IGNORED,
            vk::QUEUE_FAMILY_IGNORED,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_READ,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1),
        );

        // record egui cmd
        egui_cmd.record(self.render_command_buffers[self.current_frame], index);

        // end command buffer
        unsafe { device.end_command_buffer(self.render_command_buffers[self.current_frame]) }?;

        // reset fence
        unsafe {
            device.reset_fences(std::slice::from_ref(
                &self.in_flight_fences[self.current_frame],
            ))
        }?;

        // submit command buffer
        let buffers_to_submit = [self.render_command_buffers[self.current_frame]];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&buffers_to_submit)
            .wait_semaphores(std::slice::from_ref(
                &self.image_available_semaphores[self.current_frame],
            ))
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .signal_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ));
        unsafe {
            device.queue_submit(
                queue,
                std::slice::from_ref(&submit_info),
                self.in_flight_fences[self.current_frame],
            )?;
        };

        // present swapchain image
        let image_indices = [index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(
                &self.render_finished_semaphores[self.current_frame],
            ))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(&image_indices);
        let result = unsafe { swapchain_loader.queue_present(queue, &present_info) };
        let is_dirty_swapchain = match result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => panic!("Failed to present queue. Cause: {}", error),
            _ => false,
        };
        self.dirty_flag = is_dirty_swapchain;

        // update current_frame
        self.current_frame = (self.current_frame + 1) % self.in_flight_fences.len();

        Ok(())
    }

    fn destroy(
        &self,
        device: &Device,
        surface_loader: &ash::khr::surface::Instance,
        swapchain_loader: &ash::khr::swapchain::Device,
        command_pool: vk::CommandPool,
    ) {
        // wait device idle
        unsafe {
            device
                .device_wait_idle()
                .expect("Failed to wait device idle")
        };

        // cleanup old swapchain and sync objects
        unsafe {
            for &fence in self.in_flight_fences.iter() {
                device.destroy_fence(fence, None);
            }
            for &semaphore in self.image_available_semaphores.iter() {
                device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in self.render_finished_semaphores.iter() {
                device.destroy_semaphore(semaphore, None);
            }
            for cmd in self.render_command_buffers.iter() {
                device.free_command_buffers(command_pool, std::slice::from_ref(cmd));
            }
            swapchain_loader.destroy_swapchain(self.swapchain, None);
            surface_loader.destroy_surface(self.surface, None);
        }
    }
}

pub struct Presenters {
    entry: Entry,
    instance: Instance,
    physical_device: vk::PhysicalDevice,
    device: Device,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    presenters: HashMap<egui::ViewportId, Presenter>,
    clear_color: [f32; 4],
    present_mode: vk::PresentModeKHR,
}
impl Presenters {
    pub(crate) fn new(
        entry: Entry,
        instance: Instance,
        physical_device: vk::PhysicalDevice,
        device: Device,
        surface_loader: ash::khr::surface::Instance,
        swapchain_loader: ash::khr::swapchain::Device,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        clear_color: [f32; 4],
        present_mode: vk::PresentModeKHR,
    ) -> Self {
        Self {
            entry,
            instance,
            physical_device,
            device,
            surface_loader,
            swapchain_loader,
            queue,
            command_pool,
            presenters: HashMap::new(),
            clear_color,
            present_mode,
        }
    }

    pub(crate) fn dirty_swapchain(&mut self, viewport_id: egui::ViewportId) {
        self.presenters
            .entry(viewport_id)
            .and_modify(|p| p.dirty_flag = true);
    }

    pub(crate) fn recreate_swapchain_if_needed(
        &mut self,
        viewport_id: egui::ViewportId,
        window: &winit::window::Window,
    ) {
        self.presenters.entry(viewport_id).and_modify(|p| {
            if p.dirty_flag {
                p.recreate(
                    self.physical_device,
                    &self.device,
                    &self.surface_loader,
                    &self.swapchain_loader,
                    self.command_pool,
                    window,
                );
            }
        });

        if self.presenters.get(&viewport_id).is_none() {
            if let Some(presenter) = Presenter::create(
                &self.entry,
                &self.instance,
                self.physical_device,
                self.device.clone(),
                &self.surface_loader,
                &self.swapchain_loader,
                self.command_pool,
                window,
                self.clear_color,
                self.present_mode,
            ) {
                self.presenters.insert(viewport_id, presenter);
            }
        }
    }

    pub(crate) fn destroy_swapchain_if_needed(&mut self, viewport_id: egui::ViewportId) {
        if let Some(presenter) = self.presenters.remove(&viewport_id) {
            presenter.destroy(
                &self.device,
                &self.surface_loader,
                &self.swapchain_loader,
                self.command_pool,
            );
        }
    }

    pub(crate) fn present_egui(&mut self, viewport_id: egui::ViewportId, egui_cmd: EguiCommand) {
        if let Some(presenter) = self.presenters.get_mut(&viewport_id) {
            // ignore Err to presenting swapchain image
            let _ = presenter.present(egui_cmd, &self.device, &self.swapchain_loader, self.queue);
        }
    }

    pub(crate) fn destroy_viewports(&mut self, active_viewport_ids: &egui::ViewportIdSet) {
        let remove_viewports = self
            .presenters
            .keys()
            .filter(|id| !active_viewport_ids.contains(id))
            .filter(|id| id != &&egui::ViewportId::ROOT)
            .map(|id| id.clone())
            .collect::<Vec<_>>();

        for id in remove_viewports {
            if let Some(presenter) = self.presenters.remove(&id) {
                presenter.destroy(
                    &self.device,
                    &self.surface_loader,
                    &self.swapchain_loader,
                    self.command_pool,
                );
            }
        }
    }

    pub(crate) fn destroy_root(&mut self) {
        for (_, presenter) in self.presenters.drain() {
            presenter.destroy(
                &self.device,
                &self.surface_loader,
                &self.swapchain_loader,
                self.command_pool,
            );
        }
    }
}
