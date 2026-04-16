//! A reusable `ColorEngine` that clears render targets to a solid color.
//!
//! Used by multiple examples to demonstrate the v2 `EngineRenderer` trait
//! without requiring a full 3D pipeline.

#![allow(dead_code)]

use ash::vk;
use egui_ash::{CompletedFrame, EngineContext, EngineEvent, EngineRenderer, RenderTarget};

// ─────────────────────────────────────────────────────────────────────────────
// UiState / EngineState
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ColorUiState {
    pub clear_color: [f32; 3],
    pub should_crash: bool,
}

impl Default for ColorUiState {
    fn default() -> Self {
        Self {
            clear_color: [0.2, 0.3, 0.8],
            should_crash: false,
        }
    }
}

#[derive(Clone, Default)]
pub struct ColorEngineState {
    pub frame_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ColorEngine
// ─────────────────────────────────────────────────────────────────────────────

pub struct ColorEngine {
    device: Option<ash::Device>,
    queue: vk::Queue,
    queue_family_index: u32,
    queue_mutex: Option<std::sync::Arc<std::sync::Mutex<()>>>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    frame_count: u64,
}

impl ColorEngine {
    pub fn new() -> Self {
        Self {
            device: None,
            queue: vk::Queue::null(),
            queue_family_index: u32::MAX,
            queue_mutex: None,
            command_pool: vk::CommandPool::null(),
            command_buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            frame_count: 0,
        }
    }
}

impl EngineRenderer for ColorEngine {
    type UiState = ColorUiState;
    type EngineState = ColorEngineState;

    fn init(&mut self, ctx: EngineContext) {
        self.queue = ctx.queue;
        self.queue_family_index = ctx.queue_family_index;
        self.queue_mutex = ctx.queue_mutex;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(ctx.queue_family_index);

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            let pool = ctx
                .device
                .create_command_pool(&pool_info, None)
                .expect("failed to create command pool");
            let cmd = ctx
                .device
                .allocate_command_buffers(&cmd_alloc_info.command_pool(pool))
                .expect("failed to allocate command buffer")[0];
            self.command_pool = pool;
            self.command_buffer = cmd;
            self.fence = ctx
                .device
                .create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
                .expect("failed to create fence");
        }

        self.device = Some(ctx.device);
    }

    fn render(
        &mut self,
        target: RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> CompletedFrame {
        if ui_state.should_crash {
            panic!("Crash requested by UI");
        }

        let device = self.device.as_ref().expect("engine not initialized");
        let cmd = self.command_buffer;

        unsafe {
            // Wait for the previous frame's GPU work to complete before reusing the command buffer.
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .expect("failed to wait for fence");
            device
                .reset_fences(&[self.fence])
                .expect("failed to reset fence");

            // Reset and begin command buffer
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("failed to reset command buffer");

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin_info)
                .expect("failed to begin command buffer");

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            // If cross-queue-family: insert acquire barrier
            // Otherwise: transition from UNDEFINED to TRANSFER_DST_OPTIMAL
            if let Some(acquire) = target.acquire_barrier() {
                let barriers = [*acquire];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                // COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_DST_OPTIMAL
                let to_transfer = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [to_transfer];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                // Same-family: simple transition UNDEFINED -> TRANSFER_DST_OPTIMAL
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            // Clear the image
            let clear_color = vk::ClearColorValue {
                float32: [
                    ui_state.clear_color[0],
                    ui_state.clear_color[1],
                    ui_state.clear_color[2],
                    1.0,
                ],
            };
            let ranges = [subresource_range];
            device.cmd_clear_color_image(
                cmd,
                target.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &ranges,
            );

            // Transition to SHADER_READ_ONLY_OPTIMAL (for compositor sampling)
            if let Some(release) = target.release_barrier() {
                // First: TRANSFER_DST_OPTIMAL -> COLOR_ATTACHMENT_OPTIMAL
                let to_color_attach = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [to_color_attach];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);

                // Cross-family release barrier
                let barriers = [*release];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            } else {
                // Same-family: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(target.image)
                    .subresource_range(subresource_range);
                let barriers = [barrier];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
                device.cmd_pipeline_barrier2(cmd, &dep_info);
            }

            device
                .end_command_buffer(cmd)
                .expect("failed to end command buffer");

            // Wait on compositor timeline, signal engine timeline.
            let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.wait_semaphore)
                .value(target.wait_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
                .semaphore(target.signal_semaphore)
                .value(target.signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_semaphore_infos)
                .signal_semaphore_infos(&signal_semaphore_infos)
                .command_buffer_infos(&cmd_buf_infos);
            {
                let _qlock = self
                    .queue_mutex
                    .as_ref()
                    .map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
                device
                    .queue_submit2(self.queue, &[submit_info], self.fence)
                    .expect("failed to submit command buffer");
            }
        }

        self.frame_count += 1;
        engine_state.frame_count = self.frame_count;

        target.complete()
    }

    fn handle_event(&mut self, _event: EngineEvent) {
        // No input handling needed for this simple engine.
    }

    fn destroy(&mut self) {
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                if self.fence != vk::Fence::null() {
                    device.destroy_fence(self.fence, None);
                    self.fence = vk::Fence::null();
                }
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                    self.command_pool = vk::CommandPool::null();
                }
            }
        }
    }
}

impl Drop for ColorEngine {
    fn drop(&mut self) {
        // Safety net: destroy GPU resources even after a panic.
        if let Some(device) = self.device.take() {
            unsafe {
                device.device_wait_idle().ok();
                if self.fence != vk::Fence::null() {
                    device.destroy_fence(self.fence, None);
                    self.fence = vk::Fence::null();
                }
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                    self.command_pool = vk::CommandPool::null();
                }
            }
        }
    }
}
