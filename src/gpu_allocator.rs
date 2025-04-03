use std::sync::{Arc, Mutex};

use anyhow::Result;
use gpu_allocator::{vulkan::*, MemoryLocation};

use crate::allocator;

impl allocator::Allocation for Allocation {
    unsafe fn memory(&self) -> ash::vk::DeviceMemory {
        Allocation::memory(self)
    }

    fn offset(&self) -> u64 {
        Allocation::offset(self)
    }

    fn size(&self) -> u64 {
        Allocation::size(self)
    }

    fn mapped_ptr(&self) -> Option<std::ptr::NonNull<std::ffi::c_void>> {
        Allocation::mapped_ptr(self)
    }
}

impl allocator::MemoryLocation for MemoryLocation {
    fn gpu_only() -> Self {
        MemoryLocation::GpuOnly
    }

    fn cpu_to_gpu() -> Self {
        MemoryLocation::CpuToGpu
    }

    fn gpu_to_cpu() -> Self {
        MemoryLocation::GpuToCpu
    }
}

impl allocator::AllocationCreateInfo for AllocationCreateDesc<'static> {
    type MemoryLocation = gpu_allocator::MemoryLocation;
    fn new(
        name: Option<&'static str>,
        requirements: ash::vk::MemoryRequirements,
        location: Self::MemoryLocation,
        linear: bool,
    ) -> Self {
        let name = name.unwrap_or("Name not provided");
        Self {
            name,
            requirements,
            location,
            linear,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }
    }
}

impl allocator::Allocator for Arc<Mutex<Allocator>> {
    type Allocation = Allocation;
    type AllocationCreateInfo = AllocationCreateDesc<'static>;

    fn allocate(&self, desc: Self::AllocationCreateInfo) -> Result<Self::Allocation> {
        Ok(Allocator::allocate(&mut self.lock().unwrap(), &desc)?)
    }

    fn free(&self, allocation: Self::Allocation) -> Result<()> {
        Ok(Allocator::free(&mut self.lock().unwrap(), allocation)?)
    }
}
