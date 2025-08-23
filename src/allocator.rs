use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Result;
use ash::vk::*;

/// Represents a memory allocation.
pub trait Allocation: Send + Sync {
    /// Returns the vk::DeviceMemory object that is backing this allocation.
    /// # Safety
    /// TODO
    unsafe fn memory(&self) -> DeviceMemory;

    /// Returns the offset of the allocation on the vk::DeviceMemory. When binding the memory to a buffer or image, this offset needs to be supplied as well.
    fn offset(&self) -> u64;

    /// Returns the size of the allocation
    fn size(&self) -> u64;

    /// Returns a valid mapped pointer if the memory is host visible, otherwise it will return None. The pointer already points to the exact memory region of the suballocation, so no offset needs to be applied.
    fn mapped_ptr(&self) -> Option<NonNull<c_void>>;
}

/// Represents a memory location for [`AllocationCreateInfo`].
pub trait MemoryLocation: Clone + Send + Sync {
    fn gpu_only() -> Self;
    fn cpu_to_gpu() -> Self;
    fn gpu_to_cpu() -> Self;
}

/// allocation create info for [`Allocator::allocate`].
pub trait AllocationCreateInfo {
    type MemoryLocation: MemoryLocation;
    fn new(
        name: Option<&'static str>,
        requirements: MemoryRequirements,
        location: Self::MemoryLocation,
        linear: bool,
    ) -> Self;
}

/// trait for GPU memory allocator.
pub trait Allocator: Clone + Send + Sync {
    type Allocation: Allocation;
    type AllocationCreateInfo: AllocationCreateInfo;

    /// Allocate a new memory region.
    fn allocate(&self, desc: Self::AllocationCreateInfo) -> Result<Self::Allocation>;

    /// Free a memory region.
    fn free(&self, allocation: Self::Allocation) -> Result<()>;
}
