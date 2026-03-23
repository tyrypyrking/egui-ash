use crate::types::{CompletedFrame, EngineContext, RenderTarget};
use crate::event::EngineEvent;

/// Trait implemented by the user's rendering engine.
/// Runs on a dedicated thread spawned by the library.
///
/// # Panic Safety
/// If any method panics, the library catches it via `catch_unwind`.
/// The UI remains responsive, showing a black viewport.
/// Use `EngineHandle::restart()` from the UI to spawn a fresh instance.
///
/// # Drop
/// `destroy()` is called on clean shutdown only. Implement `Drop` for
/// GPU resource cleanup that must happen even after a panic.
pub trait EngineRenderer: Send + 'static {
    /// State written by the UI closure, read by the engine each frame.
    type UiState: Send + Sync + Clone + Default;

    /// State written by the engine, read by the UI closure each frame.
    type EngineState: Send + Sync + Clone + Default;

    /// Called once on the engine thread after spawning.
    fn init(&mut self, ctx: EngineContext);

    /// Render a frame into the provided target.
    ///
    /// The engine must:
    /// 1. Wait on `target.wait_value` (GPU wait via timeline semaphore in submit)
    /// 2. Include `target.acquire_barrier()` in the command buffer if `Some`
    /// 3. Render into `target.image`
    /// 4. Include `target.release_barrier()` in the command buffer if `Some`
    /// 5. Signal `target.signal_value` via timeline semaphore in submit
    /// 6. Return `target.complete()`
    fn render(
        &mut self,
        target: RenderTarget,
        ui_state: &Self::UiState,
        engine_state: &mut Self::EngineState,
    ) -> CompletedFrame;

    /// Process an input event forwarded from the host.
    fn handle_event(&mut self, event: EngineEvent);

    /// Called on clean shutdown before the engine thread exits.
    fn destroy(&mut self);
}
