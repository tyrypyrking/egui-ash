use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use crate::engine::EngineRenderer;
use crate::event::EngineEvent;
use crate::mailbox::{MailboxSender, TargetReceiver};
use crate::state_exchange::{StateReader, StateWriter};
use crate::types::EngineContext;

// Health encoding for AtomicU8
pub(crate) const HEALTH_STARTING: u8 = 0;
pub(crate) const HEALTH_RUNNING: u8 = 1;
pub(crate) const HEALTH_STOPPED: u8 = 2;
pub(crate) const HEALTH_CRASHED: u8 = 3;

/// Shared health state between engine thread and host.
pub(crate) struct EngineHealthState {
    pub(crate) health: AtomicU8,
    pub(crate) crash_message: std::sync::Mutex<Option<String>>,
    pub(crate) frames_delivered: std::sync::atomic::AtomicU64,
    pub(crate) last_frame_time_ns: std::sync::atomic::AtomicU64,
}

impl EngineHealthState {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            health: AtomicU8::new(HEALTH_STARTING),
            crash_message: std::sync::Mutex::new(None),
            frames_delivered: std::sync::atomic::AtomicU64::new(0),
            last_frame_time_ns: std::sync::atomic::AtomicU64::new(0),
        })
    }
}

/// Spawn the engine thread. Returns the JoinHandle.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_engine_thread<E: EngineRenderer>(
    mut engine: E,
    ctx: EngineContext,
    targets_rx: TargetReceiver<crate::types::RenderTarget>,
    frames_tx: MailboxSender,
    events_rx: mpsc::Receiver<EngineEvent>,
    ui_state_reader: StateReader<E::UiState>,
    engine_state_writer: StateWriter<E::EngineState>,
    health: Arc<EngineHealthState>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("egui-ash-engine".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                engine.init(ctx);
                health.health.store(HEALTH_RUNNING, Ordering::Release);

                let mut engine_state = E::EngineState::default();

                while let Ok(target) = targets_rx.recv() {
                    // Drain input events
                    for event in events_rx.try_iter() {
                        engine.handle_event(event);
                    }

                    // Read latest UI state
                    let ui_state = ui_state_reader.read();

                    let frame_start = Instant::now();
                    let frame = engine.render(target, &ui_state, &mut engine_state);
                    let frame_time = frame_start.elapsed();

                    // Publish engine state
                    engine_state_writer.publish(engine_state.clone());

                    // Update stats
                    health.frames_delivered.fetch_add(1, Ordering::Relaxed);
                    health
                        .last_frame_time_ns
                        .store(frame_time.as_nanos() as u64, Ordering::Relaxed);

                    // Send completed frame
                    frames_tx.send(frame);
                }

                engine.destroy();
            }));

            match result {
                Ok(()) => {
                    health.health.store(HEALTH_STOPPED, Ordering::Release);
                }
                Err(panic_info) => {
                    let message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    *health.crash_message.lock().unwrap() = Some(message);
                    health.health.store(HEALTH_CRASHED, Ordering::Release);
                }
            }
        })
        .expect("failed to spawn engine thread")
}
