use std::sync::{mpsc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::types::{CompletedFrame, RenderTarget};

// ── Mailbox: single-slot overwrite channel ──

pub(crate) struct MailboxSender {
    slot: std::sync::Arc<MailboxInner>,
}

pub(crate) struct MailboxReceiver {
    slot: std::sync::Arc<MailboxInner>,
}

struct MailboxInner {
    data: Mutex<Option<CompletedFrame>>,
    closed: AtomicBool,
}

pub(crate) fn mailbox() -> (MailboxSender, MailboxReceiver) {
    let inner = std::sync::Arc::new(MailboxInner {
        data: Mutex::new(None),
        closed: AtomicBool::new(false),
    });
    (
        MailboxSender { slot: inner.clone() },
        MailboxReceiver { slot: inner },
    )
}

impl MailboxSender {
    pub(crate) fn send(&self, frame: CompletedFrame) {
        let mut lock = self.slot.data.lock().unwrap();
        *lock = Some(frame);
    }
}

impl Drop for MailboxSender {
    fn drop(&mut self) {
        self.slot.closed.store(true, Ordering::Release);
    }
}

impl MailboxReceiver {
    pub(crate) fn try_recv(&self) -> Option<CompletedFrame> {
        let mut lock = self.slot.data.lock().unwrap();
        lock.take()
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.slot.closed.load(Ordering::Acquire)
    }
}

// ── Target channel: bounded mpsc for RenderTarget ──

pub(crate) use mpsc::SyncSender as TargetSender;
pub(crate) use mpsc::Receiver as TargetReceiver;

pub(crate) fn target_channel() -> (TargetSender<RenderTarget>, TargetReceiver<RenderTarget>) {
    mpsc::sync_channel(2)
}
