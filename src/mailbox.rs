use std::sync::{mpsc, Mutex};

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
}

pub(crate) fn mailbox() -> (MailboxSender, MailboxReceiver) {
    let inner = std::sync::Arc::new(MailboxInner {
        data: Mutex::new(None),
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

impl MailboxReceiver {
    pub(crate) fn try_recv(&self) -> Option<CompletedFrame> {
        let mut lock = self.slot.data.lock().unwrap();
        lock.take()
    }
}

// ── Target channel: bounded mpsc for RenderTarget ──

pub(crate) use mpsc::SyncSender as TargetSender;
pub(crate) use mpsc::Receiver as TargetReceiver;

pub(crate) fn target_channel() -> (TargetSender<RenderTarget>, TargetReceiver<RenderTarget>) {
    mpsc::sync_channel(2)
}
