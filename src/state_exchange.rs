use arc_swap::ArcSwap;
use std::sync::Arc;

pub(crate) struct StateWriter<T> {
    swap: Arc<ArcSwap<T>>,
}

pub(crate) struct StateReader<T> {
    swap: Arc<ArcSwap<T>>,
}

pub(crate) fn state_exchange<T: Default>() -> (StateWriter<T>, StateReader<T>) {
    let swap = Arc::new(ArcSwap::from_pointee(T::default()));
    (
        StateWriter { swap: swap.clone() },
        StateReader { swap },
    )
}

impl<T> StateWriter<T> {
    pub(crate) fn publish(&self, value: T) {
        self.swap.store(Arc::new(value));
    }
}

impl<T: Clone> StateReader<T> {
    pub(crate) fn read(&self) -> T {
        (**self.swap.load()).clone()
    }
}
