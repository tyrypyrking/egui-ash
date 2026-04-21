//! Persistence store for window geometry, egui memory, and user-defined
//! key/value data.
//!
//! The `Storage` type is always available on the public API so that the
//! UI-closure signature passed to [`run`](crate::run) stays stable across
//! feature-flag configurations. Actual disk I/O and typed accessors are
//! gated on the `persistence` feature:
//!
//! - **With** `persistence`: `Storage` is backed by a RON file at
//!   `<data_dir>/<app_id>/app.ron` and exposes `set_value` / `get_value`
//!   for user-defined persistent state. Window layout and egui memory are
//!   also auto-saved when the corresponding `RunOption` flags are set.
//! - **Without** `persistence`: `Storage` is a zero-sized stub. The typed
//!   accessors are not compiled. User closures can still refer to the
//!   parameter; calls simply don't exist to invoke.

#[cfg(feature = "persistence")]
use anyhow::Result;
#[cfg(feature = "persistence")]
use egui_winit::WindowSettings;
#[cfg(feature = "persistence")]
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
};

// ─────────────────────────────────────────────────────────────────────────────
// Feature-on: real storage backed by a RON file on disk.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "persistence")]
struct InnerStorage {
    filepath: PathBuf,
    kv: HashMap<String, String>,
    dirty: bool,
    save_join_handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "persistence")]
impl InnerStorage {
    fn storage_dir(app_id: &str) -> Option<PathBuf> {
        directories_next::ProjectDirs::from("", "", app_id)
            .map(|project_dirs| project_dirs.data_dir().to_owned())
    }

    fn from_app_id(app_id: &str) -> Result<Self> {
        if let Some(dir) = Self::storage_dir(app_id) {
            match std::fs::create_dir_all(&dir) {
                Ok(_) => {
                    let filepath = dir.join("app.ron");
                    match std::fs::File::open(&filepath) {
                        Ok(file) => {
                            let kv: HashMap<String, String> = match ron::de::from_reader(file) {
                                Ok(kv) => kv,
                                Err(err) => {
                                    log::error!("Failed to deserialize storage: {}", err);
                                    HashMap::new()
                                }
                            };

                            Ok(Self {
                                filepath,
                                kv,
                                dirty: false,
                                save_join_handle: None,
                            })
                        }
                        Err(_) => Ok(Self {
                            filepath,
                            kv: HashMap::new(),
                            dirty: false,
                            save_join_handle: None,
                        }),
                    }
                }
                Err(err) => {
                    anyhow::bail!("Failed to create directory {dir:?}: {err}");
                }
            }
        } else {
            anyhow::bail!("Failed to get storage directory");
        }
    }

    fn get_value<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.kv
            .get(key)
            .cloned()
            .and_then(|value| match ron::from_str(&value) {
                Ok(value) => Some(value),
                Err(err) => {
                    log::error!("failed to deserialize value: {}", err);
                    None
                }
            })
    }

    fn set_value<T: serde::Serialize>(&mut self, key: &str, value: &T) {
        match ron::to_string(value) {
            Ok(value) => {
                self.kv.insert(key.to_owned(), value);
                self.dirty = true;
            }
            Err(err) => {
                log::error!("failed to serialize value: {}", err);
            }
        }
    }

    fn save_to_disk(filepath: &PathBuf, kv: HashMap<String, String>) {
        if let Some(parent_dir) = filepath.parent() {
            if !parent_dir.exists() {
                if let Err(err) = std::fs::create_dir_all(parent_dir) {
                    log::error!("Failed to create directory {parent_dir:?}: {err}");
                }
            }
        }

        match std::fs::File::create(filepath) {
            Ok(file) => {
                let config = ron::ser::PrettyConfig::new();

                if let Err(err) = ron::Options::default().to_io_writer_pretty(file, &kv, config) {
                    log::error!("Failed to serialize app state: {}", err);
                }
            }
            Err(err) => {
                log::error!("Failed to create file {filepath:?}: {err}");
            }
        }
    }

    fn flush(&mut self) {
        // In-memory only (fallback when from_app_id failed) — skip disk write.
        if self.filepath.as_os_str().is_empty() {
            self.dirty = false;
            return;
        }
        if self.dirty {
            self.dirty = false;
            let kv = self.kv.clone();
            let filepath = self.filepath.clone();

            if let Some(join_handle) = self.save_join_handle.take() {
                join_handle.join().ok();
            }

            self.save_join_handle = Some(std::thread::spawn(move || {
                Self::save_to_disk(&filepath, kv)
            }));
        }
    }

    fn set_egui_memory(&mut self, egui_memory: &egui::Memory) {
        self.set_value(STORAGE_EGUI_MEMORY_KEY, egui_memory);
    }

    fn get_egui_memory(&self) -> Option<egui::Memory> {
        self.get_value(STORAGE_EGUI_MEMORY_KEY)
    }

    fn set_windows(&mut self, windows: &HashMap<egui::ViewportId, WindowSettings>) {
        let mut prev_windows = self.get_windows().unwrap_or_default();
        for (id, window) in windows {
            prev_windows.insert(*id, *window);
        }
        self.set_value(STORAGE_WINDOWS_KEY, &prev_windows);
    }

    fn get_windows(&self) -> Option<HashMap<egui::ViewportId, WindowSettings>> {
        self.get_value(STORAGE_WINDOWS_KEY)
    }
}

#[cfg(feature = "persistence")]
impl Drop for InnerStorage {
    fn drop(&mut self) {
        if let Some(join_handle) = self.save_join_handle.take() {
            join_handle.join().ok();
        }
    }
}

#[cfg(feature = "persistence")]
impl Default for InnerStorage {
    fn default() -> Self {
        // In-memory only — an empty filepath tells `flush` to skip disk I/O.
        // Produced by `Storage::initialize` when `from_app_id` fails so the
        // UI closure still sees a valid `Storage`.
        Self {
            filepath: PathBuf::new(),
            kv: HashMap::new(),
            dirty: false,
            save_join_handle: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public Storage type — always available, methods feature-gated.
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent key/value store handed to the UI closure each frame.
///
/// With the `persistence` feature enabled, `set_value` / `get_value` persist
/// user state to an on-disk RON file keyed by the `app_id` passed to
/// [`run`](crate::run). Without the feature, this type is a zero-sized stub
/// and those methods do not exist — calls will not compile, keeping the
/// distinction between "persistence was configured" and "persistence is a
/// no-op" as a compile-time fact rather than silent runtime loss.
#[derive(Clone, Default)]
pub struct Storage {
    #[cfg(feature = "persistence")]
    inner: Arc<Mutex<InnerStorage>>,
}

impl Storage {
    /// Unified constructor used by `run()`. With the `persistence` feature
    /// enabled, attempts to load existing RON state from the per-app data
    /// directory; on failure logs and falls back to an in-memory-only
    /// store (so the UI closure still sees a valid `Storage`). Without the
    /// feature, returns a zero-sized stub.
    pub(crate) fn initialize(_app_id: &str) -> Self {
        #[cfg(feature = "persistence")]
        {
            match InnerStorage::from_app_id(_app_id) {
                Ok(inner) => Self {
                    inner: Arc::new(Mutex::new(inner)),
                },
                Err(e) => {
                    log::warn!(
                        "persistence: failed to open storage for app '{}': {:?}",
                        _app_id,
                        e
                    );
                    Self::default()
                }
            }
        }
        #[cfg(not(feature = "persistence"))]
        {
            Self::default()
        }
    }
}

#[cfg(feature = "persistence")]
impl Storage {
    pub(crate) fn flush(&self) {
        self.inner.lock().unwrap().flush();
    }

    pub(crate) fn set_egui_memory(&mut self, egui_memory: &egui::Memory) {
        self.inner.lock().unwrap().set_egui_memory(egui_memory);
    }

    pub(crate) fn get_egui_memory(&self) -> Option<egui::Memory> {
        self.inner.lock().unwrap().get_egui_memory()
    }

    pub(crate) fn set_windows(&mut self, windows: &HashMap<egui::ViewportId, WindowSettings>) {
        self.inner.lock().unwrap().set_windows(windows);
    }

    pub(crate) fn get_windows(&self) -> Option<HashMap<egui::ViewportId, WindowSettings>> {
        self.inner.lock().unwrap().get_windows()
    }

    /// Persist a typed value under the given key.
    ///
    /// Serialises via RON. Serialisation failures are logged and silently
    /// dropped — they do not propagate. Flushed to disk on shutdown, on any
    /// `auto_save_interval` tick, or when the `Storage` is dropped.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned — i.e., a prior panic on
    /// another thread occurred while holding it. Under normal operation
    /// this does not happen; the background save thread does not panic.
    pub fn set_value<T: serde::Serialize>(&mut self, key: &str, value: &T) {
        self.inner.lock().unwrap().set_value(key, value);
    }

    /// Load a typed value previously stored under `key`.
    ///
    /// Returns `None` if the key doesn't exist or if deserialization fails
    /// (e.g. the stored shape no longer matches `T` after a schema change).
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned — see [`Self::set_value`]
    /// for details. Under normal operation this does not happen.
    pub fn get_value<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.inner.lock().unwrap().get_value(key)
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage").finish()
    }
}

#[cfg(feature = "persistence")]
pub(crate) const STORAGE_EGUI_MEMORY_KEY: &str = "egui_memory";
#[cfg(feature = "persistence")]
pub(crate) const STORAGE_WINDOWS_KEY: &str = "egui_windows";
