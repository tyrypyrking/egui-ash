use anyhow::Result;
use egui_winit::WindowSettings;
use std::{
    collections::HashMap,
    fmt::Debug,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

struct InnerStorage {
    filepath: PathBuf,
    kv: HashMap<String, String>,
    dirty: bool,
    save_join_handle: Option<std::thread::JoinHandle<()>>,
}
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
                let mut writer = std::io::BufWriter::new(file);
                let config = Default::default();

                if let Err(err) = ron::ser::to_writer_pretty(&mut writer, &kv, config)
                    .and_then(|_| writer.flush().map_err(|err| err.into()))
                {
                    log::error!("Failed to serialize app state: {}", err);
                }
            }
            Err(err) => {
                log::error!("Failed to create file {filepath:?}: {err}");
            }
        }
    }

    fn flush(&mut self) {
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
impl Drop for InnerStorage {
    fn drop(&mut self) {
        if let Some(join_handle) = self.save_join_handle.take() {
            join_handle.join().ok();
        }
    }
}

#[derive(Clone)]
pub struct Storage {
    inner: Arc<Mutex<InnerStorage>>,
}
impl Storage {
    pub(crate) fn from_app_id(app_id: &str) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(Mutex::new(InnerStorage::from_app_id(app_id)?)),
        })
    }

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

    /// Set value to storage.
    pub fn set_value<T: serde::Serialize>(&mut self, key: &str, value: &T) {
        self.inner.lock().unwrap().set_value(key, value);
    }

    /// Get value from storage.
    pub fn get_value<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.inner.lock().unwrap().get_value(key)
    }
}
impl Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage").finish()
    }
}

pub(crate) const STORAGE_EGUI_MEMORY_KEY: &str = "egui_memory";
pub(crate) const STORAGE_WINDOWS_KEY: &str = "egui_windows";
