pub struct Background {
    pub color: [f32; 3],
}

pub struct Suzanne {
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub rotation_z: f32,
    pub diffuse_color: [f32; 3],
    pub specular_color: [f32; 3],
    pub shininess: f32,
}

pub struct Light {
    pub position: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
}

pub struct Scene {
    pub background: Background,
    pub suzanne: Suzanne,
    pub light: Light,
}
impl Scene {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            background: Background {
                color: [0.0, 0.2, 0.4],
            },
            suzanne: Suzanne {
                rotation_x: 0.0,
                rotation_y: 0.0,
                rotation_z: 0.0,
                diffuse_color: [0.8, 0.5, 0.5],
                specular_color: [1.0, 1.0, 1.0],
                shininess: 10.0,
            },
            light: Light {
                position: [0.0, 5.0, -5.0],
                intensity: 2.0,
                color: [1.0, 1.0, 1.0],
            },
        }
    }
}
