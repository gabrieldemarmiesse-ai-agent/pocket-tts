//! Loading pocket-tts safetensors weights into xn-ptts, whose names for them differ, and reading
//! the voice states the Python package writes.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const SPEAKER_SINK: &str = "flow_lm.condition_provider.conditioners.speaker_wavs.sink";

/// Where a tensor sits in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    start: u64,
    end: u64,
}

/// The header of a safetensors file, read without loading the tensors.
pub struct SafetensorsFile {
    path: PathBuf,
    data_start: u64,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)?;
        let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header)
            .with_context(|| format!("{} is not a safetensors file", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                continue;
            }
            let info = (|| {
                let dtype = value.get("dtype")?.as_str()?.to_string();
                let shape = value
                    .get("shape")?
                    .as_array()?
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize))
                    .collect::<Option<Vec<_>>>()?;
                let offsets = value.get("data_offsets")?.as_array()?;
                Some(TensorInfo {
                    dtype,
                    shape,
                    start: offsets.first()?.as_u64()?,
                    end: offsets.get(1)?.as_u64()?,
                })
            })()
            .with_context(|| format!("bad entry {name} in {}", path.display()))?;
            tensors.insert(name, info);
        }
        Ok(Self {
            path: path.to_path_buf(),
            data_start: 8 + len,
            tensors,
        })
    }

    fn read_bytes(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let mut file = std::fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.data_start + info.start))?;
        let mut bytes = vec![0u8; (info.end - info.start) as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    /// A floating point tensor, converted to f32.
    pub fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("no tensor {name}"))?;
        let bytes = self.read_bytes(info)?;
        Ok((bytes_to_f32(&info.dtype, &bytes)?, info.shape.clone()))
    }

    /// An integer tensor, converted to i64.
    pub fn read_i64(&self, name: &str) -> Result<Vec<i64>> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("no tensor {name}"))?;
        let bytes = self.read_bytes(info)?;
        Ok(match info.dtype.as_str() {
            "I64" => bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b))
                .collect(),
            "I32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b) as i64)
                .collect(),
            other => bail!("tensor {name} has dtype {other}, expected an integer"),
        })
    }
}

pub fn bytes_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(match dtype {
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect(),
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => bail!("unsupported dtype {other}, expected a float"),
    })
}

/// pocket-tts key names to xn-ptts ones.
fn remap_key(name: &str) -> Option<String> {
    let name = ptts::loader::remap_key(name)?;
    // pocket-tts prepends `bos_before_voice` to the voice prompt, which xn-ptts models as the
    // sink of the `speaker_wavs` conditioning, see `Config::to_ptts`.
    if name == "flow_lm.bos_before_voice" {
        return Some(SPEAKER_SINK.to_string());
    }
    Some(name)
}

/// The weights as xn-ptts loads them.
pub struct Weights {
    pub vb: xn::nn::Path<xn::CpuDevice>,
}

/// Memory-maps the checkpoint at `path`.
pub fn load(path: &Path) -> Result<Weights> {
    let vb = xn::nn::VB::load_with_key_map(&[path], xn::CPU, remap_key)
        .with_context(|| format!("cannot load the weights in {}", path.display()))?;
    Ok(Weights { vb: vb.root() })
}
