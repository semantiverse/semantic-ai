//! Pocket TTS model configuration, in the YAML format of the upstream package.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::nn::mimi::{MimiConfig, ProjectedTransformerConfig};
use crate::nn::seanet::SeanetConfig;
use crate::{Error, Result};

/// The configs shipped with upstream Pocket TTS (MIT licensed), by model name.
const BUILTIN: &[(&str, &str)] = &[
    ("english", include_str!("../../configs/tts/english.yaml")),
    (
        "english_2026-01",
        include_str!("../../configs/tts/english_2026-01.yaml"),
    ),
    (
        "english_2026-04",
        include_str!("../../configs/tts/english_2026-04.yaml"),
    ),
    (
        "english_2026-04_24l",
        include_str!("../../configs/tts/english_2026-04_24l.yaml"),
    ),
    (
        "english_2026-09",
        include_str!("../../configs/tts/english_2026-09.yaml"),
    ),
    (
        "english_2026-09_2",
        include_str!("../../configs/tts/english_2026-09_2.yaml"),
    ),
    (
        "english_2026-09_24l",
        include_str!("../../configs/tts/english_2026-09_24l.yaml"),
    ),
    (
        "english_2026-10",
        include_str!("../../configs/tts/english_2026-10.yaml"),
    ),
    (
        "english_drifting_26-09",
        include_str!("../../configs/tts/english_drifting_26-09.yaml"),
    ),
    (
        "english_18m_2026-10",
        include_str!("../../configs/tts/english_18m_2026-10.yaml"),
    ),
    (
        "english_18m_refl_2026-10",
        include_str!("../../configs/tts/english_18m_refl_2026-10.yaml"),
    ),
    (
        "english_7m_2026-10",
        include_str!("../../configs/tts/english_7m_2026-10.yaml"),
    ),
    (
        "english_7m_refl_2026-10",
        include_str!("../../configs/tts/english_7m_refl_2026-10.yaml"),
    ),
    (
        "english_2m_2026-10",
        include_str!("../../configs/tts/english_2m_2026-10.yaml"),
    ),
    (
        "english_2m_refl_2026-10",
        include_str!("../../configs/tts/english_2m_refl_2026-10.yaml"),
    ),
    ("french", include_str!("../../configs/tts/french.yaml")),
    (
        "french_24l",
        include_str!("../../configs/tts/french_24l.yaml"),
    ),
    ("german", include_str!("../../configs/tts/german.yaml")),
    (
        "german_24l",
        include_str!("../../configs/tts/german_24l.yaml"),
    ),
    ("italian", include_str!("../../configs/tts/italian.yaml")),
    (
        "italian_24l",
        include_str!("../../configs/tts/italian_24l.yaml"),
    ),
    (
        "portuguese",
        include_str!("../../configs/tts/portuguese.yaml"),
    ),
    (
        "portuguese_24l",
        include_str!("../../configs/tts/portuguese_24l.yaml"),
    ),
    ("spanish", include_str!("../../configs/tts/spanish.yaml")),
    (
        "spanish_24l",
        include_str!("../../configs/tts/spanish_24l.yaml"),
    ),
    ("dutch", include_str!("../../configs/tts/dutch.yaml")),
    (
        "dutch_24l",
        include_str!("../../configs/tts/dutch_24l.yaml"),
    ),
];

/// Names of the built-in models, for [`super::Tts::load`].
pub fn builtin_models() -> impl Iterator<Item = &'static str> {
    BUILTIN.iter().map(|(name, _)| *name)
}

pub(crate) fn builtin_config(name: &str) -> Result<TtsConfig> {
    let (_, yaml) = BUILTIN.iter().find(|(n, _)| *n == name).ok_or_else(|| {
        let available = builtin_models().collect::<Vec<_>>().join(", ");
        Error::config(format!("unknown model {name:?}, available: {available}"))
    })?;
    TtsConfig::from_yaml(yaml)
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsConfig {
    pub flow_lm: FlowLmConfig,
    pub mimi: CodecConfig,
    pub weights_path: Option<String>,
    pub weights_path_without_voice_cloning: Option<String>,
    #[serde(default)]
    pub pad_with_spaces_for_short_inputs: bool,
    #[serde(default)]
    pub remove_semicolons: bool,
    #[serde(default = "yes")]
    pub append_terminal_punctuation: bool,
    #[serde(default = "yes")]
    pub capitalize_first_letter: bool,
    /// Per-character rewrites applied before tokenization; an empty value deletes.
    #[serde(default)]
    pub replace_characters: BTreeMap<String, String>,
    pub model_recommended_frames_after_eos: Option<usize>,
    #[serde(default = "default_temperature")]
    pub default_temperature: f32,
    #[serde(default = "default_eos_threshold")]
    pub default_eos_threshold: f32,
}

fn yes() -> bool {
    true
}

fn default_temperature() -> f32 {
    0.3
}

fn default_eos_threshold() -> f32 {
    -4.0
}

impl TtsConfig {
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_yaml::from_str(yaml).map_err(|e| Error::config(e.to_string()))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLmConfig {
    pub dtype: String,
    pub flow: FlowHeadConfig,
    pub transformer: FlowLmTransformerConfig,
    /// Width of the backbone output fed to the heads, through a projection; `d_model` if unset.
    pub z_dim: Option<usize>,
    pub lookup_table: LookupTableConfig,
    pub weights_path: Option<String>,
    #[serde(default)]
    pub insert_bos_before_voice: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlowKind {
    /// Lagrangian self-distillation: two time conditions.
    #[default]
    Lsd,
    /// Optimal-transport flow matching with Euler integration: one time condition.
    FlowMatching,
    /// The head maps noise to a sample in one step.
    Drifting,
    /// One step, noise plus the head's output.
    Pointwise,
}

impl FlowKind {
    pub fn num_time_conditions(self) -> usize {
        match self {
            Self::Lsd => 2,
            Self::FlowMatching => 1,
            Self::Drifting | Self::Pointwise => 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowHeadConfig {
    pub dim: usize,
    pub depth: usize,
    #[serde(rename = "type", default)]
    pub kind: FlowKind,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowLmTransformerConfig {
    pub hidden_scale: usize,
    pub max_period: f64,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub attn_dim: Option<usize>,
    pub num_kv_heads: Option<usize>,
    pub dim_feedforward: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookupTableConfig {
    pub dim: usize,
    pub n_bins: usize,
    pub tokenizer: String,
    pub tokenizer_path: String,
    /// Low-rank table: `rank`-wide embeddings and a projection to `dim`.
    pub rank: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodecConfig {
    pub dtype: String,
    pub sample_rate: u32,
    pub channels: usize,
    pub frame_rate: f64,
    pub seanet: SeanetConfig,
    pub transformer: ProjectedTransformerConfig,
    pub quantizer: QuantizerConfig,
    pub weights_path: Option<String>,
    pub inner_dim: Option<usize>,
    pub outer_dim: Option<usize>,
    pub decoder_seanet: Option<SeanetConfig>,
    pub decoder_transformer: Option<ProjectedTransformerConfig>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuantizerConfig {
    /// Latent width, as generated by the flow LM.
    pub dimension: usize,
    pub output_dimension: usize,
}

impl CodecConfig {
    pub(crate) fn mimi(&self) -> MimiConfig {
        let decoder = self
            .decoder_seanet
            .clone()
            .unwrap_or_else(|| self.seanet.clone());
        MimiConfig {
            sample_rate: self.sample_rate,
            frame_rate: self.frame_rate,
            inner_dim: self.inner_dim.unwrap_or(self.seanet.dimension),
            outer_dim: self.outer_dim.unwrap_or(decoder.dimension),
            encoder: self.seanet.clone(),
            decoder,
            encoder_transformer: self.transformer.clone(),
            decoder_transformer: self
                .decoder_transformer
                .clone()
                .unwrap_or_else(|| self.transformer.clone()),
            exact_gelu: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_builtin_configs_parse() {
        for name in builtin_models() {
            let config = builtin_config(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(config.mimi.sample_rate, 24_000, "{name}");
        }
    }
}
