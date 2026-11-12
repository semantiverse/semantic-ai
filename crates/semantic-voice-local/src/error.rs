/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid model configuration: {0}")]
    Config(String),

    #[error("could not fetch {path}: {reason}")]
    Download { path: String, reason: String },

    #[error("audio error: {0}")]
    Audio(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error(
        "voice cloning needs the gated model weights: accept the terms at \
         https://huggingface.co/kyutai/pocket-tts and set HF_TOKEN, or use a predefined voice"
    )]
    VoiceCloningUnavailable,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub(crate) fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub(crate) fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }
}
