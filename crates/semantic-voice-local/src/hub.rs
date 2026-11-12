//! Resolution of model file references to local paths.
//!
//! A reference is either a local path or `hf://<owner>/<repo>/<file>[@<revision>]`, the format
//! of the upstream Pocket TTS configs. Hub files are cached in the standard Hugging Face cache;
//! a token is read from `HF_TOKEN` or the Hugging Face CLI login.

use std::path::PathBuf;

use crate::{Error, Result};

/// A file in a Hugging Face model repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HubFile {
    pub repo: String,
    pub file: String,
    pub revision: Option<String>,
}

impl HubFile {
    /// Parses `hf://owner/repo/path/to/file[@revision]`.
    pub fn parse(reference: &str) -> Option<Self> {
        let rest = reference.strip_prefix("hf://")?;
        let (path, revision) = match rest.rsplit_once('@') {
            Some((path, revision)) => (path, Some(revision.to_string())),
            None => (rest, None),
        };
        let mut parts = path.splitn(3, '/');
        let owner = parts.next()?;
        let name = parts.next()?;
        let file = parts.next()?;
        Some(Self {
            repo: format!("{owner}/{name}"),
            file: file.to_string(),
            revision,
        })
    }

    fn display(&self) -> String {
        match &self.revision {
            Some(revision) => format!("hf://{}/{}@{revision}", self.repo, self.file),
            None => format!("hf://{}/{}", self.repo, self.file),
        }
    }

    #[cfg(feature = "hub")]
    pub fn fetch(&self) -> Result<PathBuf> {
        use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

        let err = |reason: String| Error::Download {
            path: self.display(),
            reason,
        };
        let mut builder = ApiBuilder::from_env();
        if let Ok(token) = std::env::var("HF_TOKEN")
            && !token.is_empty()
        {
            builder = builder.with_token(Some(token));
        }
        let api = builder.build().map_err(|e| err(e.to_string()))?;
        let repo = match &self.revision {
            Some(revision) => {
                Repo::with_revision(self.repo.clone(), RepoType::Model, revision.clone())
            }
            None => Repo::model(self.repo.clone()),
        };
        tracing::debug!("fetching {}", self.display());
        api.repo(repo)
            .get(&self.file)
            .map_err(|e| err(e.to_string()))
    }

    #[cfg(not(feature = "hub"))]
    pub fn fetch(&self) -> Result<PathBuf> {
        Err(Error::Download {
            path: self.display(),
            reason: "built without the `hub` feature, use local files".into(),
        })
    }
}

/// Resolves a file reference (local path or `hf://` URL) to a local path.
pub(crate) fn resolve(reference: &str) -> Result<PathBuf> {
    match HubFile::parse(reference) {
        Some(file) => file.fetch(),
        None if reference.starts_with("hf://") => Err(Error::config(format!(
            "malformed hub reference {reference:?}"
        ))),
        None => {
            let path = PathBuf::from(reference);
            if path.exists() {
                Ok(path)
            } else {
                Err(Error::Download {
                    path: reference.to_string(),
                    reason: "file not found".into(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hub_references() {
        let file = HubFile::parse("hf://kyutai/pocket-tts/languages/english/model.safetensors@abc")
            .unwrap();
        assert_eq!(file.repo, "kyutai/pocket-tts");
        assert_eq!(file.file, "languages/english/model.safetensors");
        assert_eq!(file.revision.as_deref(), Some("abc"));

        let file = HubFile::parse("hf://kyutai/tts-voices/alba-mackenna/casual.wav").unwrap();
        assert_eq!(file.file, "alba-mackenna/casual.wav");
        assert_eq!(file.revision, None);

        assert!(HubFile::parse("hf://kyutai").is_none());
        assert!(HubFile::parse("/tmp/model.safetensors").is_none());
    }
}
