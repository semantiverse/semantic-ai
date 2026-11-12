//! A pure-Rust SentencePiece unigram tokenizer.
//!
//! Kyutai's models use SentencePiece unigram vocabularies with byte fallback, distributed either
//! as SentencePiece `.model` files or as Hugging Face `tokenizer.json` conversions of them. Both
//! are loaded here and encoded with SentencePiece's Viterbi segmentation, which gives the same
//! ids as the reference implementations.

use std::collections::HashMap;
use std::path::Path;

use crate::{Error, Result};

const WHITESPACE: char = '\u{2581}';
/// Penalty SentencePiece gives unknown characters, below the lowest piece score.
const UNKNOWN_PENALTY: f32 = 10.0;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<Piece>,
    /// Pieces the segmentation may produce, by text.
    lookup: HashMap<String, u32>,
    max_piece_len: usize,
    byte_pieces: Option<[u32; 256]>,
    unk_id: u32,
    unk_score: f32,
    normalizer: Normalizer,
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    score: f32,
    kind: PieceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte(u8),
    Unused,
}

#[derive(Debug, Clone)]
struct Normalizer {
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    /// Set when the model needs a normalization this implementation does not have.
    unsupported: Option<String>,
}

impl Tokenizer {
    /// Loads a SentencePiece `.model` or a `tokenizer.json`, chosen by file extension.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        if path.extension().is_some_and(|e| e == "json") {
            Self::from_tokenizer_json(&bytes)
        } else {
            Self::from_sentencepiece(&bytes)
        }
    }

    /// Parses a serialized SentencePiece `ModelProto`.
    pub fn from_sentencepiece(bytes: &[u8]) -> Result<Self> {
        let model = proto::ModelProto::parse(bytes)
            .map_err(|e| Error::Tokenizer(format!("invalid sentencepiece model: {e}")))?;
        if model.model_type != proto::MODEL_TYPE_UNIGRAM {
            return Err(Error::Tokenizer(format!(
                "only unigram sentencepiece models are supported (model type {})",
                model.model_type
            )));
        }
        let pieces = model
            .pieces
            .into_iter()
            .map(|p| {
                let kind = match p.kind {
                    1 => PieceKind::Normal,
                    2 => PieceKind::Unknown,
                    3 => PieceKind::Control,
                    4 => PieceKind::UserDefined,
                    6 => PieceKind::Byte(parse_byte_piece(&p.piece).ok_or_else(|| {
                        Error::Tokenizer(format!("invalid byte piece {:?}", p.piece))
                    })?),
                    _ => PieceKind::Unused,
                };
                Ok(Piece {
                    text: p.piece,
                    score: p.score,
                    kind,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n = model.normalizer;
        let unsupported =
            (n.name != "identity" && !n.precompiled_charsmap.is_empty()).then(|| n.name.clone());
        let normalizer = Normalizer {
            add_dummy_prefix: n.add_dummy_prefix,
            remove_extra_whitespaces: n.remove_extra_whitespaces,
            escape_whitespaces: n.escape_whitespaces,
            unsupported,
        };
        Self::new(pieces, model.byte_fallback, normalizer)
    }

    /// Parses a Hugging Face `tokenizer.json` holding a unigram model, as converted from
    /// SentencePiece (Metaspace pre-tokenizer, byte fallback).
    pub fn from_tokenizer_json(bytes: &[u8]) -> Result<Self> {
        let json: json::TokenizerJson = serde_json::from_slice(bytes)
            .map_err(|e| Error::Tokenizer(format!("invalid tokenizer.json: {e}")))?;
        if json.model.kind != "Unigram" {
            return Err(Error::Tokenizer(format!(
                "only unigram tokenizers are supported, got {}",
                json.model.kind
            )));
        }
        let special: HashMap<u32, ()> = json
            .added_tokens
            .iter()
            .filter(|t| t.special)
            .map(|t| (t.id, ()))
            .collect();
        let unk_id = json.model.unk_id.unwrap_or(0);
        let pieces = json
            .model
            .vocab
            .into_iter()
            .enumerate()
            .map(|(id, (text, score))| {
                let id = id as u32;
                let kind = if id == unk_id {
                    PieceKind::Unknown
                } else if special.contains_key(&id) {
                    PieceKind::Control
                } else if let Some(byte) = parse_byte_piece(&text) {
                    PieceKind::Byte(byte)
                } else {
                    PieceKind::Normal
                };
                Piece { text, score, kind }
            })
            .collect();
        let normalizer = Normalizer {
            add_dummy_prefix: true,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            unsupported: None,
        };
        Self::new(pieces, json.model.byte_fallback, normalizer)
    }

    fn new(pieces: Vec<Piece>, byte_fallback: bool, normalizer: Normalizer) -> Result<Self> {
        let mut lookup = HashMap::new();
        let mut byte_pieces = [u32::MAX; 256];
        let mut unk_id = None;
        let mut min_score = f32::INFINITY;
        for (id, piece) in pieces.iter().enumerate() {
            let id = id as u32;
            match piece.kind {
                PieceKind::Normal | PieceKind::UserDefined => {
                    lookup.insert(piece.text.clone(), id);
                    min_score = min_score.min(piece.score);
                }
                PieceKind::Byte(byte) => byte_pieces[byte as usize] = id,
                PieceKind::Unknown => unk_id = Some(id),
                PieceKind::Control | PieceKind::Unused => {}
            }
        }
        let unk_id = unk_id.ok_or_else(|| Error::Tokenizer("vocabulary has no <unk>".into()))?;
        let byte_pieces = if byte_fallback {
            if byte_pieces.contains(&u32::MAX) {
                return Err(Error::Tokenizer(
                    "byte fallback without all 256 byte pieces".into(),
                ));
            }
            Some(byte_pieces)
        } else {
            None
        };
        let max_piece_len = lookup.keys().map(|k| k.len()).max().unwrap_or(1);
        Ok(Self {
            pieces,
            lookup,
            max_piece_len,
            byte_pieces,
            unk_id,
            unk_score: min_score - UNKNOWN_PENALTY,
            normalizer,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Splits `text` into token ids.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if let Some(name) = &self.normalizer.unsupported {
            return Err(Error::Tokenizer(format!("unsupported normalizer {name:?}")));
        }
        let normalized = self.normalize(text);
        let mut ids = Vec::new();
        for (start, end, id) in self.segment(&normalized) {
            match (id, &self.byte_pieces) {
                (None, Some(bytes)) => {
                    ids.extend(normalized[start..end].bytes().map(|b| bytes[b as usize]));
                }
                (None, None) => ids.push(self.unk_id),
                (Some(id), _) => ids.push(id),
            }
        }
        Ok(ids)
    }

    /// Turns token ids back into text. Incomplete UTF-8 byte sequences become U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(piece) = self.pieces.get(id as usize) else {
                continue;
            };
            match piece.kind {
                PieceKind::Byte(byte) => bytes.push(byte),
                PieceKind::Normal | PieceKind::UserDefined => {
                    bytes.extend_from_slice(piece.text.as_bytes())
                }
                PieceKind::Unknown => bytes.extend_from_slice(" \u{2047} ".as_bytes()),
                PieceKind::Control | PieceKind::Unused => {}
            }
        }
        let text = String::from_utf8_lossy(&bytes).replace(WHITESPACE, " ");
        match text.strip_prefix(' ') {
            Some(stripped) if self.normalizer.add_dummy_prefix => stripped.to_string(),
            _ => text,
        }
    }

    fn normalize(&self, text: &str) -> String {
        let n = &self.normalizer;
        let mut text = if n.remove_extra_whitespaces {
            text.split(' ')
                .filter(|w| !w.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            text.to_string()
        };
        if n.add_dummy_prefix && !text.is_empty() {
            text.insert(0, ' ');
        }
        if n.escape_whitespaces {
            text = text.replace(' ', &WHITESPACE.to_string());
        }
        text
    }

    /// Viterbi segmentation into `(byte_start, byte_end, piece)`, `None` for unknown characters.
    fn segment(&self, text: &str) -> Vec<(usize, usize, Option<u32>)> {
        #[derive(Clone, Copy)]
        struct Best {
            score: f32,
            start: usize,
            id: Option<u32>,
        }
        let len = text.len();
        let mut best: Vec<Option<Best>> = vec![None; len + 1];
        best[0] = Some(Best {
            score: 0.0,
            start: 0,
            id: None,
        });

        for (start, ch) in text.char_indices() {
            let Some(here) = best[start] else { continue };
            let mut relax = |end: usize, score: f32, id: Option<u32>| {
                let candidate = Best {
                    score: here.score + score,
                    start,
                    id,
                };
                if best[end].is_none_or(|b| candidate.score > b.score) {
                    best[end] = Some(candidate);
                }
            };
            let mut has_single_char = false;
            let max_end = (start + self.max_piece_len).min(len);
            for end in (start + 1..=max_end).filter(|&e| text.is_char_boundary(e)) {
                if let Some(&id) = self.lookup.get(&text[start..end]) {
                    relax(end, self.pieces[id as usize].score, Some(id));
                    has_single_char |= end == start + ch.len_utf8();
                }
            }
            if !has_single_char {
                relax(start + ch.len_utf8(), self.unk_score, None);
            }
        }

        let mut segments = Vec::new();
        let mut end = len;
        while end > 0 {
            let b = best[end].expect("every character boundary is reachable");
            segments.push((b.start, end, b.id));
            end = b.start;
        }
        segments.reverse();
        segments
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

mod json {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct TokenizerJson {
        pub model: Model,
        #[serde(default)]
        pub added_tokens: Vec<AddedToken>,
    }

    #[derive(Deserialize)]
    pub struct Model {
        #[serde(rename = "type")]
        pub kind: String,
        #[serde(default)]
        pub vocab: Vec<(String, f32)>,
        pub unk_id: Option<u32>,
        #[serde(default)]
        pub byte_fallback: bool,
    }

    #[derive(Deserialize)]
    pub struct AddedToken {
        pub id: u32,
        #[serde(default)]
        pub special: bool,
    }
}

/// The subset of SentencePiece's `sentencepiece_model.proto` needed for inference.
mod proto {
    pub const MODEL_TYPE_UNIGRAM: u64 = 1;

    pub struct ModelProto {
        pub pieces: Vec<SentencePiece>,
        pub model_type: u64,
        pub byte_fallback: bool,
        pub normalizer: NormalizerSpec,
    }

    pub struct SentencePiece {
        pub piece: String,
        pub score: f32,
        pub kind: u64,
    }

    pub struct NormalizerSpec {
        pub name: String,
        pub precompiled_charsmap: Vec<u8>,
        pub add_dummy_prefix: bool,
        pub remove_extra_whitespaces: bool,
        pub escape_whitespaces: bool,
    }

    impl ModelProto {
        pub fn parse(bytes: &[u8]) -> Result<Self, String> {
            let mut model = ModelProto {
                pieces: Vec::new(),
                model_type: MODEL_TYPE_UNIGRAM,
                byte_fallback: false,
                normalizer: NormalizerSpec::parse(&[])?,
            };
            for field in Reader(bytes) {
                match field? {
                    (1, Value::Bytes(b)) => model.pieces.push(SentencePiece::parse(b)?),
                    (2, Value::Bytes(b)) => {
                        for field in Reader(b) {
                            match field? {
                                (3, Value::Varint(v)) => model.model_type = v,
                                (35, Value::Varint(v)) => model.byte_fallback = v != 0,
                                _ => {}
                            }
                        }
                    }
                    (3, Value::Bytes(b)) => model.normalizer = NormalizerSpec::parse(b)?,
                    _ => {}
                }
            }
            Ok(model)
        }
    }

    impl SentencePiece {
        fn parse(bytes: &[u8]) -> Result<Self, String> {
            let mut piece = SentencePiece {
                piece: String::new(),
                score: 0.0,
                kind: 1,
            };
            for field in Reader(bytes) {
                match field? {
                    (1, Value::Bytes(b)) => {
                        piece.piece = String::from_utf8(b.to_vec()).map_err(|e| e.to_string())?
                    }
                    (2, Value::Fixed32(v)) => piece.score = f32::from_bits(v),
                    (3, Value::Varint(v)) => piece.kind = v,
                    _ => {}
                }
            }
            Ok(piece)
        }
    }

    impl NormalizerSpec {
        fn parse(bytes: &[u8]) -> Result<Self, String> {
            let mut spec = NormalizerSpec {
                name: String::new(),
                precompiled_charsmap: Vec::new(),
                add_dummy_prefix: true,
                remove_extra_whitespaces: true,
                escape_whitespaces: true,
            };
            for field in Reader(bytes) {
                match field? {
                    (1, Value::Bytes(b)) => spec.name = String::from_utf8_lossy(b).into_owned(),
                    (2, Value::Bytes(b)) => spec.precompiled_charsmap = b.to_vec(),
                    (3, Value::Varint(v)) => spec.add_dummy_prefix = v != 0,
                    (4, Value::Varint(v)) => spec.remove_extra_whitespaces = v != 0,
                    (5, Value::Varint(v)) => spec.escape_whitespaces = v != 0,
                    _ => {}
                }
            }
            Ok(spec)
        }
    }

    enum Value<'a> {
        Varint(u64),
        Fixed64,
        Bytes(&'a [u8]),
        Fixed32(u32),
    }

    /// Iterates over the `(field number, value)` pairs of a protobuf message.
    struct Reader<'a>(&'a [u8]);

    impl<'a> Reader<'a> {
        fn varint(&mut self) -> Result<u64, String> {
            let mut value = 0u64;
            for shift in (0..64).step_by(7) {
                let (&byte, rest) = self.0.split_first().ok_or("truncated varint")?;
                self.0 = rest;
                value |= u64::from(byte & 0x7f) << shift;
                if byte & 0x80 == 0 {
                    return Ok(value);
                }
            }
            Err("varint too long".into())
        }

        fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
            if self.0.len() < n {
                return Err("truncated message".into());
            }
            let (head, rest) = self.0.split_at(n);
            self.0 = rest;
            Ok(head)
        }

        fn field(&mut self) -> Result<(u64, Value<'a>), String> {
            let key = self.varint()?;
            let value = match key & 7 {
                0 => Value::Varint(self.varint()?),
                1 => {
                    self.take(8)?;
                    Value::Fixed64
                }
                2 => {
                    let len = self.varint()? as usize;
                    Value::Bytes(self.take(len)?)
                }
                5 => Value::Fixed32(u32::from_le_bytes(self.take(4)?.try_into().unwrap())),
                wire_type => return Err(format!("unsupported wire type {wire_type}")),
            };
            Ok((key >> 3, value))
        }
    }

    impl<'a> Iterator for Reader<'a> {
        type Item = Result<(u64, Value<'a>), String>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.0.is_empty() {
                return None;
            }
            let field = self.field();
            if field.is_err() {
                self.0 = &[];
            }
            Some(field)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> Tokenizer {
        let mut pieces = vec![
            Piece {
                text: "<unk>".into(),
                score: 0.0,
                kind: PieceKind::Unknown,
            },
            Piece {
                text: "<s>".into(),
                score: 0.0,
                kind: PieceKind::Control,
            },
        ];
        for (text, score) in [("▁he", -1.0), ("llo", -1.0), ("▁hello", -1.5), ("l", -3.0)] {
            pieces.push(Piece {
                text: text.into(),
                score,
                kind: PieceKind::Normal,
            });
        }
        for (text, score) in [("o", -3.0), ("▁", -2.0), ("h", -3.0), ("e", -3.0)] {
            pieces.push(Piece {
                text: text.into(),
                score,
                kind: PieceKind::Normal,
            });
        }
        pieces.extend((0..=255u8).map(|b| Piece {
            text: format!("<0x{b:02X}>"),
            score: 0.0,
            kind: PieceKind::Byte(b),
        }));
        let normalizer = Normalizer {
            add_dummy_prefix: true,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            unsupported: None,
        };
        Tokenizer::new(pieces, true, normalizer).unwrap()
    }

    #[test]
    fn viterbi_prefers_best_total_score() {
        let tok = toy();
        // "▁hello" (-1.5) beats "▁he" + "llo" (-2.0).
        assert_eq!(tok.encode("hello").unwrap(), vec![4]);
        assert_eq!(tok.decode(&[4]), "hello");
    }

    #[test]
    fn unknown_characters_fall_back_to_bytes() {
        let tok = toy();
        let ids = tok.encode("hé").unwrap();
        // "▁" "h" then the two UTF-8 bytes of "é".
        let byte = |b: u8| 10 + b as u32;
        assert_eq!(ids, vec![7, 8, byte(0xc3), byte(0xa9)]);
        assert_eq!(tok.decode(&ids), "hé");
        assert_eq!(tok.decode(&ids[..3]), "h\u{fffd}");
    }
}
