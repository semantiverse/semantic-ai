//! Text normalization and splitting into the chunks the model generates one at a time, ported
//! from upstream `pocket_tts/models/text_chunking.py`.
//!
//! The model is trained on single sentences, so long inputs are split on sentence boundaries
//! and regrouped into chunks that fit a token budget.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::tokenizer::Tokenizer;
use crate::{Error, Result};

/// The text rules of a model config.
#[derive(Debug, Clone)]
pub struct TextRules {
    pub pad_with_spaces_for_short_inputs: bool,
    pub remove_semicolons: bool,
    pub append_terminal_punctuation: bool,
    pub capitalize_first_letter: bool,
    pub replace_characters: BTreeMap<String, String>,
}

/// A prepared prompt and how many frames to generate after the end of speech is detected.
pub struct Prompt {
    pub text: String,
    pub frames_after_eos: usize,
}

static SENTENCE_END_BEFORE_PAUSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([.!?\x{2026}])\s*[,;:]").unwrap());
/// A segment ending on a decimal number, as in "a fee of $822.95," or "de 822,95 €,".
static ENDS_WITH_DECIMAL_NUMBER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\d[.,]\d+\W*$").unwrap());

const TERMINAL_PUNCTUATION: &str = ".!?\u{2026}";
const WEAK_PUNCTUATION: &str = ",;:-\u{2013}\u{2014}";
const CLOSERS: &str = "\"'\u{201d}\u{2019})]\u{bb}";
/// Abbreviations that precede a name or an example, never end a sentence.
const TITLES: &[&str] = &[
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "vs.", "e.g.", "i.e.",
];
/// A closing quote can span several byte tokens; bound the lookahead.
const MAX_CLOSER_TOKENS: usize = 8;

impl TextRules {
    /// Normalizes one chunk of text for the model.
    pub fn prepare(&self, text: &str) -> Result<Prompt> {
        let mut text = text.trim().to_string();
        if !self.replace_characters.is_empty() {
            let replaced: String = text
                .chars()
                .map(|c| {
                    let mut buf = [0u8; 4];
                    match self.replace_characters.get(c.encode_utf8(&mut buf) as &str) {
                        Some(replacement) => replacement.clone(),
                        None => c.to_string(),
                    }
                })
                .collect();
            text = replaced.split_whitespace().collect::<Vec<_>>().join(" ");
            // Deleted quotes leave '"Hi?", she said' as 'Hi?, she said': keep the sentence mark.
            text = SENTENCE_END_BEFORE_PAUSE
                .replace_all(&text, "$1")
                .into_owned();
        }
        if text.is_empty() {
            return Err(Error::invalid_input("text to synthesize is empty"));
        }
        text = text.replace(['\n', '\r'], " ").replace("  ", " ");
        if self.remove_semicolons {
            text = text.replace(';', ",");
        }
        let frames_after_eos = if text.split_whitespace().count() <= 4 {
            3
        } else {
            1
        };

        if self.capitalize_first_letter {
            let mut chars = text.chars();
            if let Some(first) = chars.next()
                && !first.is_uppercase()
            {
                text = first.to_uppercase().chain(chars).collect();
            }
        }
        if self.append_terminal_punctuation {
            text = ensure_terminal_punctuation(&text);
        }
        // The model does poorly on very few tokens: leading spaces add some.
        if self.pad_with_spaces_for_short_inputs && text.split_whitespace().count() < 5 {
            text.insert_str(0, &" ".repeat(8));
        }
        Ok(Prompt {
            text,
            frames_after_eos,
        })
    }

    /// Splits `text` into chunks of at most about `max_tokens` tokens, on sentence boundaries,
    /// falling back to commas, semicolons and colons for overlong sentences.
    pub fn split_into_chunks(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        max_tokens: usize,
    ) -> Result<Vec<String>> {
        let text = self.prepare(text)?.text;
        let tokens = tokenizer.encode(text.trim())?;

        let sentence_ends = boundary_tokens(tokenizer, ".!...?")?;
        let boundaries = find_boundaries(&tokens, &sentence_ends, tokenizer);
        let sentences = segments(&tokens, &boundaries, tokenizer);

        let clause_ends = boundary_tokens(tokenizer, ",;:")?;
        let mut refined = Vec::new();
        for (n_tokens, sentence) in sentences {
            if n_tokens <= max_tokens {
                refined.push((n_tokens, sentence));
                continue;
            }
            let sub_tokens = tokenizer.encode(sentence.trim())?;
            let sub_boundaries = find_boundaries(&sub_tokens, &clause_ends, tokenizer);
            let sub_segments = segments(&sub_tokens, &sub_boundaries, tokenizer);
            if sub_segments.len() > 1 {
                refined.extend(sub_segments);
            } else {
                refined.push((n_tokens, sentence));
            }
        }

        // Never end a chunk on a decimal number: the model would take the decimal point for
        // the end of the sentence. Such a segment stays with the next one.
        let mut glued: Vec<(usize, String)> = Vec::new();
        for (n_tokens, segment) in refined {
            match glued.last_mut() {
                Some((prev_tokens, prev)) if ENDS_WITH_DECIMAL_NUMBER.is_match(prev) => {
                    *prev_tokens += n_tokens;
                    *prev = format!("{} {}", prev.trim_end(), segment.trim());
                }
                _ => glued.push((n_tokens, segment)),
            }
        }

        let mut chunks = Vec::new();
        let mut current = String::new();
        let mut current_tokens = 0;
        for (n_tokens, segment) in glued {
            if current.is_empty() {
                current = segment;
                current_tokens = n_tokens;
            } else if current_tokens + n_tokens > max_tokens {
                chunks.push(current.trim().to_string());
                current = segment;
                current_tokens = n_tokens;
            } else {
                current.push(' ');
                current.push_str(&segment);
                current_tokens += n_tokens;
            }
        }
        if !current.is_empty() {
            chunks.push(current.trim().to_string());
        }
        Ok(chunks)
    }
}

fn ensure_terminal_punctuation(text: &str) -> String {
    let core = text.trim_end_matches(|c| CLOSERS.contains(c) || c == ' ');
    let closers = text[core.len()..].trim();
    match core.chars().last() {
        None => text.to_string(),
        Some(c) if TERMINAL_PUNCTUATION.contains(c) => text.to_string(),
        Some(c) if WEAK_PUNCTUATION.contains(c) => {
            let core = core.trim_end_matches(|c| WEAK_PUNCTUATION.contains(c) || c == ' ');
            format!("{core}.{closers}")
        }
        Some(_) => format!("{text}."),
    }
}

/// The tokens of the punctuation in `marks`, without the leading word-start token.
fn boundary_tokens(tokenizer: &Tokenizer, marks: &str) -> Result<Vec<u32>> {
    Ok(tokenizer.encode(marks)?.into_iter().skip(1).collect())
}

/// Token indices delimiting segments: starts with 0, ends with `tokens.len()`.
fn find_boundaries(tokens: &[u32], boundary: &[u32], tokenizer: &Tokenizer) -> Vec<usize> {
    let mut indices = vec![0];
    let mut previous_was_boundary = false;
    for (idx, token) in tokens.iter().enumerate() {
        if boundary.contains(token) {
            previous_was_boundary = true;
            continue;
        }
        if previous_was_boundary {
            previous_was_boundary = false;
            if is_inner_punctuation(tokens, idx, tokenizer) {
                continue;
            }
            let start = skip_closers(tokens, idx, tokenizer);
            if start < tokens.len() {
                indices.push(start);
            }
        }
    }
    indices.push(tokens.len());
    indices
}

/// Whether the punctuation before `start` does not end a sentence: it is directly followed by
/// a letter or digit ("3.14", "www.kyutai.org"), or ends a title before a name ("Dr. Miller").
fn is_inner_punctuation(tokens: &[u32], start: usize, tokenizer: &Tokenizer) -> bool {
    let after = tokenizer.decode(&tokens[start - 1..(start + 1).min(tokens.len())]);
    if after.chars().nth(1).is_some_and(char::is_alphanumeric) {
        return true;
    }
    let window_start = start.saturating_sub(6);
    let window = tokenizer.decode(&tokens[window_start..start]);
    let words: Vec<&str> = window.split_whitespace().collect();
    // The first word of the window may be cut, unless the window starts the text.
    words.last().is_some_and(|w| TITLES.contains(w)) && (words.len() > 1 || window_start == 0)
}

/// Moves a segment start past closing quotes and brackets that directly follow the
/// punctuation, so that they stay with their sentence. Closers made of several byte tokens
/// decode to U+FFFD until complete.
fn skip_closers(tokens: &[u32], start: usize, tokenizer: &Tokenizer) -> usize {
    let punctuation = tokenizer.decode(&tokens[start - 1..start]);
    let mut new_start = start;
    let stop = tokens.len().min(start + MAX_CLOSER_TOKENS);
    for end in start + 1..=stop {
        let window = tokenizer.decode(&tokens[start - 1..end]);
        let after = window.get(punctuation.len()..).unwrap_or("");
        if after.is_empty()
            || after
                .chars()
                .any(|c| !CLOSERS.contains(c) && c != '\u{fffd}')
        {
            break;
        }
        if !after.contains('\u{fffd}') {
            new_start = end;
        }
    }
    new_start
}

fn segments(tokens: &[u32], boundaries: &[usize], tokenizer: &Tokenizer) -> Vec<(usize, String)> {
    boundaries
        .windows(2)
        .map(|w| (w[1] - w[0], tokenizer.decode(&tokens[w[0]..w[1]])))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rules() -> TextRules {
        TextRules {
            pad_with_spaces_for_short_inputs: false,
            remove_semicolons: false,
            append_terminal_punctuation: true,
            capitalize_first_letter: true,
            replace_characters: BTreeMap::new(),
        }
    }

    #[test]
    fn prepare_capitalizes_and_terminates() {
        let prompt = rules().prepare("  hello world  ").unwrap();
        assert_eq!(prompt.text, "Hello world.");
        assert_eq!(prompt.frames_after_eos, 3);
        assert_eq!(rules().prepare("is it?").unwrap().text, "Is it?");
        assert_eq!(
            rules().prepare("so, \"wait\",").unwrap().text,
            "So, \"wait\"."
        );
        assert_eq!(rules().prepare("(quietly)").unwrap().text, "(quietly).");
        assert!(rules().prepare("   ").is_err());
    }

    #[test]
    fn prepare_replaces_characters() {
        let mut rules = rules();
        rules.replace_characters.insert("\"".into(), "".into());
        assert_eq!(
            rules.prepare("\"Hi?\", she said").unwrap().text,
            "Hi? she said."
        );
    }
}
