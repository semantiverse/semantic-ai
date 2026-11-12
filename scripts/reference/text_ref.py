"""Reference tokenization and text chunking of Pocket TTS."""
import json, sys
from pocket_tts.utils.config import load_config, CONFIGS_DIR
from pocket_tts.modules.text_conditioner import build_tokenizer
from pocket_tts.models.text_chunking import split_into_best_sentences
from pocket_tts.utils.utils import download_if_necessary

TEXTS = [
    "Hello world.",
    "  leading spaces and   double  spaces ",
    "Dr. Miller paid $822.95, i.e. a lot; then he left at 4 p.m. for www.kyutai.org.",
    "“Stop!” she said. (Quietly.) Then: 'why?' — nobody knew…",
    "Café, naïve, Zürich, 東京, emoji 🙂 and math 3.14 vs 3,14.",
    "I counted to 3. 14 people were there. Mr. Smith arrived late",
    "This is a very long sentence without any full stop, which keeps going on and on, with commas, semicolons; and colons: so that the splitter has to fall back to clause boundaries, because otherwise it would be far too long for a single chunk of the model, which only handles about fifty tokens at a time, as you know",
    "Short. Another short one! And a question? Then some more text that follows here.",
]
out = {}
for model in sys.argv[1:]:
    cfg = load_config(CONFIGS_DIR / f"{model}.yaml")
    lut = cfg.flow_lm.lookup_table
    tok = build_tokenizer(lut.n_bins, lut.tokenizer_path, lut.tokenizer)
    out[model] = {
        "tokenizer": str(download_if_necessary(lut.tokenizer_path)),
        "cases": [{
            "text": t,
            "ids": tok.encode(t),
            "decoded": tok.decode(tok.encode(t)),
            "chunks": split_into_best_sentences(tok, t, 50, cfg.pad_with_spaces_for_short_inputs, cfg.remove_semicolons,
                cfg.append_terminal_punctuation, cfg.capitalize_first_letter, cfg.replace_characters),
        } for t in TEXTS],
    }
json.dump(out, open("text.json", "w"), ensure_ascii=False, indent=1)
print("ok", {m: len(v["cases"]) for m, v in out.items()})
