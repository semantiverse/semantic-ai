"""Reference Pocket TTS generation at temperature 0 (deterministic), as raw little-endian f32."""
import sys, torch, numpy as np
from pocket_tts import TTSModel
lang, text, out = sys.argv[1], sys.argv[2], sys.argv[3]
torch.manual_seed(0)
m = TTSModel.load_model(language=lang, temp=0.0)
state = m.get_state_for_audio_prompt("alba")
audio = m.generate_audio(state, text)
audio.numpy().astype("<f4").tofile(out)
print(lang, audio.shape)
