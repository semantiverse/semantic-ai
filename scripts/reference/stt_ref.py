"""Reference Kyutai STT run: writes the 24 kHz input and the text tokens per step."""
import itertools, json, math, sys
import julius, sphn, torch, moshi.models
repo, in_file, out_prefix = sys.argv[1:4]
info = moshi.models.loaders.CheckpointInfo.from_hf_repo(repo)
mimi = info.get_mimi(device="cpu")
tok = info.get_text_tokenizer()
lm = info.get_moshi(device="cpu", dtype=torch.float32)
gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)
audio, sr = sphn.read(in_file)
audio = torch.from_numpy(audio)[:1]
audio = julius.resample_frac(audio, sr, mimi.sample_rate)
audio.numpy().astype("<f4").tofile(out_prefix + ".f32")
if audio.shape[-1] % mimi.frame_size:
    audio = torch.nn.functional.pad(audio, (0, mimi.frame_size - audio.shape[-1] % mimi.frame_size))
pre = math.ceil(info.stt_config.get("audio_silence_prefix_seconds", 1.0) * mimi.frame_rate)
suf = math.ceil(info.stt_config.get("audio_delay_seconds", 5.0) * mimi.frame_rate)
silence = torch.zeros((1, 1, mimi.frame_size))
chunks = itertools.chain(itertools.repeat(silence, pre), torch.split(audio[:, None], mimi.frame_size, dim=-1), itertools.repeat(silence, suf))
tokens = []
with torch.no_grad(), mimi.streaming(1), gen.streaming(1):
    for c in chunks:
        t = gen.step(mimi.encode(c))
        tokens.append(int(t[0, 0, 0]))
text = "".join(tok.id_to_piece(t) for t in tokens if t not in (0, 3)).replace("▁", " ").strip()
json.dump({"prefix_frames": pre, "tokens": tokens, "text": text}, open(out_prefix + ".json", "w"))
print(text)
