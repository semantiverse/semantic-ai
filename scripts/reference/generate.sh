#!/usr/bin/env bash
# Generates the fixtures of crates/semantic-voice-local/tests/parity.rs with the upstream Python
# implementations, then runs the parity tests against them:
#
#   scripts/reference/generate.sh /tmp/parity-fixtures
#   PARITY_FIXTURES=/tmp/parity-fixtures cargo test --release --test parity -- --ignored
#
# Needs `uv`. Downloads PyTorch (CPU), the upstream packages and the models.
set -euo pipefail

out=$(realpath -m "${1:?usage: $0 <output directory>}")
here=$(cd "$(dirname "$0")" && pwd)
# Upstream revisions the port was checked against.
pocket_tts_rev=306918e29901a6b1e22e832e2ab050ca3ab44bb3
dsm_rev=4c4f65e147df056adf3346290d64c7b9649b18c9

mkdir -p "$out"
cd "$out"
if [ ! -d .venv ]; then
    uv venv -q -p 3.12 .venv
    VIRTUAL_ENV=$PWD/.venv uv pip install -q --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple torch
    VIRTUAL_ENV=$PWD/.venv uv pip install -q protobuf moshi==0.2.13 julius sphn \
        "pocket-tts @ git+https://github.com/kyutai-labs/pocket-tts@$pocket_tts_rev"
fi
python=".venv/bin/python -I"
text="Hello world. This is a test of the Rust port."

for model in english english_2m_2026-10; do
    $python "$here/tts_ref.py" "$model" "$text" "tts_$model.f32"
done
$python "$here/text_ref.py" english english_2m_2026-10 french

curl -sSfL -o bria.mp3 \
    "https://github.com/kyutai-labs/delayed-streams-modeling/raw/$dsm_rev/audio/bria.mp3"
$python "$here/stt_ref.py" kyutai/stt-1b-en_fr-candle bria.mp3 stt_bria
