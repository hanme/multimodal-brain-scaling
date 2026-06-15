# Shared helper — resolve whisper-family delta_T feature dirs + per-model layer config.
# Set MODEL_ID before sourcing (defaults to whisper-base). Features resolve to OUR local copy
# if present, else fall back to Sophie's read-only tree (tiny/small/medium are there; base is
# local; large is not extracted yet). Neural EEG h5s are model-independent (one set for all).
#
# Usage in a script:  MODEL_ID="${MODEL_ID:-whisper-base}"; source scripts/_whisper_features.sh
#                     D1=$(resolve_feat "")        # in-domain Broderick
#                     D2=$(resolve_feat "-surprisal")
#                     D3=$(resolve_feat "-d3")     # may be absent -> build from D1 u D2

MODEL_ID="${MODEL_ID:-whisper-base}"
SOPHIE_FEAT="/work/upschrimpf1/sigfstea/multimodal-brain-scaling/outputs/features"
LAYERS="configs/extraction/audio/${MODEL_ID//-/_}_layers.json"
if [ ! -f "$LAYERS" ]; then
  echo "ERROR: layer config not found: $LAYERS (supported: whisper-tiny/base/small/medium/large)"
  exit 1
fi

# echo a merged feature dir containing .h5 files for ${MODEL_ID}-delta-t<suffix>, or return 1.
resolve_feat () {
  local sub="${MODEL_ID}-delta-t${1}/merged" base d
  for base in "outputs/features" "$SOPHIE_FEAT"; do
    d="$base/$sub"
    if [ -d "$d" ] && ls "$d"/*.h5 >/dev/null 2>&1; then echo "$d"; return 0; fi
  done
  return 1
}
