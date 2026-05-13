CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python scripts/extract_dinov2_dense_features.py \
    --input-dir "${INPUT_DIR:-data/Hawaii_wildfire_2023_1_336_intile/tile_post}" \
    --output-dir "${OUTPUT_DIR:-.cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-value-L31-segvlad/dense/tile_post}" \
    --model-type "${MODEL_TYPE:-dinov2_vitg14}" \
    --desc-layer "${DESC_LAYER:-31}" \
    --desc-facet "${DESC_FACET:-value}" \
    --patch-stride "${PATCH_STRIDE:-14}" \
    "$@"
