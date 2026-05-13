DATASET_ROOT=${DATASET_ROOT:-data/Hawaii_wildfire_2023_1_336_intile}
CACHE_DIR=${CACHE_DIR:-.cache/backbone_ablation}
EXP_ID=${EXP_ID:-clip_cls}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python scripts/backbone_ablation_CVGL.py \
    --prog.cache-dir "${CACHE_DIR}" \
    --cvgl-dataset-root "${DATASET_ROOT}" \
    --backbone clip \
    --model-name ViT-B/16 \
    --desc-layer -1 \
    --desc-facet token \
    --agg-method vlad \
    --global-agg cls \
    --tile-size-px 512 \
    --offset-prediction-method slide_ncc \
    --exp-id "${EXP_ID}"
