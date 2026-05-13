CUDA_VISIBLE_DEVICES=2 python scripts/dino_v2_segvlad_CVGL.py \
    --prog.cache-dir .cache \
    --exp-id cvgl_segvlad_v31_run \
    --task-mode cvgl \
    --cvgl-dataset-root data/California_wildfire_2025_2_336_intile \
    --pretrained_vlad_centers .cache/vocabulary/merged/dinov2_vitg14-value-L31-segvlad/california2_fewshot/tiles-004_vlad-C32/c_centers.pt \
    --data-split test \
    --model-type dinov2_vitg14 \
    --num-clusters 32 \
    --desc-layer 31 \
    --desc-facet value \
    --coarse-device cuda \
    --segment-descriptor segvlad \
    --segvlad-neighbor-order 3 \
    --seg-cfg.neighbor-method delaunay \
    --segvlad-centroid-pe-num-freqs 4 \
    --segvlad-centroid-pe-weight 0.1 \
    --segvlad-relative-context-num_freqs 0 \
    --segvlad-relative-context-weight 0.0 \
    --segvlad-relative-context-order 3 \
    --segvlad-relative-context-ref-grid-size 36 \
    --coarse-top-k 10 \
    --tile-size-px 512 \
    --use-offset-head \
    --offset-prediction-method slide_ncc \
    --no-use-local-rerank \
    # --local-match-method sim_map \
    # --pretrained_vlad_centers .cache/vocabulary/VPAir/c_centers.pt \
    