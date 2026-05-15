CUDA_VISIBLE_DEVICES=2 python scripts/dino_v2_segvlad_CVGL.py \
    --prog.cache-dir .cache \
    --exp-id cvgl_segvlad_run \
    --task-mode cvgl \
    --cvgl-dataset-root data/Hawaii_wildfire_2023_1_336_intile \
    --data-split test \
    --model-type dinov2_vitg14 \
    --num-clusters 32 \
    --desc-layer 23 \
    --desc-facet key \
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
    --offset-prediction-method sim_map \
    --no-use-local-rerank \
    # --local-match-method sim_map \
    # --pretrained_vlad_centers .cache/vocabulary/VPAir/c_centers.pt \
    