CUDA_VISIBLE_DEVICES=1 python scripts/dino_v2_segvlad_CVGL.py \
    --prog.cache-dir .cache \
    --exp-id cvgl_segvlad_run \
    --task-mode cvgl \
    --cvgl-dataset-root data/Hawaii_wildfire_2023_1_336_intile \
    --data-split test \
    --model-type dinov2_vitg14 \
    --num-clusters 32 \
    --desc-layer 23 \
    --desc-facet key \
    --segvlad-neighbor-order 3 \
    --seg-cfg.neighbor-method delaunay \
    --segvlad-centroid-pe-num-freqs 4 \
    --segvlad-centroid-pe-weight 0.2 \
    --coarse-top-k 10 \
    --tile-size-px 512 \
    --use-local-rerank \
    --local-match-method sim_map \
    --use-offset-head \
    --offset-prediction-method sim_map
    # --pretrained_vlad_centers .cache/vocabulary/VPAir/c_centers.pt \
    