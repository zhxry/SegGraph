CUDA_VISIBLE_DEVICES=0 python scripts/dino_v2_segvlad_CVGL.py \
    --prog.cache-dir .cache \
    --exp-id cvgl_segvlad_run \
    --task-mode cvgl \
    --cvgl-dataset-root data/University-1652 \
    --data-split test \
    --model-type dinov2_vitg14 \
    --num-clusters 32 \
    --desc-layer 23 \
    --desc-facet key \
    --segvlad-neighbor-order 3 \
    --seg-cfg.neighbor-method delaunay \
    --segvlad-centroid-pe-num-freqs 4 \
    --segvlad-centroid-pe-weight 0.1 \
    --coarse-top-k 10 \
    --tile-size-px 512 \
    --no-use-offset-head \
    --no-use-local-rerank \
    # --local-match-method sim_map \
    # --pretrained_vlad_centers .cache/vocabulary/VPAir/c_centers.pt \
    