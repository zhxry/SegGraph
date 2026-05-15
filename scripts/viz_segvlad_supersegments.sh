python scripts/viz_segvlad_supersegments.py \
    --results-json .cache/experiments/cvgl_segvlad_run/hawaii1_k23_segvlad_o3.json \
    --output viz/supersegment_viz.png \
    --top-segments 5 \
    --tile-index 8 \
    --orders 0 1 2 3 \
    --no-headers \
    --device cpu
    # --no-pdf \
    # --all-tiles \