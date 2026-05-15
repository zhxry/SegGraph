python scripts/viz_topk_retrieval.py \
    --labeled-result wildfire=.cache/experiments/cvgl_segvlad_run/hawaii1_k23_segvlad_pe_o3.json \
    --labeled-result tornado=.cache/experiments/cvgl_segvlad_run/moore1_k23_segvlad.json \
    --labeled-result earthquake=.cache/experiments/cvgl_segvlad_run/turkey2_k23_segvlad.json \
    --dpi 300 \
    --output-scale 1.5 \
    --no-headers \
    --no-row-labels \
    --output viz/top-k_viz.png