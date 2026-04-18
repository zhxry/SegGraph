python scripts/prepare_cvgl_dataset.py \
    --satellite-image-path data/California-wildfire-2025-2/pre_center_16x16_4096.jpg \
    --satellite-size-hw 4096x4096 \
    --uav-image-path data/California-wildfire-2025-2/after_center_16x16_4096.jpg \
    --uav-size-hw 4096x4096 \
    --output-root data/California_wildfire_2025_2_336_intile \
    --tile-size 512 \
    --query-size 336 \
    --query-count 1000 \
    --overwrite \
    --ensure-query-within-single-tile
    # --rotate