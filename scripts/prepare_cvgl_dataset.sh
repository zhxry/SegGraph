python scripts/prepare_cvgl_dataset.py \
    --satellite-image-path data/Paradise-wildfire-2018-1/pre_center_16x16_4096.jpg \
    --satellite-size-hw 4096x4096 \
    --uav-image-path data/Paradise-wildfire-2018-1/after_center_16x16_4096.jpg \
    --uav-size-hw 4096x4096 \
    --output-root data/Paradise_wildfire_2018_1_336_intile \
    --tile-size 512 \
    --query-size 336 \
    --query-count 1000 \
    --overwrite \
    --post-only \
    --ensure-query-within-single-tile
    # --rotate