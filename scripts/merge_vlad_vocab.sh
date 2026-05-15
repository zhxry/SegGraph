python scripts/merge_vlad_vocabularies.py \
    --num-clusters 32 \
    .cache/cvgl_descs/Moore_tornado_2013_1_336_intile-3a575b04b209/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    .cache/cvgl_descs/Moore_tornado_2013_1_336_intile-3a575b04b209/dinov2_vitg14-value-L31-segvlad/dense/tile_post \
    --output .cache/vocabulary/merged/dinov2_vitg14-value-L31-segvlad/moore1_prepost
    # --single-scene-sweep
    # .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-value-L31-segvlad/dense/tile_post \
    # .cache/cvgl_descs/Acapulco_tornado_2023_1_336_intile-9b05a025b74a/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Acapulco_tornado_2023_2_336_intile-c57f40af8490/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/California_wildfire_2025_1_336_intile-d79c7233cd21/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Moore_tornado_2013_1_336_intile-3a575b04b209/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Paradise_wildfire_2018_1_336_intile-a0508d9df5f6/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Paradise_wildfire_2018_2_336_intile-0ed5edea3a5b/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Turkey_earthquake_2023_1_336_intile-cf4967024ee8/dinov2_vitg14-value-L31-segvlad/dense/tiles \
    # .cache/cvgl_descs/Turkey_earthquake_2023_2_336_intile-c4b34e774204/dinov2_vitg14-value-L31-segvlad/dense/tiles \