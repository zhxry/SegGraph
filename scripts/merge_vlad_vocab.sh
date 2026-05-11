python scripts/merge_vlad_vocabularies.py \
    .cache/cvgl_descs/California_wildfire_2025_1_336_intile-d79c7233cd21/dinov2_vitg14-key-L23-segvlad/dense/tiles \
    .cache/cvgl_descs/California_wildfire_2025_2_336_intile-7288fff21038/dinov2_vitg14-key-L23-segvlad/dense/tiles \
    .cache/cvgl_descs/Moore_tornado_2013_1_336_intile-3a575b04b209/dinov2_vitg14-key-L23-segvlad/dense/tiles \
    .cache/cvgl_descs/Turkey_earthquake_2023_1_336_intile-cf4967024ee8/dinov2_vitg14-key-L23-segvlad/dense/tiles \
    .cache/cvgl_descs/Turkey_earthquake_2023_2_336_intile-c4b34e774204/dinov2_vitg14-key-L23-segvlad/dense/tiles \
    --num-clusters 32 \
    --output .cache/vocabulary/merged/dinov2_vitg14-key-L23-segvlad/C1C2M1T1T2