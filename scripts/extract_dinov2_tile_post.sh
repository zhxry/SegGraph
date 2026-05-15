CUDA_VISIBLE_DEVICES=3 python scripts/extract_dinov2_dense_features.py \
    --input-dir data/Moore_tornado_2013_1_336_intile/tile_post \
    --output-dir .cache/cvgl_descs/Moore_tornado_2013_1_336_intile-3a575b04b209/dinov2_vitg14-value-L31-segvlad/dense/tile_post \
    --model-type dinov2_vitg14 \
    --desc-layer 31 \
    --desc-facet value \
    --patch-stride 14