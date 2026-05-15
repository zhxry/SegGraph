# python scripts/viz_dino_vlad_clusters.py \
#     --dataset-root data/Hawaii_wildfire_2023_1_336_intile \
#     --c-centers .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-value-L23-segvlad/vlad-C32/c_centers.pt \
#     --output-dir .cache/viz_clusters/Hawaii_1_cur_value23 \
#     --desc-layer 23 \
#     --desc-facet value \
#     --device cuda
    # --c-centers .cache/cvgl_descs/Hawaii_wildfire_2023_1_336_intile-2db138713ce6/dinov2_vitg14-key-L31-segvlad/vlad-C32/c_centers.pt \


python scripts/viz_dino_vlad_clusters.py \
    --image data/Hawaii_wildfire_2023_1_336_intile/tiles/tile_r00_c06.png \
    --c-centers /data/zhanghaofei/xry/Revisit-Anything/cache/vocabulary/dinov2_vitg14/l31_value_c32/VPAir/c_centers.pt \
    --output-dir viz/vlad_vocab \
    --no-headers \
    --device cuda