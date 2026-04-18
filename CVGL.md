# DINOv2-VLAD-CVGL Pipeline Notes

本文档总结当前仓库里 `scripts/dino_v2_vlad_CVGL.py` 所实现的 DINOv2-VLAD-CVGL baseline，目标是让后续修改者在改动前先理解当前方法到底在做什么、哪些参数真正生效、以及数据和缓存是如何流动的。

相关实现入口：

- 主脚本：`scripts/dino_v2_vlad_CVGL.py`
- 特征提取 / 检索 / 重排 / offset 预测：`cvgl_retrieval.py`
- 评估：`cvgl_metrics.py`
- CVGL 数据集接口：`custom_datasets/cvgl_dataset.py`
- 数据集准备脚本：`scripts/prepare_cvgl_dataset.py`
- DINOv2 提特征与 VLAD 实现：`utilities.py`

## 1. 整体流程

当前 pipeline 是一个两阶段检索加一个基于相似度图的 offset 预测流程：

1. 读取数据集。
2. 用 DINOv2 从每张 tile / query 图像提取 dense patch descriptors。
3. 将每张图像的 dense descriptors 聚合为一个 global descriptor。
   - 默认用 VLAD。
   - 也支持 GeM，但当前 `scripts/dino_v2_vlad_CVGL.sh` 使用的是 VLAD。
4. 用 global descriptor 在所有 satellite tiles 上做 coarse retrieval。
5. 对 coarse top-k 候选用 local descriptors 做 reranking。
6. 对 rerank 后 top-1 tile 计算 similarity map，并据此预测 query 在 tile 内的 offset。
7. 分别评估 coarse retrieval、rerank 后 retrieval、offset 误差和最终 localization 误差。

主脚本头部注释写得比较准确：当前实现不是训练式模型，而是一个 AnyLoc 风格的特征提取 + 聚合 + 检索 + 基于 similarity map 的几何近似定位 baseline。

## 2. 入口脚本做了什么

入口在 `scripts/dino_v2_vlad_CVGL.py` 的 `main(largs)`。

它的执行顺序是：

1. 固定随机种子 `seed_everything(42)`。
2. 根据 `task_mode` 加载数据集。
3. 构造 dense feature cache 和 VLAD cache 的目录。
4. 初始化 `DinoV2ExtractFeatures`。
5. 选出 database 和 query 的索引子集。
6. 如果 `global_agg == "vlad"`，先在 database 上拟合 VLAD vocabulary。
7. 为 database 和 query 分别建立 `FeatureBank`。
8. 可选地对 global descriptors 做 PCA 降维。
9. 用 global descriptors 做 coarse retrieval。
10. 计算 coarse retrieval 指标。
11. 如果启用 local rerank，则对 coarse top-k 做局部重排，并计算 refined retrieval 指标。
12. 如果启用 offset head 且方法为 `sim_map`，则对最终 top-1 预测 offset，并计算 offset / localization 指标。
13. 汇总结果，可选画 recall 曲线、写 wandb、保存 JSON 结果。

这里需要注意两点：

- 虽然脚本里有 `offset_head`、`offset_loss_type` 之类名字，但当前实现并没有 learned head，也没有训练过程。
- 当前“offset head”本质上只是 similarity-map based offset estimator 的开关。

## 3. 参数与当前默认行为

`LocalArgs` 定义了整个 baseline 的行为。对当前 pipeline 最关键的是以下参数。

### 3.1 数据相关

- `task_mode`
  - `cvgl`：读取通用 UAV-satellite JSON 数据集。
  - `vpr`：把 AnyLoc / BaseDataset 风格数据集包一层 adapter，只做 retrieval，不做 offset/localization。
- `data_split`
  - `train` / `val` / `test`。
- `cvgl_dataset_root`
  - `cvgl` 模式必填。
- `cvgl_tiles_manifest` / `cvgl_queries_manifest`
  - 可显式指定 manifest，否则自动找 `tiles_<split>.json` / `queries_<split>.json`，再退回 `tiles.json` / `queries.json`。

### 3.2 DINOv2 descriptor 相关

- `model_type`
  - 可选 `dinov2_vits14` / `vitb14` / `vitl14` / `vitg14`。
- `desc_layer`
  - 提取哪一层 block 的特征。
- `desc_facet`
  - `query` / `key` / `value` / `token`。
  - 当前示例脚本用的是 `key`。

### 3.3 Global aggregation 相关

- `global_agg`
  - `vlad` 或 `gem`。
- `num_clusters`
  - VLAD 聚类中心数。
- `vlad_assignment`
  - `hard` 或 `soft`。
- `vlad_soft_temp`
  - soft assignment 的 softmax 温度。
- `pca_dim_reduce`
  - 如果设置，则对 global descriptors 做 PCA。
- `pca_whitening`
  - PCA 时是否 whitening。

### 3.4 检索与重排相关

- `coarse_top_k`
  - coarse retrieval 后保留多少候选用于 rerank / offset。
- `top_k_vals`
  - 最终统计 recall 时看的 k 列表，默认 `1..20`。
- `use_local_rerank`
  - 是否使用局部特征重排。
- `local_match_method`
  - `cosine_pool` / `mutual_nn` / `sim_map`。
- `rerank_alpha`
  - coarse score 与 local score 的加权系数。
- `local_top_m`
  - local matching 和 similarity map 中 top-m patch 匹配使用的 m。

### 3.5 Offset / localization 相关

- `use_offset_head`
  - 当前只是是否启用 similarity-map offset estimator。
- `offset_prediction_method`
  - 当前只有 `sim_map` 真正实现。
- `tile_size_m`
  - 如果 manifest 没写 tile extent，可以手工指定物理单位的 tile 尺寸。
- `tile_size_px`
  - 同上，但用像素尺度。
- `localization_thresholds`
  - 计算 `SR@thr` / `LSR@thr` 时使用的阈值。

### 3.6 缓存相关

- `cache_dense_features`
  - 是否缓存每张图像的 dense patch descriptors。
- `cache_vlad_descs`
  - 是否缓存 VLAD residual / assignment / centers。

## 4. 数据集接口与约定

当前 CVGL baseline 依赖 `CrossViewTileDataset`，它刻意模仿 AnyLoc / BaseDataset 的索引风格。

核心约定：

- 所有 database tiles 先排在前面。
- 所有 queries 接在后面。
- `__getitem__` 返回 `(image_tensor, global_index)`。
- `database_num` 是 tile 数量。
- `queries_num` 是 query 数量。
- `soft_positives_per_query[i]` 给出第 `i` 个 query 对应的正样本 database 索引列表。

### 4.1 Manifest 格式

最基础的格式如下。

`tiles.json`

```json
[
  {
    "tile_id": "tile_0001",
    "image": "tiles/tile_0001.png",
    "center": [100.0, 200.0],
    "extent": [32.0, 32.0]
  }
]
```

`queries.json`

```json
[
  {
    "query_id": "query_0001",
    "image": "queries/query_0001.png",
    "gt_tile_id": "tile_0001",
    "offset": [2.5, -1.0],
    "coord": [102.5, 199.0]
  }
]
```

字段语义：

- `tile_id`
  - tile 唯一标识。
- `image`
  - 图像路径，相对 `dataset_root` 解析。
- `center`
  - tile 中心坐标，供 localization 使用。
- `extent`
  - tile 的横向和纵向跨度，供 offset 从归一化坐标映射回真实尺度。
- `gt_tile_id`
  - query 对应真值 tile。
- `offset`
  - query 相对真值 tile 中心的偏移。
- `coord`
  - query 的绝对坐标。

附加说明：

- query 还可以带 `positives`，会被当作 `candidate_tile_ids`，并合入 `soft_positives_per_query`。
- 如果 `gt_tile_id` 缺失但 `positives` 非空，则第一项会被视为 gt tile。
- manifest 也支持外层包一层 dict，只要里面有 `items` / `tiles` / `queries` / `data` 之一。

### 4.2 图像读取与预处理

`CrossViewTileDataset.__getitem__` 默认使用：

- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

注意：

- 当前 CVGL baseline 没有显式 resize。
- 也就是说，图像输入尺寸直接决定 patch grid 大小。
- 后续 DINO 提特征前还会做一次 center crop，使高宽都能被 patch stride 整除。

## 5. DINOv2 dense feature 是怎么提取的

底层实现是 `utilities.py` 里的 `DinoV2ExtractFeatures`。

### 5.1 模型加载

通过 `torch.hub.load('facebookresearch/dinov2', dino_model)` 加载 DINOv2 模型，并设置为 `eval()`。

### 5.2 Hook 的位置

- 如果 `facet == "token"`，在指定 transformer block 上注册 forward hook，取该层输出 token。
- 如果 `facet` 是 `query` / `key` / `value`，则在该 block 的 `attn.qkv` 上注册 hook，再把 qkv 张量按最后一维拆开取对应部分。

### 5.3 是否包含 CLS token

当前 baseline 初始化 `DinoV2ExtractFeatures` 时没有传 `use_cls=True`，因此：

- 只保留 patch tokens。
- CLS token 会被丢弃。

### 5.4 特征归一化

`DinoV2ExtractFeatures` 默认 `norm_descs=True`，因此每个 patch descriptor 在最后一维上做 L2 normalize。

## 6. Dense descriptor 到 local/global descriptor 的转换

实现位于 `cvgl_retrieval.py` 的 `extract_dense_record()`。

### 6.1 中心裁剪到 14 的倍数

默认 `patch_stride = 14`。

对于输入图像 `[C, H, W]`：

- 先计算 `h_new = floor(H / 14) * 14`
- 再计算 `w_new = floor(W / 14) * 14`
- 用 `CenterCrop((h_new, w_new))` 居中裁剪

这一步非常重要，因为后续假设 patch grid 是规则的 `h_new/14 x w_new/14`。

### 6.2 Dense patch descriptors

裁剪后图像送入 DINOv2，得到：

- `patch_descs`，shape 为 `[num_patches, desc_dim]`

然后 reshape 成：

- `local_desc`，shape 为 `[grid_h, grid_w, desc_dim]`

同时记录：

- 原图大小 `image_hw`
- 裁剪后大小 `cropped_hw`
- patch grid 大小 `grid_hw`
- `patch_stride`

### 6.3 GeM 分支

如果 `global_agg == "gem"`，在 dense descriptor 上做 GeM pooling，得到一个 global descriptor，并立即做 L2 normalize。

注意：

- GeM 只影响 global descriptor。
- local descriptor 仍然被保留，后续 rerank / sim_map 还是会用到。

## 7. 缓存机制

当前实现有两层缓存：dense feature cache 和 VLAD cache。

### 7.1 Cache 根目录的构造

`scripts/dino_v2_vlad_CVGL.py` 里 `_build_cache_roots()` 会构造：

```text
<prog.cache_dir>/cvgl_descs/<dataset_cache_id>/<model-facet-layer-agg>/
```

其中 `dataset_cache_id`：

- `vpr` 模式下直接使用 `prog.vg_dataset_name`
- `cvgl` 模式下使用
  - `dataset_root`
  - `split`
  - `tiles_manifest path`
  - `queries_manifest path`
  拼出的字符串做 SHA1，前 12 位作为 hash

这样做的目的，是避免不同 manifest 或 split 之间缓存互相污染。

### 7.2 Dense feature cache

如果 `cache_dense_features=True`，cache 根目录为：

```text
.../dense/
```

单张图像缓存内容包括：

- `local_desc`
- `grid_hw`
- `image_hw`
- `cropped_hw`
- `patch_stride`
- `global_desc`，仅 GeM 模式会有值

cache key 来自 `dataset.get_image_relpaths(index)`，即路径相对形式。

### 7.3 VLAD cache

如果 `global_agg == "vlad"` 且 `cache_vlad_descs=True`，VLAD cache 根目录为：

```text
.../vlad-C<num_clusters>/
```

缓存文件包括：

- `c_centers.pt`
  - VLAD cluster centers
- `<cache_id>_r.pt`
  - residuals
- `<cache_id>_l.pt`
  - hard assignment labels
- `<cache_id>_s.pt`
  - soft assignment weights

如果 `c_centers.pt` 已存在，则 `fit_vlad_for_dataset()` 不再重新聚类，而是直接从缓存恢复 vocabulary。

## 8. VLAD vocabulary 如何拟合

当前 coarse retrieval 默认用 VLAD 聚合，拟合逻辑在 `fit_vlad_for_dataset()`。

执行方式如下：

1. 只在 database 图像上拟合，不使用 query。
2. database 索引先按 `sub_sample_db_vlad` 做下采样。
3. 对每个被选中的 database 图像：
   - 读取或提取 dense local descriptors
   - 展平为 `[num_patches, desc_dim]`
4. 把所有图像的 patch descriptors 拼接起来。
5. 用 `fast_pytorch_kmeans.KMeans` 做聚类，得到 `num_clusters` 个中心。

关键点：

- 聚类用的是 patch-level descriptor，不是 image-level descriptor。
- `VLAD` 默认 `dist_mode="cosine"`。
- `VLAD` 默认 `norm_descs=True`，所以输入 descriptor 在 fit 和 generate 前都会先 L2 normalize。
- 如果开启 `intra_norm`，每个 cluster 的 residual sum 会先做 intra-normalization，再拼接成最终 VLAD。

## 9. FeatureBank 的内容

`build_feature_bank()` 会为 database 和 query 分别构造一个 `FeatureBank`，内容包括：

- `indices`
  - 数据集原始索引
- `relpaths`
  - 图像相对路径
- `global_descs`
  - shape `[N, Dg]`
- `local_descs`
  - 长度为 `N` 的列表，每项 shape `[grid_h, grid_w, Dl]`
- `grid_hws`
- `image_hws`
- `cropped_hws`
- `patch_stride`

这意味着当前 pipeline 在整个评估阶段会同时保留：

- 所有图像的 global descriptors
- 所有图像的 local descriptor 网格

后者是 local rerank 和 offset 预测的基础。

## 10. Global descriptor 如何得到

### 10.1 VLAD 分支

对每张图像：

1. 取该图像的 `local_desc`。
2. 展平成 `[num_patches, desc_dim]`。
3. 调用 `vlad.generate(local_flat, relpath)`。
4. 对输出再次做 L2 normalize。

因此当前 VLAD global descriptor 的流程是：

- patch desc L2 normalize
- 按 cluster 计算 residual
- cluster 内 residual sum
- 可选 intra-normalization
- 拼接各 cluster 残差向量
- 全局 L2 normalize

### 10.2 GeM 分支

如果 `global_agg == "gem"`：

1. 在 patch descriptor 上做 GeM pooling。
2. 得到单个向量后做 L2 normalize。

当前脚本虽然支持 GeM，但名字仍叫 `dino_v2_vlad_CVGL.py`，因此后续如果扩展 GeM-only baseline，最好考虑重命名或拆脚本，避免误导。

## 11. PCA 的使用方式

如果 `pca_dim_reduce` 非空，会调用 `_maybe_apply_pca()`：

1. 用 database global descriptors 作为 PCA 的训练集。
2. 用同一个 PCA 变换 query global descriptors。
3. 如果 `pca_whitening=True`，使用 whitening。
4. PCA 之后再次做 L2 normalize。

这一步只作用在 global descriptors 上，不影响 local descriptors。

## 12. Coarse retrieval

`coarse_retrieve_topk()` 的实现很直接：

1. 对 database 和 query 的 global descriptors 按行做 L2 normalize。
2. 计算相似度矩阵：

```text
sim = query_global @ db_global^T
```

3. 对每个 query 取 top-k。

因此 coarse retrieval 实际上是 cosine similarity retrieval。

主脚本里请求的 top-k 为：

```text
max(max(top_k_vals), coarse_top_k)
```

原因是：

- 评估 recall 需要足够大的 top-k
- rerank 只需要 `coarse_top_k`

## 13. Ground truth 索引是如何映射的

因为 database 可能被 `sub_sample_db` 下采样，query 的 gt tile 索引必须先映射到当前 feature bank 的索引空间。

`_collect_gt_db_indices()` 的逻辑是：

1. 建立 `orig_db_index -> bank_index` 的映射。
2. 对每个 query 读取 `soft_positives_per_query`。
3. 只保留那些仍然出现在采样后 database 中的 positive。
4. 取第一个作为当前 query 的 gt db index。

这意味着：

- 如果 gt tile 因为 `sub_sample_db` 被采样掉了，该 query 会变成无效 retrieval query。
- 后续 retrieval metric 会跳过这些 query。

## 14. Local reranking

如果 `use_local_rerank=True`，则只对 coarse top-k 候选做 rerank，而不是对整个 database 做局部匹配。

### 14.1 三种 local matching 方法

`compute_local_match_score()` 当前支持三种打分方式。

#### a. `cosine_pool`

1. 计算 query patch 和 tile patch 的全连接 cosine similarity。
2. 对每个 query patch 取其最佳匹配。
3. 从这些最佳匹配分数中再取 top-m 平均。

直觉上，这是“query patch 是否能在 tile 中找到高质量匹配”的聚合。

#### b. `mutual_nn`

1. 对每个 query patch 找最相似的 tile patch。
2. 对每个 tile patch 找最相似的 query patch。
3. 只保留 mutual nearest neighbor 的配对分数并求平均。
4. 如果没有 mutual 对，则退化为所有 query patch 的 best-match 平均。

这是一个更稀疏、更接近局部匹配的打分方式。

#### c. `sim_map`

1. 先调用 `compute_local_similarity_map()`。
2. 得到 tile patch 网格上的分数图。
3. 取分数图展平后的 top-m 平均作为 local score。

当前默认就是 `sim_map`。

### 14.2 Similarity map 是怎么计算的

`compute_local_similarity_map()` 的逻辑：

1. 把 query local desc 和 tile local desc 都展平，并逐 patch L2 normalize。
2. 计算所有 query patches 到所有 tile patches 的相似度矩阵 `sims`。
3. 对每个 tile patch，看它和所有 query patches 的相似度。
4. 取其中 top-m 个 query patch 相似度的平均值。
5. 最终 reshape 回 tile patch 网格大小，得到 `sim_map[h, w]`。

所以 `sim_map[y, x]` 的语义是：

- tile 上该 patch 位置，与 query 中“最像它”的若干 patch 的平均相似度。

### 14.3 Rerank 分数如何融合

对每个 coarse 候选，最终 combined score 为：

```text
combined = rerank_alpha * coarse_score + (1 - rerank_alpha) * local_score
```

然后按 combined score 降序重新排序。

注意：

- 当前实现没有做 score calibration。
- cosine-based coarse score 和 local score 直接线性加权。
- `rerank_alpha=0.5` 表示二者同权。

## 15. Offset prediction

当前只有 `sim_map` 方法实现。

### 15.1 对哪个候选做 offset

只对最终排序后的 top-1 tile 预测 offset，不会对 top-k 每个候选都估计。

### 15.2 从 similarity map 到 offset 的步骤

`estimate_offset_from_similarity_map()` 的过程：

1. 先生成 top-1 tile 的 `sim_map`。
2. 对 `sim_map` 展平后乘以 `temperature=15.0`。
3. 做 softmax，得到 tile 网格上的概率分布。
4. 在 y 和 x 两个轴上分别构造从 `-0.5` 到 `0.5` 的 patch-center 坐标。
5. 计算概率分布下的期望位置 `(exp_x, exp_y)`。
6. 再乘以 tile 的实际 extent，得到最终 offset。

换句话说，当前 offset 预测不是取最大值位置，而是：

- 把 similarity map 解释成一个离散概率分布
- 取 soft-argmax 的期望坐标

### 15.3 坐标范围与尺度

归一化坐标系中：

- x 范围约为 `[-0.5, 0.5]`
- y 范围约为 `[-0.5, 0.5]`

再按 tile extent 缩放：

- 如果数据集 manifest 里有 `extent`，优先使用 `dataset.get_tile_extent()`
- 否则退回脚本参数 `tile_size_m` 或 `tile_size_px`
- 如果这些都没有，就退回以 patch grid 尺寸作为尺度

这意味着当前 offset 的物理语义依赖于 `extent` / `tile_size_m` / `tile_size_px` 是否正确。

## 16. Retrieval / offset / localization 指标

### 16.1 Retrieval 指标

`evaluate_cvgl_retrieval()` 计算：

- `R@k`
- `Top-1-Tile-Accuracy`
- `Num-Retrieval-Queries`

这里的 gt 只取每个 query 的第一个有效 positive。

### 16.2 Offset 指标

`evaluate_offset_predictions()` 计算：

- `Offset-Mean-Error`
- `Offset-Median-Error`
- `Offset-Eval-Count`
- `Offset-SR@thr`

其中误差是预测 offset 与 gt offset 的欧氏距离。

### 16.3 Localization 指标

`evaluate_localization()` 的逻辑更接近真实定位：

1. 先取预测 top-1 tile。
2. 取该 tile 中心 `pred_center`。
3. 取 gt tile 中心 `gt_center`。
4. 分别加上预测 offset 和 gt offset，得到两个绝对点。
5. 计算它们的欧氏距离。

如果 tile center 不存在，但预测 tile 正好等于 gt tile，则退化为只比较 offset。

最终输出：

- `Localization-Mean-Error`
- `Localization-Median-Error`
- `Localization-Eval-Count`
- `LSR@thr`

## 17. 输出结果里包含什么

主脚本最终会输出一个 `results` dict，并可保存成 JSON。

重要字段包括：

- 运行配置摘要
- `Coarse-R@k`
- `Refined-R@k`
- `Offset-*`
- `Localization-*`
- `Coarse-Indices`
- `Coarse-Scores`
- `Final-Indices`
- `Final-DB-Indices-Orig`
- `Final-Scores`
- `Pred-Offsets`
- `DB-Relpaths`
- `Query-Relpaths`
- `Local-Score-Table`

说明：

- `Final-Indices` 是 feature bank 内部索引。
- `Final-DB-Indices-Orig` 才映射回原始 dataset database 索引。
- `Sim-Maps` 本来能保存，但因为太大，当前被注释掉了。

## 18. 示例运行配置

仓库里的 `scripts/dino_v2_vlad_CVGL.sh` 给出了一个当前实际在用的例子：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/dino_v2_vlad_CVGL.py \
    --prog.cache-dir .cache \
    --task-mode cvgl \
    --cvgl-dataset-root data/Turkey_earthquake_2023_1_336_intile \
    --data-split test \
    --model-type dinov2_vitg14 \
    --global-agg vlad \
    --num-clusters 32 \
    --desc-layer 31 \
    --desc-facet key \
    --coarse-top-k 10 \
    --tile-size-px 512 \
    --use-local-rerank \
    --local-match-method sim_map \
    --use-offset-head \
    --offset-prediction-method sim_map \
    --exp-id cvgl_baseline_run
```

这套配置的含义是：

- 用 `dinov2_vitg14` 第 31 层的 `key` 特征。
- 用 32-cluster 的 VLAD 做 global aggregation。
- 对 coarse top-10 做 `sim_map` rerank。
- 再对 rerank 后 top-1 用 `sim_map` 做 offset 预测。
- tile 尺度按 512 px 解释。

## 19. `prepare_cvgl_dataset.py` 生成的数据集长什么样

当前仓库还提供了一个用于快速构建 CVGL 数据集的脚本：`scripts/prepare_cvgl_dataset.py`。

它的前提是假设：

- satellite 图和 UAV canvas 已经对齐到同一 2D 像素坐标系
- query 的真值位置就是 UAV crop 的中心

它会：

1. 把整张 satellite 图按固定 `tile_size` 切成规则 tiles。
2. 在 UAV canvas 上随机采样 query 中心。
3. 从 UAV canvas 中裁出 query 图像。
4. 为每个 query 自动记录：
   - `gt_tile_id`
   - `offset`
   - `coord`
   - `crop_bbox`
   - `rotation_deg`
   - `tile_row` / `tile_col`

其中 `offset` 的定义正是：

- query 中心减去所在 tile 的中心

因此它和上面 baseline 中的 localization 公式是对齐的。

## 20. 当前实现里容易被误解的点

### 20.1 这不是训练式 baseline

虽然参数名里有 `offset_head`、`offset_loss_type`，但当前脚本没有训练 learned head，也没有 loss backward。现在它完全是推理式方法。

### 20.2 `global_agg="gem"` 也会保留 local descriptors

即便 coarse retrieval 改成 GeM，local rerank 和 sim_map offset 仍然依赖 DINO dense descriptors。

### 20.3 Query 与 tile 的 patch grid 可以不同

当前 local matching 直接把 query patch 集和 tile patch 集做全连接相似度，因此不要求两者 grid 尺寸一致。

### 20.4 当前 similarity map 不是显式几何对齐

`sim_map` 的每个 tile patch 位置，只是聚合了与 query patch 的语义相似度，没有显式建模旋转、尺度、透视变化，也没有做 cross-correlation 或几何验证。

### 20.5 `joblib` 当前没有真正参与主流程

主脚本导入了 `joblib`，但当前版本没有在关键路径中使用它。

### 20.6 `VPAir_Distractor` 当前未在该脚本里实际使用

脚本导入了 `VPAir_Distractor`，但 `_load_anyloc_dataset()` 没有走到它。

## 21. 如果后续要改，最可能动到哪些位置

### 21.1 改 coarse global descriptor

重点看：

- `cvgl_retrieval.py`
  - `extract_dense_record()`
  - `fit_vlad_for_dataset()`
  - `build_feature_bank()`
- `utilities.py`
  - `DinoV2ExtractFeatures`
  - `VLAD`

### 21.2 改 local rerank

重点看：

- `compute_local_match_score()`
- `compute_local_similarity_map()`
- `rerank_with_local_features()`

### 21.3 改 offset 预测

重点看：

- `estimate_offset_from_similarity_map()`
- `predict_top1_offsets()`
- `evaluate_localization()`

### 21.4 改数据格式

重点看：

- `CrossViewTileDataset.from_json()`
- `soft_positives_per_query` 的构造
- `get_tile_center()` / `get_tile_extent()` / `get_query_gt_offset()`

## 22. 一句话概括当前 baseline

当前 DINOv2-VLAD-CVGL 的方法可以概括为：

先用 DINOv2 dense descriptors 构造每张 satellite tile 和 UAV query 的全局表示，利用 VLAD 做 coarse tile retrieval；再用 patch-level 相似度对 coarse top-k 候选做局部重排；最后把 top-1 tile 上的 patch similarity map 通过 soft-argmax 转成 query 相对 tile 中心的 offset，并据此评估最终 cross-view geo-localization 误差。
