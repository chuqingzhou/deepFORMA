# `deepforma/model`：分割网络与权重加载（中文说明）

本目录实现 **3D 类 TransUNet（CNN + Transformer）** 分割模型，以及推理时读取 `.pt` 权重的辅助逻辑。与「训练」「跑库」相关的脚本入口在仓库根目录的 `scripts/`，本目录只放 **模型定义与加载**。

---

## 1. 各文件职责

| 文件 | 作用 |
|------|------|
| **`transunet3d.py`** | 定义 **`TransUNet3D`**：`CNNEncoder` 提取多尺度特征 → **`PatchEmbedding3D`** 将瓶颈特征切成 patch 并映射为 token → 若干层 **`TransformerBlock`**（多头自注意力 + MLP）→ **`CNNDecoder`** 与 skip 连接融合并上采样 → **`Conv3d(…, 1, 1)`** 输出单通道 logits（前景/背景由后续流程解释）。 |
| **`checkpoint.py`** | **`load_segmentation_model`**：从磁盘读取 `torch` 保存的对象，识别其中的 `state_dict` / `model_state` 等常见键，将权重载入 **`build_model()`** 构建的 `TransUNet3D`；另支持仅供流水线冒烟的 **`demo_zscore_passthrough`** 伪模型（见 `scripts/create_demo_assets.py`）。 |
| **`__init__.py`** | 对外暴露包内常用符号（若需从包外 `import`，以实际 `__init__.py` 为准）。 |

默认结构超参在 **`build_model()`** 中写死并与训练脚本一致，例如：`embed_dim=768`、`num_heads=12`、`num_layers=6`、`patch_size=2`、`in_channels=1` 等。若你改动了网络宽度/深度，需保证 **训练保存的 checkpoint 与这里的 `build_model` 一致**，否则 `load_state_dict` 会报错或行为异常。

---

## 2. 训练阶段（如何用到本目录）

- **入口脚本**：`scripts/train_transformer_kfold.py`（具体参数见 `--help`）。
- **典型流程**：准备 H5/NRRD 等数据 → 配置折次与超参 → 脚本内部构建与本目录相同的 **`TransUNet3D`** → 前向、反传、验证 → 将 **`state_dict`（或项目约定的字典键，如 `model_state`）** 写入 `.pt`。
- **本目录角色**：只提供 **前向网络定义**；数据增强、损失、优化器、日志、分布式等逻辑均在 **训练脚本或其它模块** 中，不在 `transunet3d.py` 内展开。

训练得到的权重文件（例如 Zenodo 提供的 **`best_transformer.pt`**）在推理时应放在任意路径，由 **`scripts/build_database.py` 的 `--model-path`** 指向该文件。

---

## 3. 推理 / 建库阶段（如何用到本目录）

- **入口脚本**：`scripts/build_database.py`（以及同一条链路上的预处理、连通域拆分等）。
- **典型流程**：读入原始或预处理后的 3D 体数据 → 调用 **`checkpoint.load_segmentation_model`** → 得到 **`nn.Module`** 与设备字符串 → `model.eval()` 后在 **`torch.no_grad()`** 下前向 → 得到 logits 或概率图 → 阈值化、连通域标记、按 well 导出 H5、再交给特征与 Excel 导出。
- **本目录角色**：保证 **与训练时相同的 `TransUNet3D` 拓扑** 被实例化，并把 checkpoint 里的张量 **按层名对齐加载**。

若 checkpoint 内含 **`model_type: demo_zscore_passthrough`**，则不会构建 `TransUNet3D`，而是走演示用直通模型，仅用于验证安装与 IO，**不能用于真实分割**。

---

## 4. 数据在模型里怎么走（直观版）

1. **输入**：单通道 3D 体，形状大致为 `(B, 1, D, H, W)`（具体由上游预处理决定，常见为固定或滑窗 patch）。
2. **CNN 编码器**：四级卷积下采样，得到瓶颈张量及三组 skip。
3. **Patch 嵌入**：在瓶颈尺度上用 **`Conv3d` 核大小 = patch_size、步幅 = patch_size** 的非重叠 patch，把每个 patch 压成一维向量并线性映射到 **`embed_dim`**，得到 token 序列 `(B, N, embed_dim)`，并记录空间栅格大小 `(D', H', W')`。
4. **Transformer**：默认 **6 层** `TransformerBlock`，每层为 **Pre-LN 风格**（先 `LayerNorm` 再注意力/MLP）加残差，在 token 维上做全局自注意力。
5. **CNN 解码器**：将 token 投回通道维并 reshape 为 3D 特征图，再与 skip 逐级拼接、上采样，最后 **1×1 卷积** 得到 **单通道 logits**。

损失函数、sigmoid/sigmoid+阈值、后处理 **不在** `transunet3d.py` 的 `forward` 末尾强制绑定；由训练或推理脚本决定。

---

## 5. 与仓库其它部分的关系（便于审稿人定位）

- **特征与形态学指标**：`deepforma/features/`（如 `nine_metrics.py`），在分割与掩膜确定之后计算。
- **读写与格式**：`deepforma/io/`（H5、NRRD 等）。
- **与论文图表脚本**：本仓库为 **核心软件**；论文级作图脚本若存在，一般在其它目录或独立材料包，不在 `model/` 内。

英文版说明见同目录 **`README.md`**。
