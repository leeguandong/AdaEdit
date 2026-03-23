# AdaEdit 快速开始指南

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/AdaEdit.git
cd AdaEdit
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

下载 FLUX.1-dev 模型权重：

```bash
# 使用 huggingface-cli
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/flux-dev

# 或者手动下载并放置到 checkpoints/ 目录
```

## 基础使用

### 命令行方式

```bash
python adaedit.py \
    -i examples/source.jpg \
    -sp "A photo of a cat" \
    -tp "A photo of a dog" \
    -o outputs/
```

### Python API 方式

```python
from api import AdaEditPipeline

# 初始化
pipeline = AdaEditPipeline(device="cuda")

# 编辑图像
result = pipeline.edit(
    source_image="examples/source.jpg",
    source_prompt="A photo of a cat",
    target_prompt="A photo of a dog",
    edit_object="cat",
    inject_schedule="sigmoid",
    use_channel_ls=True,
    seed=42
)

# 保存结果
result.save("outputs/result.jpg")
```

## 常见编辑场景

### 1. 对象替换

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --inject_schedule sigmoid \
    --use_channel_ls
```

### 2. 属性修改

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A red car" \
    -tp "A blue car" \
    --edit_object "car" \
    --inject_schedule cosine
```

### 3. 风格迁移

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A portrait photo" \
    -tp "A portrait in anime style" \
    --edit_type style \
    --inject_schedule cosine
```

### 4. 添加对象

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A room" \
    -tp "A room with a vase on the table" \
    --edit_type add \
    --inject_schedule sigmoid
```

## 核心参数说明

### 注入调度 (Injection Schedule)

- `--inject_schedule sigmoid`（推荐）：平滑过渡，适合大多数场景
- `--inject_schedule cosine`：余弦衰减，更平滑的过渡
- `--inject_schedule linear`：线性衰减，简单直接
- `--inject_schedule binary`：二值截断（原始方法）

### 通道选择性 Latents-Shift

- `--use_channel_ls`：启用通道选择性扰动（推荐）
- `--channel_ls_temp 1.0`：温度参数，控制通道重要性分布的锐度

### 其他重要参数

- `--inject 4`：注入步数阈值（通常 3-5 步）
- `--kv_mix_ratio 0.9`：KV-Mix 混合比例（0.8-0.95）
- `--ls_ratio 0.25`：Latents-Shift 强度（0.2-0.3）
- `--guidance 4.0`：引导强度（3.0-5.0）

## 调优建议

### 背景保持不佳

- 增加 `--inject` 步数（如 5 或 6）
- 增加 `--kv_mix_ratio`（如 0.95）
- 使用 `sigmoid` 或 `cosine` 调度

### 编辑效果不明显

- 减少 `--inject` 步数（如 3）
- 增加 `--ls_ratio`（如 0.3）
- 启用 `--use_channel_ls`

### 结构变形

- 启用 `--use_channel_ls`
- 降低 `--channel_ls_temp`（如 0.5）
- 增加 `--inject` 步数

## 低显存模式

如果显存不足（< 16GB），使用 `--offload` 参数：

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A photo" \
    -tp "An edited photo" \
    --offload
```

## 批量处理示例

```bash
#!/bin/bash
for img in images/*.jpg; do
    python adaedit.py \
        -i "$img" \
        -sp "A photo" \
        -tp "An edited photo" \
        -o outputs/ \
        --seed 42
done
```

## 故障排除

### 模型加载失败

确保模型文件在正确位置：
```
checkpoints/
└── flux-dev/
    ├── flux1-dev.safetensors
    ├── ae.safetensors
    └── ...
```

### CUDA 内存不足

1. 使用 `--offload` 参数
2. 减少 `--num_steps`（如 10）
3. 使用较小的图像尺寸

### 生成结果不理想

1. 尝试不同的 `--inject_schedule`
2. 调整 `--inject` 步数
3. 启用 `--use_channel_ls`
4. 尝试不同的随机种子

## 更多示例

查看 `examples/run_example.sh` 获取更多使用示例。
