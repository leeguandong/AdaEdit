<h3 align="center">
    AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing
</h3>

<p align="center">
<a href="https://arxiv.org/abs/xxxx.xxxxx"><img alt="Paper" src="https://img.shields.io/badge/Paper-AdaEdit-b31b1b.svg"></a>
<a href="https://github.com/yourusername/AdaEdit"><img src="https://img.shields.io/static/v1?label=GitHub&message=repository&color=green"></a>
</p>

<p align="center">
<span style="color:#137cf3; font-family: Gill Sans">Guandong Li,</span><sup></sup></a>
<span style="color:#137cf3; font-family: Gill Sans">Zhaobin Chu</span></a> <br>
<span style="font-size: 13.5px">iFLYTEK Research</span><br>
</p>

## Abstract

**AdaEdit** 是一个**免训练**的自适应图像编辑框架，专为基于流模型（Flow Models）的图像编辑设计。我们通过两个核心创新解决了反演编辑中的"注入困境"：

**(1) 渐进式注入调度（Progressive Injection Schedule）**：用连续衰减函数（sigmoid/cosine/linear）替代二值截断，消除特征不连续性，降低超参数敏感度。

**(2) 通道选择性潜在扰动（Channel-Selective Latent Perturbation）**：基于通道重要性估计，对编辑相关通道施加强扰动，对结构通道保持弱扰动，在保持结构保真度的同时实现有效编辑。

在 PIE-Bench 基准测试（700张图片）上，AdaEdit 相比基线方法实现了显著改进：
- **LPIPS ↓ 8.7%**（背景保持）
- **SSIM ↑ 2.6%**（结构相似度）
- **PSNR ↑ 2.3%**（峰值信噪比）
- **CLIP ≈ -0.9%**（编辑准确性几乎无损）

## Installation

```bash
# 克隆仓库
git clone https://github.com/yourusername/AdaEdit.git
cd AdaEdit

# 安装依赖
pip install -r requirements.txt
```

## 模型下载

下载 FLUX.1-dev 模型权重并放置到 `checkpoints/` 目录：

```bash
# 下载 FLUX.1-dev 模型
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/flux-dev
```

## Usage

### 基础用法

```bash
python adaedit.py \
    --source_img source.jpg \
    --source_prompt "A photo of a cat" \
    --target_prompt "A photo of a dog" \
    --output_dir outputs/
```

### 完整参数

```bash
python adaedit.py \
    -i source.jpg \
    -sp "A photo of a cat" \
    -tp "A photo of a dog" \
    -o outputs/ \
    --edit_object "cat" \
    --num_steps 15 \
    --guidance 4.0 \
    --inject 4 \
    --inject_schedule sigmoid \
    --kv_mix_ratio 0.9 \
    --ls_ratio 0.25 \
    --use_channel_ls \
    --channel_ls_temp 1.0 \
    --seed 42
```

### 参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--source_img` | `-i` | **必需** | 源图像路径 |
| `--source_prompt` | `-sp` | **必需** | 源图像描述 |
| `--target_prompt` | `-tp` | **必需** | 目标编辑描述 |
| `--output_dir` | `-o` | `outputs/` | 输出目录 |
| `--edit_object` | | `""` | 编辑对象（用于mask提取） |
| `--num_steps` | | `15` | 采样步数 |
| `--guidance` | | `4.0` | 引导强度 |
| `--inject` | | `4` | 注入步数阈值 |
| `--inject_schedule` | | `sigmoid` | 注入调度类型：`binary`/`sigmoid`/`cosine`/`linear` |
| `--kv_mix_ratio` | | `0.9` | KV-Mix 混合比例 |
| `--ls_ratio` | | `0.25` | Latents-Shift 强度 |
| `--use_channel_ls` | | `False` | 启用通道选择性 Latents-Shift |
| `--channel_ls_temp` | | `1.0` | 通道重要性温度参数 |
| `--seed` | | `0` | 随机种子（0=随机） |
| `--offload` | | `False` | 低显存模式 |

### 示例

```bash
# 示例1：对象替换（使用渐进式sigmoid调度）
python adaedit.py \
    -i examples/cat.jpg \
    -sp "A photo of a cat on a sofa" \
    -tp "A photo of a dog on a sofa" \
    --edit_object "cat" \
    --inject_schedule sigmoid \
    --use_channel_ls

# 示例2：风格迁移（不使用Latents-Shift）
python adaedit.py \
    -i examples/portrait.jpg \
    -sp "A portrait photo" \
    -tp "A portrait in anime style" \
    --edit_type style \
    --inject_schedule cosine

# 示例3：低显存模式
python adaedit.py \
    -i examples/scene.jpg \
    -sp "A street scene" \
    -tp "A street scene at night" \
    --offload
```

## 核心特性

### 1. 渐进式注入调度

传统方法使用二值截断（前N步注入=1，后续步骤注入=0），导致特征不连续。AdaEdit 提供三种连续衰减函数：

- **Sigmoid**（推荐）：平滑过渡，中等锐度
- **Cosine**：余弦衰减，更平滑
- **Linear**：线性衰减，最简单

### 2. 通道选择性潜在扰动

不同通道编码不同信息（结构/颜色/纹理）。AdaEdit 自动估计每个通道的编辑相关性：
- 编辑相关通道：强扰动（促进内容变化）
- 结构通道：弱扰动（保持布局稳定）

### 3. 即插即用

AdaEdit 无需训练，可与多种 ODE 求解器配合使用：
- Euler（基础）
- RF-Solver（二阶）
- FireFlow（速度重用，推荐）

## Python API

```python
from adaedit import AdaEditPipeline

# 初始化
pipeline = AdaEditPipeline(
    model_path="checkpoints/flux-dev",
    device="cuda"
)

# 编辑图像
result = pipeline.edit(
    source_image="source.jpg",
    source_prompt="A photo of a cat",
    target_prompt="A photo of a dog",
    edit_object="cat",
    inject_schedule="sigmoid",
    use_channel_ls=True,
    num_steps=15,
    guidance=4.0,
    seed=42
)

# 保存结果
result.save("output.jpg")
```

## 技术细节

### 渐进式注入权重计算

```python
# Sigmoid 调度
w(t) = 1 / (1 + exp(k * (t/T_inj - 0.7)))

# Cosine 调度
w(t) = 0.5 * (1 + cos(π * t/T_inj))

# Linear 调度
w(t) = max(1 - t/T_inj, 0)
```

### 通道重要性估计

```python
# 计算每个通道的分布差异
channel_diff = |mean(source_channel) - mean(random_channel)|

# Softmax 归一化
channel_weight = softmax(channel_diff / temperature) * num_channels
```

## Citation

如果您在研究中使用了 AdaEdit，请引用我们的论文：

```bibtex
@article{li2026adaedit,
  title={AdaEdit: Adaptive Temporal and Channel Modulation for Flow-Based Image Editing},
  author={Li, Guandong and Chu, Zhaobin},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

## Acknowledgments

- [FLUX.1](https://github.com/black-forest-labs/flux) - 基础流模型
- [FireFlow](https://github.com/xxx/fireflow) - 高效 ODE 求解器
- [PIE-Bench](https://github.com/xxx/pie-bench) - 评测基准

## License

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。
