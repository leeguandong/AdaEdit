"""
AdaEdit Python API 使用示例
"""
from api import AdaEditPipeline, edit_image
from PIL import Image

# ============================================================
# 示例 1: 基础使用
# ============================================================

def example_basic():
    """基础图像编辑示例"""
    pipeline = AdaEditPipeline(device="cuda")

    result = pipeline.edit(
        source_image="examples/cat.jpg",
        source_prompt="A photo of a cat",
        target_prompt="A photo of a dog",
        edit_object="cat",
        seed=42
    )

    result.save("outputs/basic_result.jpg")
    print("✓ Basic example completed")


# ============================================================
# 示例 2: 使用不同的注入调度
# ============================================================

def example_schedules():
    """比较不同注入调度的效果"""
    pipeline = AdaEditPipeline(device="cuda")

    schedules = ["binary", "sigmoid", "cosine", "linear"]

    for schedule in schedules:
        result = pipeline.edit(
            source_image="examples/test.jpg",
            source_prompt="A photo of a car",
            target_prompt="A photo of a red car",
            inject_schedule=schedule,
            seed=42
        )
        result.save(f"outputs/schedule_{schedule}.jpg")
        print(f"✓ {schedule} schedule completed")


# ============================================================
# 示例 3: 通道选择性 Latents-Shift
# ============================================================

def example_channel_selective():
    """使用通道选择性扰动"""
    pipeline = AdaEditPipeline(device="cuda")

    # 不使用通道选择性
    result_standard = pipeline.edit(
        source_image="examples/scene.jpg",
        source_prompt="A street scene",
        target_prompt="A street scene at night",
        use_channel_ls=False,
        seed=42
    )
    result_standard.save("outputs/standard_ls.jpg")

    # 使用通道选择性
    result_channel = pipeline.edit(
        source_image="examples/scene.jpg",
        source_prompt="A street scene",
        target_prompt="A street scene at night",
        use_channel_ls=True,
        channel_ls_temp=1.0,
        seed=42
    )
    result_channel.save("outputs/channel_ls.jpg")

    print("✓ Channel-selective comparison completed")


# ============================================================
# 示例 4: 风格迁移
# ============================================================

def example_style_transfer():
    """风格迁移示例（不使用 Latents-Shift）"""
    pipeline = AdaEditPipeline(device="cuda")

    result = pipeline.edit(
        source_image="examples/portrait.jpg",
        source_prompt="A portrait photo",
        target_prompt="A portrait in anime style",
        edit_type="style",
        inject_schedule="cosine",
        seed=42
    )

    result.save("outputs/style_transfer.jpg")
    print("✓ Style transfer completed")


# ============================================================
# 示例 5: 批量处理
# ============================================================

def example_batch_processing():
    """批量处理多张图像"""
    pipeline = AdaEditPipeline(device="cuda")

    images = [
        ("examples/img1.jpg", "A cat", "A dog"),
        ("examples/img2.jpg", "A red car", "A blue car"),
        ("examples/img3.jpg", "Day scene", "Night scene"),
    ]

    for i, (img_path, src_prompt, tgt_prompt) in enumerate(images):
        try:
            result = pipeline.edit(
                source_image=img_path,
                source_prompt=src_prompt,
                target_prompt=tgt_prompt,
                inject_schedule="sigmoid",
                use_channel_ls=True,
                seed=42
            )
            result.save(f"outputs/batch_{i}.jpg")
            print(f"✓ Processed {img_path}")
        except Exception as e:
            print(f"✗ Failed to process {img_path}: {e}")


# ============================================================
# 示例 6: 快捷函数
# ============================================================

def example_quick_edit():
    """使用快捷函数进行编辑"""
    result = edit_image(
        source_image="examples/test.jpg",
        source_prompt="A photo",
        target_prompt="An edited photo",
        inject_schedule="sigmoid",
        use_channel_ls=True,
        seed=42
    )

    result.save("outputs/quick_edit.jpg")
    print("✓ Quick edit completed")


# ============================================================
# 示例 7: 参数调优
# ============================================================

def example_parameter_tuning():
    """参数调优示例"""
    pipeline = AdaEditPipeline(device="cuda")

    # 强背景保持
    result_preserve = pipeline.edit(
        source_image="examples/test.jpg",
        source_prompt="A photo",
        target_prompt="An edited photo",
        inject=6,  # 更多注入步数
        kv_mix_ratio=0.95,  # 更高的混合比例
        inject_schedule="sigmoid",
        seed=42
    )
    result_preserve.save("outputs/strong_preserve.jpg")

    # 强编辑效果
    result_edit = pipeline.edit(
        source_image="examples/test.jpg",
        source_prompt="A photo",
        target_prompt="An edited photo",
        inject=3,  # 更少注入步数
        ls_ratio=0.3,  # 更强的扰动
        use_channel_ls=True,
        inject_schedule="sigmoid",
        seed=42
    )
    result_edit.save("outputs/strong_edit.jpg")

    print("✓ Parameter tuning examples completed")


# ============================================================
# 示例 8: 从 PIL Image 或 numpy array 编辑
# ============================================================

def example_from_memory():
    """从内存中的图像进行编辑"""
    pipeline = AdaEditPipeline(device="cuda")

    # 从 PIL Image
    pil_img = Image.open("examples/test.jpg")
    result = pipeline.edit(
        source_image=pil_img,
        source_prompt="A photo",
        target_prompt="An edited photo",
        seed=42
    )
    result.save("outputs/from_pil.jpg")

    # 从 numpy array
    import numpy as np
    np_img = np.array(pil_img)
    result = pipeline.edit(
        source_image=np_img,
        source_prompt="A photo",
        target_prompt="An edited photo",
        seed=42
    )
    result.save("outputs/from_numpy.jpg")

    print("✓ Memory-based editing completed")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("Running AdaEdit API examples...")
    print("=" * 60)

    # 运行示例（根据需要注释/取消注释）
    example_basic()
    # example_schedules()
    # example_channel_selective()
    # example_style_transfer()
    # example_batch_processing()
    # example_quick_edit()
    # example_parameter_tuning()
    # example_from_memory()

    print("=" * 60)
    print("All examples completed!")
