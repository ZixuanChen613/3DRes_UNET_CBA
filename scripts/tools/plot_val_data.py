import os
import numpy as np
import matplotlib.pyplot as plt

def save_seg_bd_ins_images(semantic_mask, instance_boundary, bc_w_result, output_dir="output_images"):
    """
    保存 (32, 640, 640) 大小的 semantic_mask、instance_boundary 和 bc_w_result 图像，
    每三张合并为一个文件，并生成32个文件。
    
    Args:
        semantic_mask (numpy.ndarray): 形状为 (32, 640, 640) 的语义掩码图像。
        instance_boundary (numpy.ndarray): 形状为 (32, 640, 640) 的实例边界图像。
        bc_w_result (numpy.ndarray): 形状为 (32, 640, 640) 的结果图像。
        output_dir (str): 输出保存图片的目录。
    """
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_images = semantic_mask.shape[0]
    for i in range(num_images):
        # 获取第 i 张图像
        semantic_img = semantic_mask[i]
        boundary_img = instance_boundary[i]
        result_img = bc_w_result[i]

        # 创建一个图像窗口，包含 1 行 3 列的子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(semantic_img, cmap="gray")
        axes[0].set_title("Semantic Mask")
        axes[0].axis("off")

        axes[1].imshow(boundary_img, cmap="gray")
        axes[1].set_title("Instance Boundary")
        axes[1].axis("off")

        axes[2].imshow(result_img, cmap="gray")
        axes[2].set_title("BC_W Result")
        axes[2].axis("off")

        # 调整布局，保存图片
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"combined_image_{i+1}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

# 示例调用
# 假设 semantic_mask, instance_boundary, bc_w_result 是你要处理的 (32, 640, 640) 大小的 NumPy 数组
# semantic_mask = np.random.randint(0, 255, (32, 640, 640), dtype=np.uint8)
# instance_boundary = np.random.randint(0, 255, (32, 640, 640), dtype=np.uint8)
# bc_w_result = np.random.randint(0, 255, (32, 640, 640), dtype=np.uint8)

# 调用函数，保存图片
# save_segmentation_images(semantic_mask, instance_boundary, bc_w_result)
