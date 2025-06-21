import matplotlib.pyplot as plt
import os
import torch

def save_im_seg_bd_images(input_data, target_list, pred_data, iter_total, save_dir="visualizations"):
    """
    可视化并保存所有深度的体积数据、多个 target 和 pred。
    
    Args:
        input_data (np.ndarray or torch.Tensor): 输入的体积数据，形状为 [B, C, D, H, W]。
        target_list (list of np.ndarray or torch.Tensor): 一个长度为 2 的列表，包含两个 shape [B, C, D, H, W] 的标签。
        pred_data (torch.Tensor): 模型的预测结果，形状为 [B, C, D, H, W]。
        iter_total (int): 当前训练迭代数，用于文件命名。
        save_dir (str): 图像保存的目录。
    """
    # 如果 input_data 是 PyTorch 张量，将其转换为 NumPy 数组
    if torch.is_tensor(input_data):
        input_data = input_data.detach().cpu().numpy()
    
    # 如果 target_list 中的元素是 PyTorch 张量，也将它们转换为 NumPy 数组
    target_list = [t.detach().cpu().numpy() if torch.is_tensor(t) else t for t in target_list]

    # 如果 pred_data 是 PyTorch 张量，将其转换为 NumPy 数组
    if torch.is_tensor(pred_data):
        pred_data = pred_data.detach().cpu().numpy()

    # 获取深度 D
    _, _, D, _, _ = input_data.shape  # 假设 input_data, target_list 和 pred_data 的形状一致

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历每一个 depth_idx，保存图像
    for depth_idx in range(D):
        # 提取第一个样本的指定深度切片
        input_slice = input_data[0, 0, depth_idx, :, :]  # 输入数据的切片
        pred_slices = [pred_data[0, i, depth_idx, :, :] for i in range(2)]  # 预测结果的两个通道切片

        # 提取 target_list 中的每个 target 的指定深度切片
        target_slices = [target[0, 0, depth_idx, :, :] for target in target_list]

        # 可视化体积数据、多个 target 和预测结果
        plt.figure(figsize=(25, 5))  # 调整图像大小以容纳五个子图
        
        # 显示输入数据的切片
        plt.subplot(1, 5, 1)
        plt.imshow(input_slice, cmap='gray')
        plt.title(f"Input slice at depth {depth_idx}")
        plt.axis('off')

        # 显示 target_list 中的第一个 target 的切片
        plt.subplot(1, 5, 2)
        plt.imshow(target_slices[0], cmap='gray')
        plt.title(f"GT semantic mask slice at depth {depth_idx}")
        plt.axis('off')

        # 显示 target_list 中的第二个 target 的切片
        plt.subplot(1, 5, 3)
        plt.imshow(target_slices[1], cmap='gray')
        plt.title(f"GT instance boundary slice at depth {depth_idx}")
        plt.axis('off')

        # 显示预测结果的第一个通道切片
        plt.subplot(1, 5, 4)
        plt.imshow(pred_slices[0], cmap='gray')
        plt.title(f"Pred semantic mask slice at depth {depth_idx}")
        plt.axis('off')

        # 显示预测结果的第二个通道切片
        plt.subplot(1, 5, 5)
        plt.imshow(pred_slices[1], cmap='gray')
        plt.title(f"Pred instance boundary slice at depth {depth_idx}")
        plt.axis('off')

        # 保存图像
        save_path = os.path.join(save_dir, f"iter_{iter_total}_slice_{depth_idx}.png")
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
        plt.close()  # 关闭当前图，防止内存泄漏
