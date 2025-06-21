from skimage import measure
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import numpy as np
from scipy.ndimage import binary_fill_holes
import numpy as np
from skimage.measure import regionprops, label
from skimage.morphology import binary_dilation


def filter_low_frequency_elements(array, frequency_threshold):
    """
    将三维数组中非零元素中频率小于指定阈值的数字设置为 0。
    
    参数:
        array (numpy.ndarray): 输入的三维数组。
        frequency_threshold (int): 频率阈值，小于此值的数字将被设置为 0。
        
    返回:
        numpy.ndarray: 修改后的数组。
    """
    # 获取 unique 值和对应频率
    unique_values, counts = np.unique(array, return_counts=True)
    
    # 找出频率小于阈值的值
    low_frequency_values = unique_values[counts < frequency_threshold]
    
    # 创建一个副本并将低频值设置为 0
    modified_array = array.copy()
    for value in low_frequency_values:
        modified_array[array == value] = 0
    
    return modified_array

# 为实例动态分配不同的颜色
def generate_color_map(num_colors):
    np.random.seed(0)  # 固定随机种子确保一致性
    colors = np.random.rand(num_colors + 1, 3)  # 生成 num_colors + 1 个随机颜色，包含背景颜色
    colors[0] = [0, 0, 0]  # 将背景颜色设置为黑色
    return colors


def improved_instance_segmentation(
    semantic_mask,
    instance_boundary,
    semantic_threshold=0.9,
    boundary_threshold=12,
    mask_threshold=150,
    seedpoint_threshold=500):
    """
    利用语义掩码与边界信息进行改进的实例分割流程。
    
    参数：
    -------
    semantic_mask : np.ndarray
        语义分割得到的掩码 (通常是 3D: [Z, Y, X] 或 2D: [Y, X] ), 
        取值范围一般在 0 ~ 255。

    instance_boundary : np.ndarray
        实例边界图，形状与 semantic_mask 相同，用于辅助寻找种子点。

    semantic_threshold : float, 默认 0.9
        用于根据语义置信度过滤种子点，通常需乘以 255。

    boundary_threshold : float, 默认 12
        边界置信度阈值，用于进一步筛选远离边界的像素/体素。

    mask_threshold : int, 默认 150
        将语义掩码二值化时使用的阈值。

    seedpoint_threshold : int, 默认 500
        用于滤除小连通区域（种子点）的最小体素数阈值。

    filter_low_freq_func : callable, 可选
        处理小连通区域的函数，应接受 (label_map, min_size) 两个参数，
        返回过滤后的连通区域标签图。

    color_map_func : callable, 可选
        生成颜色映射表的函数，应接受 (max_label_value) 参数并返回 (N, 3) 形状的颜色表。
        例如 generate_color_map(np.max(instance_id)) 。

    返回：
    -------
    colored_instance_id : np.ndarray
        与输入形状相同的彩色实例图，每个实例用不同的颜色编码。
    """
    # 1. 将语义掩码二值化
    binary_semantic_mask = (semantic_mask >= mask_threshold).astype(int)

    # 2. 在每个切片上使用 binary_fill_holes 填充空洞
    binary_filled_mask = np.zeros_like(binary_semantic_mask)
    # 如果是 3D（Z, Y, X），逐 Z 处理；如果是 2D，则只需填充一次
    if binary_semantic_mask.ndim == 3:
        for z in range(binary_semantic_mask.shape[0]):
            binary_filled_mask[z] = binary_fill_holes(binary_semantic_mask[z])
    elif binary_semantic_mask.ndim == 2:
        binary_filled_mask = binary_fill_holes(binary_semantic_mask)
    else:
        raise ValueError("语义掩码维度不符合预期，仅支持 2D 或 3D。")

    # 3. 利用语义和边界阈值，生成种子点
    #    (semantic_mask > 255*semantic_threshold) 取高置信度区
    #    (instance_boundary < boundary_threshold) 取远离边界区域
    seed_map = (semantic_mask > int(255 * semantic_threshold)) & (instance_boundary < boundary_threshold)

    # 4. 连通区域标记
    seed_points = measure.label(seed_map)

    # 5. 滤除体素数小于 seedpoint_threshold 的小连通区
    filtered_seed_points = filter_low_frequency_elements(seed_points, seedpoint_threshold)

    # 6. 计算距离变换
    distance_map = distance_transform_edt(binary_filled_mask)

    # 7. 利用 watershed 进行分水岭分割
    #    注意：watershed_line=False 表示不在标签之间产生分割线
    labels = watershed(-distance_map, filtered_seed_points, mask=binary_filled_mask, watershed_line=False)

    # 8. 第二次连通区域标记，获取实例 ID
    instance_id = measure.label(labels, connectivity=1)

    # 9. 根据最大实例数生成对应数量的颜色并生成彩色实例图
    colors = generate_color_map(np.max(instance_id))  # 颜色映射表
    colored_instance_id = colors[instance_id]     # 将每个实例 ID 映射到对应的颜色

    return instance_id, colored_instance_id



from scipy.ndimage import label

def remove_small_segments(instance_id, area_thres=200):
    """
    在 3D 图像中遍历每一层（Z 维度），对于同一个实例（同一 ID）出现的多连通区域，
    检查是否存在“占比 < 25% 且面积 < area_thres”的情况，一旦满足则将整个连通区域
    替换为该区域中出现次数最多的像素值。

    参数:
    ----------
    instance_id : np.ndarray
        3D 数组 (Z, H, W)，其中的值代表像素所属的实例 ID 或者背景（0）。
    area_thres : int
        区域大小阈值，当连通区域内部某个像素值占比小于 25% 且其计数小于该阈值时，触发替换逻辑。

    返回:
    ----------
    new_instance_id : np.ndarray
        替换后得到的 3D 数组 (Z, H, W)，尺寸与输入相同。
    """

    new_instance_id = instance_id.copy()

    # 遍历每一层 (Z 轴)
    for z in range(instance_id.shape[0]):
        current_slice = instance_id[z, :, :]  # 当前 Z 层的切片
        # 获取当前层的实例 ID（包括背景 0）
        unique_ids = np.unique(current_slice)

        # 标记当前切片中的连通区域
        curr_mask = (current_slice > 0)
        current_slice_regions, _ = label(curr_mask)
    
        # 遍历每个实例 ID
        for instance in unique_ids:
            # 跳过背景
            if instance == 0:
                continue

            # 生成当前实例的二值掩膜
            binary_mask = (current_slice == instance)
            # # remove small regions 只有
            if binary_mask.sum() < area_thres:
                new_instance_id[z, :, :][binary_mask] = 0
                continue

            # 对该实例掩膜进行连通区域分析
            labeled_mask, num_features = label(binary_mask)

            
            # 若有多个连通区域，合并""过分割"中的占比小的实例，用周围像素覆盖
            if num_features > 1:
                max_area_label = -1
                max_area = 0
                # 保留最大的区域
                for region_label in range(1, num_features + 1):
                    # 生成当前区域的布尔掩膜
                    region_mask = (labeled_mask == region_label)
                    # 计算当前区域的面积（像素数）
                    area = region_mask.sum()
                    # 若该区域面积大于当前最大值，则更新
                    if area > max_area:
                        max_area = area
                        max_area_label = region_label
                #####################################################
                # remove 含有多个 small regions 
                if max_area < area_thres:
                    new_instance_id[z, :, :][binary_mask] = 0
                    continue
                #####################################################

                # 计算每个连通区域的面积（像素数）
                for region_label in range(1, num_features + 1):
                    if max_area_label == region_label:
                        continue
                    region_mask = (labeled_mask == region_label)
                    # 获取 overlapping_labels 
                    overlapping_labels = np.unique(current_slice_regions[region_mask])
                    mask = (current_slice_regions == overlapping_labels)
                    total_area = mask.sum()
                    
                    # 提取该区域对应的像素值
                    region_ids = current_slice[mask]
                    # 获取唯一像素值及其出现计数
                    region_unique_ids, counts = np.unique(region_ids, return_counts=True)

                    # 检查是否所有像素值占比都 >= 25%
                    all_greater_than_25_percent = all(
                        (count / total_area) >= 0.25 for count in counts
                    )

                    if not all_greater_than_25_percent: # 排除欠分割得实例
                        # print('False')
                        # 若存在某些像素值占比 < 25%，
                        need_replace = False
                        # 判断当前的region mask是不是在 curr_mask_region中占比小
                        if region_mask.sum() / total_area < 0.25:    
                            need_replace = True

                        if need_replace:
                            # 找到外边界得像素值
                            # 将整个 region_mask 向外膨胀一圈
                            dilated = binary_dilation(region_mask)
                            outer_neighbors = dilated & ~region_mask
                            outer_vals = current_slice[outer_neighbors]

                            outer_vals_nonzero = outer_vals[outer_vals != 0]
                            unique_vals, counts = np.unique(outer_vals_nonzero, return_counts=True)
                            max_count_idx = np.argmax(counts)
                            most_freq_value = unique_vals[max_count_idx]

                            new_instance_id[z, :, :][region_mask] = most_freq_value

    return new_instance_id

def analyze_undersegmentation(instance_id, area_threshold=50, orientation_threshold=3.0, consecutive_slices=2):
    """
    检测 3D 分割结果中的欠分割实例：
      1) 逐层对每个实例计算“有效连通区域数”
         （先把同一层、同一实例的多个区域，基于“主轴方向”是否接近进行合并）
      2) 若同一个实例在连续 consecutive_slices 层都有“有效连通区域数” > 1，视为欠分割

    参数:
    ----------
    instance_id : np.ndarray
        3D 分割结果，形状假设为 [Z, H, W]
        (若你的数据是 [H, W, Z]，请自行调整遍历顺序)
    area_threshold : int
        面积阈值，小于此面积的连通区域会被忽略
    orientation_threshold : float
        主轴方向允许的差值阈值（单位“度”，非弧度），小于此差值则视为同一“有效”区域
    consecutive_slices : int
        需要连续多少层满足“有效连通区域数 > 1”才认定为欠分割。

    返回:
    ----------
    undersegmented_ids : list
        判定为欠分割的实例 ID 列表
    """

    # （1）先统计：instance_features_count[inst_id][z] = 当 z 层合并后有效连通区域数
    #     若没有出现（或有效区域=0），则默认=0。
    instance_features_count = {}

    Z = instance_id.shape[0]

    for z in range(Z):
        current_slice = instance_id[z, :, :]
        unique_ids = np.unique(current_slice)
        
        for inst_id in unique_ids:
            if inst_id == 0:
                continue  # 忽略背景

            # 获取该实例在当前层的二值掩码
            binary_mask = (current_slice == inst_id)

            # 先做连通区域标记
            labeled_mask, num_features = label(binary_mask)
            if num_features <= 1:
                # 无需合并就只有 0 或 1 个连通区域：若1个且面积够，则有效区域数=1，否则=0
                if num_features == 1:
                    # 检查面积是否达到阈值
                    props = regionprops(labeled_mask)
                    if props[0].area >= area_threshold:
                        effective_count = 1
                    else:
                        effective_count = 0
                else:
                    effective_count = 0
            else:
                # 当 num_features > 1 时，需要:
                #   1) 过滤掉面积 < area_threshold 的区域
                #   2) 对剩余区域根据 orientation 进行合并(若方向差小于 orientation_threshold 则归为同一类)
                effective_count = merge_connected_components(
                    labeled_mask, 
                    area_threshold=area_threshold, 
                    orientation_threshold=orientation_threshold
                )

            if inst_id not in instance_features_count:
                instance_features_count[inst_id] = {}
            instance_features_count[inst_id][z] = effective_count

    # （2）判断是否“连续 consecutive_slices 层都有有效连通区域数>1”
    undersegmented_ids = []
    for inst_id, slice_dict in instance_features_count.items():
        # slice_dict 形如 {z1: count1, z2: count2, ...}
        # 先取出所有 z 及对应有效连通数，并按 z 排序
        z_indices = sorted(slice_dict.keys())
        # 在这些 z 中找有没有“连续 consecutive_slices”满足 count>1
        if has_consecutive_slices(slice_dict, z_indices, consecutive=consecutive_slices):
            undersegmented_ids.append(inst_id)

    return undersegmented_ids


def merge_connected_components(labeled_mask, area_threshold=50, orientation_threshold=3.0):
    """
    先过滤掉面积 < area_threshold 的区域，
    再把剩余区域中“主轴方向（orientation）近似”的区域合并，得到最后的合并数。
    返回 合并后的有效连通区域数量 (int)。

    说明：
    - orientation_threshold 单位是“度”，需要先转弧度比较。
    - regionprops(labeled_mask) 返回的 orientation 是弧度制。
    """
    props_all = regionprops(labeled_mask)
    # 仅保留面积达标的区域
    props_valid = [p for p in props_all if p.area >= area_threshold]
    if len(props_valid) <= 1:
        # 若最终有效区域 <=1，返回其数值即可
        return len(props_valid)

    # 计算各区域的 orientation
    orientations = [p.orientation for p in props_valid]

    # 构建相似度图（orientation 差值 < 阈值）
    rad_th = np.deg2rad(orientation_threshold)  # 将度 -> 弧度
    N = len(props_valid)
    adjacency = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i+1, N):
            if abs(orientations[i] - orientations[j]) < rad_th:
                adjacency[i, j] = adjacency[j, i] = True

    # 对 adjacency 做连通分量分析（或 BFS / DFS），得到“合并后”的簇数量
    visited = set()
    cluster_count = 0

    for i in range(N):
        if i in visited:
            continue
        # DFS/BFS合并
        stack = [i]
        visited.add(i)
        while stack:
            top = stack.pop()
            for neighbor in range(N):
                if adjacency[top, neighbor] and (neighbor not in visited):
                    visited.add(neighbor)
                    stack.append(neighbor)
        cluster_count += 1

    return cluster_count


def has_consecutive_slices(slice_dict, z_indices, consecutive=3):
    """
    检查 slice_dict 中是否存在 连续 consecutive 层(索引连续)都有 count > 1。
    z_indices 为 slice_dict.keys() 的有序列表。
    
    slice_dict: {z: effective_count, ...}
    z_indices : sorted list of all z
    consecutive: 需要连续几层
    
    返回 bool
    """
    # 思路：在 z_indices 上滑动窗口，看对应 count 是否 > 1
    #      并且索引要连续，例如 [10,11,12] 才算3层连续
    # 假设 consecutive=3, 我们要在 z_indices 中找三连，比如 (z, z+1, z+2)
    # 并保证 slice_dict[z], slice_dict[z+1], slice_dict[z+2] 都>1
    n = len(z_indices)
    if n < consecutive:
        return False

    # 用双指针或简单遍历
    for i in range(n - consecutive + 1):
        # 检查从 i 到 i+consecutive-1 是否连续
        # z_indices[i], z_indices[i+1], ... z_indices[i+consecutive-1]
        start_z = z_indices[i]
        ok = True
        for offset in range(consecutive):
            z_val = start_z + offset
            # 判断 z_val 是否在 z_indices[i+offset]，且 slice_dict[z_val]>1
            if z_indices[i+offset] != z_val:
                ok = False
                break
            if slice_dict[z_val] <= 1:
                ok = False
                break

        if ok:
            # 找到一段连续 consecutive 层都有效连通区域数>1
            return True

    return False


def Undersegmentation_watershed(region_mask, prev_mask_value):
    """
    Refine the segmentation of a region using watershed algorithm.

    Parameters:
        region_mask (numpy.ndarray): Boolean mask of the current region.
        prev_instances_mask (numpy.ndarray): Mask of overlapping instances from the previous layer.

    Returns:
        numpy.ndarray: Refined region with new labels.
    """

    prev_distance = distance_transform_edt(prev_mask_value)
    threshold = np.percentile(prev_distance[prev_mask_value > 0], 25)  # 仅计算非零区域的阈值
    # 创建布尔掩码，仅保留前 50% 的值
    distance_mask = (prev_distance >= threshold)
    # 应用到 prev_mask_value，仅保留前 50% 的区域
    masked_prev_values = np.zeros_like(prev_mask_value)
    masked_prev_values[distance_mask] = prev_mask_value[distance_mask]


    # Compute the distance transform of the current region
    distance = distance_transform_edt(region_mask)
    # Use the previous instances as seeds for the watershed algorithm
    markers = masked_prev_values
    # Apply watershed segmentation
    refined_region = watershed(-distance, markers, mask=region_mask)

    return refined_region

def find_valid_overlaps(curr_mask, refined_volume, prev_z, overlap_ids, thres=0.25):     #这个值过大过小都不行
    """
    Identify valid overlaps where the current region overlaps more than 50% of previous instances.

    Parameters:
        curr_mask (numpy.ndarray): Boolean mask of the current region.
        refined_volume (numpy.ndarray): The refined volume so far.
        z (int): Current slice index.
        overlap_ids (list): List of overlapping instance IDs from the previous slice.

    Returns:
        list: valid overlap IDs.
    """
    valid_overlap_ids = []
    for overlap_id in overlap_ids:
        prev_region_mask = (refined_volume[prev_z, :, :] == overlap_id)
        overlap_area = np.sum(curr_mask & prev_region_mask)
        prev_region_area = np.sum(prev_region_mask)
        if prev_region_area == 0:
            continue
        if overlap_area / prev_region_area > thres:     # maybe need to modify 0117
            valid_overlap_ids.append(overlap_id)
    return valid_overlap_ids

def assign_refined_ids(refined_region, refined_volume, z, prev_z, overlap_ids, current_id):
    """
    Assign new IDs or match existing IDs to refined regions.

    Parameters:
        refined_region (numpy.ndarray): Refined segmentation regions.
        refined_volume (numpy.ndarray): The refined volume so far.
        z (int): Current slice index.
        overlap_ids (list): Overlap IDs from the previous slice.

    Returns:
        None: Updates refined_volume in place.
    """
    unique_refined_ids = np.unique(refined_region)
    for refined_id in unique_refined_ids:
        if refined_id == 0:
            continue  # Skip background

        region_mask = (refined_region == refined_id)
        overlap_ids, counts = np.unique(refined_volume[prev_z, :, :][region_mask], return_counts=True)
        valid_mask = overlap_ids > 0  # 创建布尔掩码，仅保留非背景值
        overlap_ids = overlap_ids[valid_mask]  # 筛选非背景的 ID
        counts = counts[valid_mask]  # 同时筛选出对应的计数

        if len(overlap_ids) > 0:
            max_overlap_id = overlap_ids[np.argmax(counts)]
            refined_volume[z, :, :][region_mask] = max_overlap_id
        else:
            # Assign a new ID
            refined_volume[z, :, :][region_mask] = current_id
            current_id += 1
    return current_id

def find_nonzero_range(volume):
    """
    Find the range of slices in a 3D volume that contain non-zero values.

    Parameters:
        volume (numpy.ndarray): 3D array representing the volume.

    Returns:
        tuple: (start_index, end_index), where:
            - start_index: First slice index with non-zero values.
            - end_index: Last slice index with non-zero values.
            - Returns (-1, -1) if all slices are zero.
    """
    start_index = -1
    end_index = -1
    for z in range(volume.shape[0]):
        if np.any(volume[z, :, :]):  # Check if the slice has any non-zero values
            if start_index == -1:  # First non-zero slice
                start_index = z
            end_index = z  # Update end index to the latest non-zero slice

    return start_index, end_index


def check_multiple_ids(refined_region):

    current_slice = refined_region.copy()  # 当前 Z 层的切片
    # 获取当前层的实例 ID（包括背景 0）
    unique_ids = np.unique(current_slice)

    # 标记当前切片中的连通区域
    curr_mask = (current_slice > 0)
    current_slice_regions, _ = label(curr_mask)

    # 遍历每个实例 ID
    for instance in unique_ids:
        # 跳过背景
        if instance == 0:
            continue

        # 生成当前实例的二值掩膜
        binary_mask = (current_slice == instance)
        
        # 对该实例掩膜进行连通区域分析
        labeled_mask, num_features = label(binary_mask)

        
        # 若有多个连通区域，合并""过分割"中的占比小的实例，用周围像素覆盖
        if num_features > 1:
            
            #####################################################
            area_ratios = []
            # need_replace = False
            # 计算每个连通区域的面积（像素数）
            for region_label in range(1, num_features + 1):
                
                region_mask = (labeled_mask == region_label)
                # 获取 overlapping_labels 
                overlapping_labels = np.unique(current_slice_regions[region_mask])
                mask = (current_slice_regions == overlapping_labels)
                total_area = mask.sum()
                area_ratios.append(region_mask.sum() / total_area)  
            
            max_ratio = max(area_ratios)
            max_index = area_ratios.index(max_ratio)

            for region_label in range(1, num_features + 1):
                if region_label == max_index + 1 :  # and max_ratio > 0.75
                    continue

                region_mask = (labeled_mask == region_label)
                # 找到外边界得像素值
                # 将整个 region_mask 向外膨胀一圈
                dilated = binary_dilation(region_mask)
                outer_neighbors = dilated & ~region_mask
                outer_vals = current_slice[outer_neighbors]

                outer_vals_nonzero = outer_vals[outer_vals != 0]
                unique_vals, counts = np.unique(outer_vals_nonzero, return_counts=True)
                max_count_idx = np.argmax(counts)
                most_freq_value = unique_vals[max_count_idx]

                current_slice[region_mask] = most_freq_value

    refined_region = current_slice
    return refined_region


def normalize_angle(angle):
    """
    将角度归一化到 [0, π) 内，消除 180° 对称性影响。
    """
    return angle % np.pi

def distance_between_parallel_lines(p1, p2, d):
    """
    计算两条平行直线之间的距离：
      直线1：过 p1，方向为 d
      直线2：过 p2，方向为 d
    公式为：diff = p2 - p1，在 d 的垂直方向上的分量的模长。
    """
    diff = p2 - p1
    # d 的垂直方向（2D中旋转 90°）
    perp = np.array([-d[1], d[0]])
    return np.abs(np.dot(diff, perp))

def merge_if_collinear(refined_region, orientation_threshold_deg=3, line_distance_threshold=10):
    """
    针对同一连通区域内恰好包含两个不同非0实例 ID 的情况：
      1. 判断两个子区域的长轴方向之差是否小于 orientation_threshold_deg；
      2. 若方向接近，则构造两条直线（每条直线：质心和对应长轴方向），
         计算两直线之间的垂直距离，如果距离小于 line_distance_threshold，
         则认为共线，将整个连通区域统一赋值为面积较大的那个实例 ID。
         
    参数：
      refined_region: 2D numpy 数组，不同非0数值代表不同实例（背景为0）。
      orientation_threshold_deg: 长轴方向差异的角度阈值（单位：度）。
      line_distance_threshold: 两条直线之间的最大垂直距离阈值（单位：像素）。
      
    返回：
      合并后的 refined_region 数组。
    """
    # 对非背景部分进行连通区域标记
    labeled_regions, num_labels = label(refined_region > 0)
    
    # 遍历每个连通区域
    for region_label in range(1, num_labels + 1):
        region_mask = (labeled_regions == region_label)
        # 获取该区域内所有非0的实例 ID
        instance_ids = np.unique(refined_region[region_mask])
        instance_ids = instance_ids[instance_ids != 0]
        if len(instance_ids) != 2:
            continue  # 只处理恰好包含两个非0数值的情况
        
        # 分别提取两个实例的区域属性
        props_dict = {}
        for inst in instance_ids:
            mask_inst = (refined_region == inst) & region_mask
            # 对该子区域做 label（通常只有一个连通部分）
            labeled_inst, _ = label(mask_inst)
            props = regionprops(labeled_inst)
            if len(props) == 0:
                continue
            # 选择面积最大的区域（如果有多个）
            prop = max(props, key=lambda p: p.area)
            props_dict[inst] = prop
        
        # 若无法获得两个区域的属性，则跳过
        if len(props_dict) != 2:
            continue
        
        # 取出各区域的 orientation、centroid 和 area
        inst1, inst2 = instance_ids[0], instance_ids[1]
        prop1 = props_dict[inst1]
        prop2 = props_dict[inst2]
        
        orient1 = normalize_angle(prop1.orientation)
        orient2 = normalize_angle(prop2.orientation)
        
        # 计算方向差值，注意由于角度归一化到 [0, π)，需要考虑对称性
        angle_diff = np.abs(orient1 - orient2)
        if angle_diff > np.pi/2:
            angle_diff = np.pi - angle_diff
            
        # 判断方向差异是否小于阈值（转换为弧度）
        if angle_diff > np.deg2rad(orientation_threshold_deg):
            # 长轴方向差异较大，不处理
            continue
        
        # 构造两条直线：
        # 直线: p = centroid + t * d，其中 d = (cos(orientation), sin(orientation))
        centroid1 = np.array(prop1.centroid)
        centroid2 = np.array(prop2.centroid)
        d1 = np.array([np.cos(orient1), np.sin(orient1)])
        # 这里假设两直线方向相似，使用 d1 计算直线间距离
        line_dist = distance_between_parallel_lines(centroid1, centroid2, d1)
        
        if line_dist < line_distance_threshold:
            # 认为两直线共线，选择面积较大的实例进行合并
            if prop1.area >= prop2.area:
                main_id = inst1
            else:
                main_id = inst2
            refined_region[region_mask] = main_id
            print(f"区域 {region_label} 合并为 {main_id} (角度差 {np.rad2deg(angle_diff):.1f}°, 直线距离 {line_dist:.2f} 像素)")
        else:
            print(f"区域 {region_label} 不合并 (角度差 {np.rad2deg(angle_diff):.1f}°, 直线距离 {line_dist:.2f} 像素)")
            
    return refined_region


def find_slice_with_max_count(volume, start_index, end_index, area_threshold=50):
    """
    在 volume 的切片范围 [start_index, end_index] 内，查找面积大于 area_threshold 
    的连通区域个数最多的切片，并返回：
      - max_z：该切片索引
      - max_count：该切片中面积 > area_threshold 的连通区域总数

    参数：
      volume        : 3D 图像数据（形如 [Z, H, W]）
      start_index   : 要遍历的起始 z 索引
      end_index     : 要遍历的结束 z 索引
      area_threshold: 面积阈值，默认 100

    返回：
      max_z, max_count
    """

    max_count = -1   # 用于记录最大区域数，初始化为 -1
    max_z = -1       # 对应最大区域数出现的 z 值

    for z in range(start_index, end_index + 1):
        current_slice = volume[z]
        labeled_slice, num_features = label(current_slice)

        # 统计当前切片中，面积 > area_threshold 的连通区域数
        count_over_threshold = 0
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_slice == region_id)
            area = np.sum(region_mask)
            if area > area_threshold:
                count_over_threshold += 1

        # 如果当前切片的该类区域数超过已记录的最大值，则更新
        if count_over_threshold > max_count:
            max_count = count_over_threshold
            max_z = z

    return max_z, max_count

def segment_3d_split_both_ways(volume, current_id, area_threshold=50):
    """
    分别从 mid_idx->end_idx(正向) 和 mid_idx->start_idx(逆向) 两次遍历，并分配 ID。
    """
    # 1) 初始化结果
    refined_volume = np.zeros_like(volume, dtype=np.int16)
    start_index, end_index = find_nonzero_range(volume)

    max_z, max_count = find_slice_with_max_count(volume, start_index, end_index)
    mid_idx = max_z
    print('mid_idx', mid_idx)
    # 2) 正向遍历 mid_idx -> end_idx
    for z in range(mid_idx, end_index+1):
        refined_volume, current_id = process_slice(
            volume, refined_volume, z,
            prev_z=(z - 1),          # 正向时，“上一层”就是 z-1
            current_id=current_id,
            area_threshold=area_threshold,
            start_index=mid_idx     # 在正向时，可以将 mid_idx 当作“首层”来处理
        )

    # 3) 逆向遍历 (mid_idx - 1) -> start_index
    #    注意：从 mid_idx - 1 开始，避免重复处理 mid_idx 这一层
    for z in range(mid_idx - 1, start_index - 1, -1):
        refined_volume, current_id = process_slice(
            volume, refined_volume, z,
            prev_z=(z + 1),         # 逆向时，“上一层”在索引上是 z+1
            current_id=current_id,
            area_threshold=area_threshold,
            start_index=start_index-1       # start_index
        )

    return refined_volume, current_id


def process_slice(volume, refined_volume, z, prev_z, current_id,
                  area_threshold, start_index):
    """
    封装“处理单个切片”的逻辑，包括：
    1) 寻找连通区域
    2) 筛选面积
    3) 与前一层重叠判断 / 分配 ID / 欠分割处理

    参数:
      - z: 当前切片索引
      - prev_z: “上一层”的索引 (正向时是 z-1, 逆向时是 z+1)
      - start_index: 正向或逆向时，可能作为“首层”参考
    """
    current_slice = volume[z]       
    labeled_slice, num_features = label(current_slice)
    for region_id in range(1, num_features + 1):
        # Compute mask for the current region
        curr_mask = (labeled_slice == region_id)
        # Check region area against the threshold
        region_area = np.sum(curr_mask)
        if region_area < area_threshold:
            continue  # Skip small regions

        # Compute overlap with previous slice instances
        if z > start_index:
            # pre slice overlap ids
            overlap_ids, counts = np.unique(refined_volume[prev_z, :, :][curr_mask], return_counts=True)
            valid_mask = overlap_ids > 0  # 创建布尔掩码，仅保留非背景值
            overlap_ids = overlap_ids[valid_mask]  # 筛选非背景的 ID
            counts = counts[valid_mask]  # 同时筛选出对应的计数
            
            if len(overlap_ids) > 0:
                # Track overlaps for fusion detection
                # Check for undersegmentation (multiple instances overlap by >50%)
                valid_overlap_ids = find_valid_overlaps(curr_mask, refined_volume, prev_z, overlap_ids)
                
                if len(valid_overlap_ids) > 1:
                    # Undersegmentation detected, refine region
                    prev_instances_mask = np.isin(refined_volume[prev_z, :, :], valid_overlap_ids)
                    prev_mask_value = np.zeros_like(refined_volume[prev_z, :, :])
                    prev_mask_value[prev_instances_mask] = refined_volume[prev_z, :, :][prev_instances_mask]
                    refined_region = Undersegmentation_watershed(curr_mask, prev_mask_value)

                    # check_multiple_ids
                    refined_region = check_multiple_ids(refined_region)
                    refined_region = merge_if_collinear(refined_region)

                    # Assign IDs to refined regions
                    current_id = assign_refined_ids(refined_region, refined_volume, z, prev_z, overlap_ids, current_id)
                    continue
                
                # Find the best matching instance based on overlap size
                max_overlap_id = overlap_ids[np.argmax(counts)]
                area_pre_max_id_mask = (refined_volume[prev_z, :, :] == max_overlap_id).sum()
                if region_area < 500 or (region_area / area_pre_max_id_mask) < 1.5:     # thres=1.25
                    # Assign the ID from the previous slice
                    refined_volume[z, :, :][curr_mask] = max_overlap_id
                else:
                    # Undersegmentation detected, refine region
                    prev_instances_mask = np.isin(refined_volume[prev_z, :, :], overlap_ids)
                    prev_mask_value = np.zeros_like(refined_volume[prev_z, :, :])
                    prev_mask_value[prev_instances_mask] = refined_volume[prev_z, :, :][prev_instances_mask]
                    refined_region = Undersegmentation_watershed(curr_mask, prev_mask_value)
                    # check_multiple_ids
                    refined_region = check_multiple_ids(refined_region)
                    refined_region = merge_if_collinear(refined_region)
                    # Assign IDs to refined regions
                    current_id = assign_refined_ids(refined_region, refined_volume, z, prev_z, overlap_ids, current_id)

            else:
                # Assign a new ID
                refined_volume[z, :, :][curr_mask] = current_id
                current_id += 1
        else:
            # Assign new IDs for the first slice
            refined_volume[z, :, :][curr_mask] = current_id
            current_id += 1
    return refined_volume, current_id

def solve_undersegmentation(instance_id):
    """
    Resolve undersegmentation issues in a 3D instance volume.

    Parameters:
        instance_id (numpy.ndarray): 3D array containing instance IDs.
        undersegmented_ids (list): List of IDs identified as undersegmented.

    Returns:
        numpy.ndarray: Updated instance array with resolved undersegmentation.
    """
    new_instance_id = remove_small_segments(instance_id)
    undersegmented_ids = analyze_undersegmentation(new_instance_id)
    updated_instance_id = new_instance_id.copy()

    for target_id in undersegmented_ids:
        # Generate a new ID starting from the maximum ID + 1
        current_id = np.max(updated_instance_id) + 1

        # Create a mask for the target_id
        mask = (updated_instance_id == target_id).astype(np.uint8)
        print(target_id)
        # Refine the undersegmented region
        refined_volume, current_id = segment_3d_split_both_ways(mask, current_id)

        # Remove the target_id regions by setting them to 0
        updated_instance_id[updated_instance_id == target_id] = 0

        # Add non-zero regions from refined_volume to updated_instance_id
        non_zero_mask = refined_volume > 0
        updated_instance_id[non_zero_mask] = refined_volume[non_zero_mask]

    return updated_instance_id



