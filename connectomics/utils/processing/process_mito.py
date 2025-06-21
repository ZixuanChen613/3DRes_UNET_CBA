# Post-processing functions of mitochondria instance segmentation model outputs
# as described in "MitoEM Dataset: Large-scale 3D Mitochondria Instance Segmentation 
# from EM Images (MICCAI 2020, https://donglaiw.github.io/page/mitoEM/index.html)".
import numpy as np

from skimage.measure import label
from skimage.transform import resize
from skimage.morphology import dilation
from skimage.segmentation import watershed

from .utils import remove_small_instances

def binary_connected(volume, thres=0.9, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    """From binary foreground probability map to instance masks via
    connected-component labeling.
    Args: 
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres (float): threshold of foreground. Default: 0.8
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    foreground = (semantic > int(255*thres))
    segm = label(foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def binary_watershed(volume, thres1=0.98, thres2=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                     remove_small_mode='background'):
    """From binary foreground probability map to instance masks via
    watershed segmentation algorithm.
    Args: 
        volume (numpy.ndarray): foreground probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.98
        thres2 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    seed_map = semantic > int(255*thres1)
    foreground = semantic > int(255*thres2)
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def bc_connected(volume, thres1=0.8, thres2=0.5, thres_small=128, scale_factors=(1.0, 1.0, 1.0), 
                 dilation_struct=(1,5,5), remove_small_mode='background'):
    """From binary foreground probability map and instance contours to 
    instance masks via connected-component labeling.
    Note:
        The instance contour provides additional supervision to distinguish closely touching
        objects. However, the decoding algorithm only keep the intersection of foreground and 
        non-contour regions, which will systematically result in imcomplete instance masks.
        Therefore we apply morphological dilation (check :attr:`dilation_struct`) to enlarge 
        the object masks.
    Args: 
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of foreground. Default: 0.8
        thres2 (float): threshold of instance contours. Default: 0.5
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        dilation_struct (tuple): the shape of the structure for morphological dilation. Default: :math:`(1, 5, 5)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    boundary = volume[1]
    foreground = (semantic > int(255*thres1)) * (boundary < int(255*thres2))

    segm = label(foreground)
    struct = np.ones(dilation_struct)
    segm = dilation(segm, struct)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)

def bc_watershed(volume, thres1=0.9, thres2=0.8, thres3=0.85, thres_small=128, scale_factors=(1.0, 1.0, 1.0),
                 remove_small_mode='background'):
    """From binary foreground probability map and instance contours to 
    instance masks via watershed segmentation algorithm.
    Args: 
        volume (numpy.ndarray): foreground and contour probability of shape :math:`(C, Z, Y, X)`.
        thres1 (float): threshold of seeds. Default: 0.9
        thres2 (float): threshold of instance contours. Default: 0.8
        thres3 (float): threshold of foreground. Default: 0.85
        thres_small (int): size threshold of small objects to remove. Default: 128
        scale_factors (tuple): scale factors for resizing in :math:`(Z, Y, X)` order. Default: :math:`(1.0, 1.0, 1.0)`
        remove_small_mode (str): ``'background'`` or ``'neighbor'``. Default: ``'background'``
    """
    semantic = volume[0]
    boundary = volume[1]
    seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2)) # seed , not contours
    foreground = (semantic > int(255*thres3))
    seed = label(seed_map)
    segm = watershed(-semantic, seed, mask=foreground)
    segm = remove_small_instances(segm, thres_small, remove_small_mode)

    if not all(x==1.0 for x in scale_factors):
        target_size = (int(semantic.shape[0]*scale_factors[0]), 
                       int(semantic.shape[1]*scale_factors[1]), 
                       int(semantic.shape[2]*scale_factors[2]))
        segm = resize(segm, target_size, order=0, anti_aliasing=False, preserve_range=True)
    return segm.astype(np.uint32)
    
import waterz
import malis

def malis_watershed(seed_map, thres1=0.9, thres2=0.8):
    if isinstance(seed_map, list):
        semantic = seed_map[0]
        boundary = seed_map[1]
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2)) 
    elif isinstance(seed_map, np.ndarray):
        seed_map = seed_map
    else:
        raise RuntimeError("seed map is wrong!")    
    # generate affinity
    output_mixs = seed_map.astype(np.int32)
    affs = malis.seg_to_affgraph(output_mixs, malis.mknhood3d())
    del output_mixs
    affs = affs.astype(np.float32)
    
    # initial watershed + agglomerate
    seg = list(waterz.agglomerate(affs, [0.50]))[0]
    del affs
    seg = seg.astype(np.uint16)
    
    # grow boundary
    seg = dilation(seg, np.ones((1,7,7)))
    seg = remove_small_instances(seg)
    
    return seg



from scipy.ndimage import distance_transform_edt
from skimage.morphology import dilation, erosion, ball

# 定义连接小实例到大实例的函数
def connect_small_to_large(labels, small_instances, large_instances, dilated_range):
    for label_id in small_instances:
        mask = labels == label_id  # 创建小实例掩码
        mask_o = mask
        
        # 计算小实例到所有大实例的距离
        dist_transform = distance_transform_edt(~mask)
        connected = False
        
        for _ in range(dilated_range):  # 设置一个膨胀范围的限制（避免无限循环）
            dilated_mask = dilation(mask, ball(1))  # 每次仅膨胀一层
            
            # 找出膨胀区域接触到的标签
            neighboring_labels = np.unique(labels[dilated_mask & ~mask])
            neighboring_labels = [label for label in neighboring_labels if label in large_instances]
            
            if neighboring_labels:
                # 找到距离最近的大实例标签
                target_label = neighboring_labels[0]
                # 找出小实例到大实例的最小距离点
                large_mask = labels == target_label
                large_dist_transform = distance_transform_edt(~large_mask)
                
                # 找到最短路径点
                path_mask = (dist_transform + large_dist_transform) == np.min(dist_transform + large_dist_transform)
                
                # 更新标签，将路径连接到大实例
                labels[path_mask] = target_label
                # 更新整个小实例区域为大实例标签
                labels[mask_o] = target_label
                connected = True
                break
            else:
                # 继续膨胀
                mask = dilated_mask

        # 若未成功连接，则可根据需求进行额外处理
        if not connected:
            print(f"实例 {label_id} 未能在限制范围内连接到大实例")

    return labels


from skimage import measure
from scipy import ndimage as ndi

def dis_watershed_test(seed_map, thres1=0.9, thres2=0.04, threshold=230):

    # 生成 seed_map
    if isinstance(seed_map, list):
        semantic = seed_map[0]
        boundary = seed_map[1]
        seed_map = (semantic > int(255*thres1)) * (boundary < int(255*thres2)) 
    elif isinstance(seed_map, np.ndarray):
        seed_map = seed_map
    else:
        raise RuntimeError("seed map is wrong!")  
    

    seed_points = measure.label(seed_map)

    # 生成距离变换图
    binary_semantic_mask = (semantic >= threshold).astype(int)
    distance = distance_transform_edt(binary_semantic_mask)

    # 第一次分水岭分割
    labels = watershed(-distance, seed_points, mask=binary_semantic_mask, watershed_line=True)

    # 生成唯一实例 ID
    instance_id = measure.label(labels, connectivity=1)     # modify


    # 获取实例数量（背景为0的区域不计入实例数量）
    num_instances = np.max(instance_id)
    print(f"Detected {num_instances} instances.")
    return instance_id

