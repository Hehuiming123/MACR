# import os
# import pandas as pd

# def generate_nabirds_csv(root_dir, output_dir):
#     """
#     生成 NABirds 数据集的训练集和测试集 CSV 文件
#     格式：
#         ,img_name,label
#         0,n02085620-Chihuahua/n02085620_2650.jpg,1
#         1,n02085620-Chihuahua/n02085620_2651.jpg,1
#         ...
#     """
#     # 加载原始元数据
#     dataset_path = os.path.join(root_dir, 'nabirds')
    
#     # 读取 images.txt（图片ID和路径）
#     image_paths = pd.read_csv(
#         os.path.join(dataset_path, 'images.txt'), 
#         sep=' ', 
#         names=['img_id', 'filepath']
#     )
    
#     # 读取 image_class_labels.txt（原始标签）
#     image_class_labels = pd.read_csv(
#         os.path.join(dataset_path, 'image_class_labels.txt'),
#         sep=' ',
#         names=['img_id', 'raw_label']
#     )
    
#     # 读取 train_test_split.txt（划分标识）
#     train_test_split = pd.read_csv(
#         os.path.join(dataset_path, 'train_test_split.txt'),
#         sep=' ',
#         names=['img_id', 'is_training_img']
#     )
    
#     # 合并数据
#     merged_data = image_paths.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
    
#     # 生成连续标签映射（与 Dataset 中的逻辑一致）
#     unique_labels = merged_data['raw_label'].unique()
#     label_map = {old_label: idx for idx, old_label in enumerate(unique_labels)}
#     merged_data['label'] = merged_data['raw_label'].map(label_map)
    
#     # 分离训练集和测试集
#     train_data = merged_data[merged_data['is_training_img'] == 1]
#     test_data = merged_data[merged_data['is_training_img'] == 0]
    
#     # 生成最终CSV格式
#     for data_split, split_name in [(train_data, 'train'), (test_data, 'test')]:
#         df = data_split[['filepath', 'label']].rename(columns={'filepath': 'img_name'})
#         df.to_csv(
#             os.path.join(output_dir, f'nabirds_{split_name}.csv'), 
#             index=True, 
#             index_label='',
#             header=['img_name', 'label']
#         )
        
#     print(f"CSV文件已生成到目录: {output_dir}")

# # 使用示例
# generate_nabirds_csv(
#     root_dir='dataset/',  # 替换为实际数据集根目录
#     output_dir='dataset/nabirds'                # CSV输出目录
# )

# import os
# import pandas as pd

# def generate_nabirds_csv(root_dir, output_dir):
#     """
#     生成 NABirds 数据集的训练集和测试集 CSV 文件
#     格式：
#         ,img_name,label
#         0,n02085620-Chihuahua/n02085620_2650.jpg,1
#         1,n02085620-Chihuahua/n02085620_2651.jpg,1
#         ...
#     """
#     # 加载原始元数据
#     dataset_path = os.path.join(root_dir, 'nabirds')
    
#     # 读取 images.txt（图片ID和路径）
#     image_paths = pd.read_csv(
#         os.path.join(dataset_path, 'images.txt'), 
#         sep=' ', 
#         names=['img_id', 'filepath']
#     )
    
#     # 读取 image_class_labels.txt（原始标签）
#     image_class_labels = pd.read_csv(
#         os.path.join(dataset_path, 'image_class_labels.txt'),
#         sep=' ',
#         names=['img_id', 'raw_label']
#     )
    
#     # 读取 train_test_split.txt（划分标识）
#     train_test_split = pd.read_csv(
#         os.path.join(dataset_path, 'train_test_split.txt'),
#         sep=' ',
#         names=['img_id', 'is_training_img']
#     )
    
#     # 合并数据
#     merged_data = image_paths.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
    
#     # 生成连续标签映射（按原始标签排序）
#     unique_labels = sorted(merged_data['raw_label'].unique())  # 新增排序
#     label_map = {old_label: idx for idx, old_label in enumerate(unique_labels)}
#     merged_data['label'] = merged_data['raw_label'].map(label_map)
    
#     # 分离训练集和测试集并重置索引
#     train_data = merged_data[merged_data['is_training_img'] == 1].reset_index(drop=True)  # 关键修改
#     test_data = merged_data[merged_data['is_training_img'] == 0].reset_index(drop=True)    # 关键修改
    
#     # 生成最终CSV格式
#     for data_split, split_name in [(train_data, 'train'), (test_data, 'test')]:
#         df = data_split[['filepath', 'label']].rename(columns={'filepath': 'img_name'})
#         df.to_csv(
#             os.path.join(output_dir, f'nabirds_{split_name}.csv'), 
#             index=True, 
#             index_label='',
#             header=['img_name', 'label']
#         )
        
#     print(f"CSV文件已生成到目录: {output_dir}")

# # 使用示例
# generate_nabirds_csv(
#     root_dir='dataset/',  # 替换为实际数据集根目录
#     output_dir='dataset/nabirds'  # CSV输出目录
# )


import os
import pandas as pd

def generate_nabirds_csv(root_dir, output_dir):
    """
    生成 NABirds 数据集的训练集和测试集 CSV 文件
    格式：
        ,img_name,label
        0,n02085620-Chihuahua/n02085620_2650.jpg,1
        1,n02085620-Chihuahua/n02085620_2651.jpg,1
        ...
    """
    # 加载原始元数据
    dataset_path = os.path.join(root_dir, 'nabirds')
    
    # 读取 images.txt（图片ID和路径）
    image_paths = pd.read_csv(
        os.path.join(dataset_path, 'images.txt'), 
        sep=' ', 
        names=['img_id', 'filepath']
    )
    
    # 读取 image_class_labels.txt（原始标签）
    image_class_labels = pd.read_csv(
        os.path.join(dataset_path, 'image_class_labels.txt'),
        sep=' ',
        names=['img_id', 'raw_label']
    )
    
    # 读取 train_test_split.txt（划分标识）
    train_test_split = pd.read_csv(
        os.path.join(dataset_path, 'train_test_split.txt'),
        sep=' ',
        names=['img_id', 'is_training_img']
    )
    
    # 合并数据
    merged_data = image_paths.merge(image_class_labels, on='img_id').merge(train_test_split, on='img_id')
    
    # 生成连续标签映射（标签从1开始）
    unique_labels = sorted(merged_data['raw_label'].unique())
    label_map = {old_label: idx+1 for idx, old_label in enumerate(unique_labels)}  # 修改点：+1
    merged_data['label'] = merged_data['raw_label'].map(label_map)
    
    # 分离训练集和测试集并重置索引
    train_data = merged_data[merged_data['is_training_img'] == 1].reset_index(drop=True)
    test_data = merged_data[merged_data['is_training_img'] == 0].reset_index(drop=True)
    
    # 生成最终CSV格式
    for data_split, split_name in [(train_data, 'train'), (test_data, 'test')]:
        df = data_split[['filepath', 'label']].rename(columns={'filepath': 'img_name'})
        df.to_csv(
            os.path.join(output_dir, f'nabirds_{split_name}.csv'), 
            index=True, 
            index_label='',
            header=['img_name', 'label']
        )
        
    print(f"CSV文件已生成到目录: {output_dir}")

# 使用示例
generate_nabirds_csv(
    root_dir='dataset/',  # 替换为实际数据集根目录
    output_dir='dataset/nabirds'  # CSV输出目录
)