# from scipy.io import loadmat
# features_struct=loadmat( r'dataset/dog/test_list.mat' )
# print(features_struct)
import pandas as pd
import scipy
from scipy import io
import os
#遍历文件夹
# for dirname, _, filenames in os.walk('dataset/dog'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#         # print(filename)
#         # print(os.path.realpath(filename))  # 获取当前文件路径
#         print(os.path.dirname(os.path.realpath(filename)))  # 从当前文件路径中获取目录
#         # print(os.path.basename(os.path.realpath(filename)))  # 获取文件名
#         (file, ext) = os.path.splitext(os.path.realpath(filename))
#         # print(file)
#         print(os.path.basename(os.path.realpath(file)))  # 获取文件名
#         # print(ext)
#         print(dirname)


#         path = os.path.join(dirname, filename)
#         # 1、导入文件
#         matfile = scipy.io.loadmat(path)
#         # 2、加载数据
#         datafile = list(matfile.values())[-1]
#         # 3、构造一个表的数据结构，data为表中的数据
#         dfdata = pd.DataFrame(data=datafile)
#         # 4、保存为.csv格式的路径
#         datapath = dirname+'\\'+os.path.basename(os.path.realpath(file))+'.csv'
#         # 5、保存为.txt格式的路径
#         dfdata.to_csv(datapath, index=False)

# from scipy.io import loadmat
# import numpy as np

# # 加载 .mat 文件
# mat_data = loadmat(r'dataset/dog/test_list.mat')

# # 打印所有顶层键
# print("所有顶层键:", mat_data.keys())

# # 遍历每个键并打印详细信息
# for key in mat_data:
#     # 忽略 MATLAB 的元数据键（如 '__header__', '__version__'）
#     if not key.startswith('__'):
#         data = mat_data[key]
#         print(f"\n变量 '{key}':")
#         print("类型:", type(data))
#         print("形状:", data.shape)
        
#         # 如果是字符数组或数值数组，直接打印示例内容
#         if isinstance(data, np.ndarray):
#             print("示例内容:")
#             if data.dtype == 'object':
#                 # 处理对象数组（如 cell 或结构体）
#                 print(data[0][0])
#             else:
#                 print(data[:2])  # 打印前两个元素

from scipy.io import loadmat
import pandas as pd
import numpy as np

# 加载.mat文件
mat_data = loadmat(r'dataset/dog/test_list.mat')

# 提取并处理数据
file_paths = [item[0][0] for item in mat_data['file_list'].ravel()]  # 解包嵌套结构
labels = mat_data['labels'].ravel().astype(int)  # 展平为一维整数数组

# 创建带有序号索引的DataFrame
df = pd.DataFrame(
    data=np.column_stack((file_paths, labels)),
    columns=['img_name', 'label']
)

# 生成符合要求的CSV（序号自动生成，不显示列名）
df.to_csv(
    'dog_test_list.csv',
    index=True,         # 生成序号列
    index_label='',     # 序号列不显示列名
    header=['img_name', 'label']  # 修正列名顺序
)

print(f"CSV已生成，共 {len(df)} 条数据")

from scipy.io import loadmat
import pandas as pd
import numpy as np

# 加载 .mat 文件
mat_data = loadmat(r'dataset/dog/train_list.mat')

# 提取 file_list 和 labels
file_list = [item[0] for item in mat_data['file_list'].ravel()]  # 解包 MATLAB cell 数组
labels = mat_data['labels'].ravel().astype(int)  # 展平为一维数组并转为整数

# 创建 DataFrame 并添加序号
df = pd.DataFrame({
    "img_name": file_list,
    "label": labels
})

# 保存为 CSV（自动生成序号列，不显示列名）
df.to_csv("output.csv", index=True, index_label="", header=["img_name", "label"], encoding='utf-8-sig')

print("CSV 生成完成！前 5 行示例如下：")
print(df.head())