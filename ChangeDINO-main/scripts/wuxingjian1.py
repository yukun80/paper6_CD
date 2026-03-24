import scipy.io as sio

# 读取你提供的样本 mat 文件
template_data = sio.loadmat("/home/yukun/codes/paper6_waterlogging/datasets/S1_Henan/dataset#3.mat")

print("dataset#3.mat 包含的变量及维度如下：")
for key, value in template_data.items():
    if not key.startswith("__"):  # 过滤掉 MATLAB 内置的全局属性
        print(f"变量名 (Key): '{key}', 维度 (Shape): {value.shape}, 数据类型 (Dtype): {value.dtype}")
