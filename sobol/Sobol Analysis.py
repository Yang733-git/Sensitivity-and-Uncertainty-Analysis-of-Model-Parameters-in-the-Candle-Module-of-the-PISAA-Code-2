# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun'] # 宋体，黑体：SimHei
plt.rcParams['axes.unicode_minus'] = False   # 解决标题中文显示问题

# ==================== 配置区 ====================
problem = {
    # 必须与采样时的参数定义完全一致
    'names': ['K', 'ρ', 'T_m', 'H', 'μ', 'sigma', 'gamma', 
             'alph', 'L', 'D_h', 'm', 'Tup', 'Tdown', 'Ap', 'An'],
    'bounds': [
        [3.5, 4.0],        # 与采样代码保持一致
        [9000, 10000],     # 各参数范围必须严格对应
        [1700, 1900],
        [2e5, 4e5],
        [0.08, 0.12],
        [0.3, 0.4],
        [0.025, 0.035],
        [3e-6, 4e-6],
        [0.7, 0.8],
        [2.5, 2.8],
        [2300, 2400],
        [2900, 3100],
        [2700, 2900],
        [0.8, 1.0],
        [1.7, 2.0]
    ],
    'num_vars': 15
}

# 数据提取配置
config = {
    'result_dir': 'C:/Users/11201/Desktop/sensities/Candle/1/',    # 结果文件存放路径
    'target_node': '(0,0)',       # 要提取的节点坐标，与文件内容中的括号格式一致
    'target_time_index': 2,       # 选择的时间点索引(从0开始)
    'target_variable': 'mFrozen'    # 要分析的输出变量名称(需包含在文件名中)
}

# ==================== 数据加载 ====================
def extract_sample_id(filename):
    """从文件名提取样本序号的正则表达式方法"""
    match = re.match(r'^(\d+)', filename)  # 匹配开头的数字部分
    return int(match.group(1)) if match else -1

def load_results(config):
    """加载所有结果文件并提取目标值"""
    sample_values = []
    
    # 遍历结果目录
    for filename in sorted(os.listdir(config['result_dir']), 
                          key=extract_sample_id):
        if not filename.startswith(tuple(str(i) for i in range(10))):
            continue  # 跳过非数字开头的文件
            
        # 验证文件名包含目标变量
        if config['target_variable'] not in filename:
            continue
            
        # 提取样本ID
        sample_id = extract_sample_id(filename)
        if sample_id == -1:
            continue
            
        # 构建完整文件路径
        filepath = os.path.join(config['result_dir'], filename)
        
        # 读取文件内容
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # 解析时间行
        time_header = lines[0].split()
        try:
            time_point = float(time_header[config['target_time_index'] + 1])
        except IndexError:
            print(f"文件 {filename} 时间索引越界")
            continue
            
        # 查找目标节点数据行
        target_prefix = f"{config['target_variable']}{config['target_node']}"
        data_line = None
        for line in lines[1:]:
            if line.startswith(target_prefix):
                data_line = line
                break
                
        if not data_line:
            print(f"文件 {filename} 未找到目标节点数据")
            continue
            
        # 提取具体数值
        try:
            values = list(map(float, data_line.split()[1:]))
            target_value = values[config['target_time_index']]
            sample_values.append((sample_id, target_value))
        except Exception as e:
            print(f"文件 {filename} 数据解析失败: {str(e)}")
            continue
            
    # 按样本ID排序并转换为数组
    sample_values.sort(key=lambda x: x[0])
    Y = np.array([v for _, v in sample_values])
    
    return Y

# ==================== 执行分析 ====================
# 加载输出结果
Y = load_results(config)

# 验证数据完整性
if len(Y) == 0:
    raise ValueError("未加载到有效数据，请检查配置参数")
print(f"成功加载 {len(Y)} 个有效样本输出")

# 执行Sobol分析
Si = sobol.analyze(
    problem, 
    Y,
    calc_second_order=False,  # 与采样时设置一致
    print_to_console=False
)

# ==================== 结果处理 ====================
# 构建结果数据框架
results_df = pd.DataFrame({
    '参数': problem['names'],
    '一阶指数(Si)': Si['S1'],
    '总效应指数(STi)': Si['ST'],
    '置信区间下限(Si)': Si['S1_conf'],
    '置信区间上限(STi)': Si['ST_conf']
})

# 输出关键参数排序
print("\n重要参数排序(按总效应指数):")
print(results_df.sort_values('总效应指数(STi)', ascending=False)[['参数', '总效应指数(STi)']])

# ==================== 可视化 ====================
plt.figure(figsize=(12, 6))

# 一阶指数
plt.subplot(121)
plt.barh(results_df['参数'], results_df['一阶指数(Si)'], 
        xerr=results_df['置信区间下限(Si)'], 
        color='skyblue', edgecolor='black')
plt.xlabel('一阶敏感性指数')
plt.gca().invert_yaxis()

# 总效应指数
plt.subplot(122)
plt.barh(results_df['参数'], results_df['总效应指数(STi)'], 
        xerr=results_df['置信区间上限(STi)'], 
        color='lightgreen', edgecolor='black')
plt.xlabel('总效应敏感性指数')
plt.tight_layout()

# 保存图表
plt.savefig('sobol_analysis.png', dpi=300)
plt.show()
