# -*- coding: utf-8 -*-
import numpy as np
from SALib.sample import saltelli
import json  # 新增json模块用于解析参数文件

# 定义参数文件路径 (Windows路径使用双引号)
PARAM_FILE = "C:\\Users\\11201\\Desktop\\sensities\\sobol\\sobol_parameter.txt"

# 读取参数文件并解析JSON
try:
    with open(PARAM_FILE, 'r', encoding='utf-8') as f:
        params_data = json.load(f)  # 从JSON文件加载参数数据
    
    # 从解析的JSON中提取参数名称和边界
    param_names = [param['symbol'] for param in params_data]  # 获取所有参数的symbol
    param_bounds = [param['bounds'] for param in params_data]  # 获取所有参数的bounds
    
    # 创建SALib问题定义
    problem = {
        'names': param_names,  # 参数名称列表
        'bounds': param_bounds,  # 参数边界列表
        'num_vars': len(param_names)  # 自动计算参数数量
    }
    
    # 提取参数的单位信息 (用于后续写入文件头)
    units = [param['unit'] for param in params_data]  # 获取所有参数的单位
    
except Exception as e:
    print(f"错误: 无法读取或解析参数文件 - {e}")
    exit(1)  # 发生错误时退出程序

# 生成Sobol序列样本
N = 512  # 基础样本量，推荐使用2的幂次
param_values = saltelli.sample(problem, N, 
    calc_second_order=True  # 关闭二阶效应计算以节省样本量
)

# 样本量信息
"""
参数说明：
- 公式: N*(2D + 2) (D为参数数量)
- 当前参数数量 D = {len(param_names)}
- 实际样本量: {N}*(2*{len(param_names)} + 2) = {len(param_values)}
- 若需要二阶效应分析，可设置calc_second_order=True
"""

# 准备文件头
header_names = "# " + "\t".join(problem['names']) + "\n"  # 参数名称行
header_units = "# Unit:\t" + "\t".join(units) + "\n"  # 参数单位行 (从文件中读取)

# 格式化数值输出
formatted_values = []
for row in param_values:
    # 对每行数据应用科学计数法格式化，保留4位小数
    formatted_row = []
    for val in row:
        if abs(val) >= 1e4 or abs(val) < 1e-3:  # 较大或较小的数用科学计数法
            formatted_val = "{:.4e}".format(val)
        else:
            formatted_val = "{:.4f}".format(val)  # 普通数用固定小数位
        formatted_row.append(formatted_val)
    formatted_values.append("\t".join(formatted_row))

# 写入样本输出文件
OUTPUT_FILE = 'C:/Users/11201/Desktop/sensities/sobol/sobol_samples.txt'
try:
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(header_names)  # 写入参数名称行
        f.write(header_units)  # 写入单位行
        f.write("\n".join(formatted_values))  # 写入格式化后的样本数据
    
    # 输出执行信息
    print(f"成功生成样本量: {len(param_values)}")
    print(f"文件已保存至: {OUTPUT_FILE}")
    print("参数文件格式验证:")
    print(f"参数列表: {param_names}")
    print(f"首行示例: {header_names.strip()}")
    print(f"单位行示例: {header_units.strip()}")
    print(f"数据首行示例: {formatted_values[0]}")
    
except Exception as e:
    print(f"错误: 写入输出文件失败 - {e}")
    exit(1)
