# -*- coding: utf-8 -*-
"""
材料参数Morris采样生成方案
功能：仅生成Morris样本矩阵，兼容C++读入
环境需求：Python 3.10 + SALib 1.4.7
"""

# ============== 步骤1：参数系统整合 ==============
import numpy as np
from SALib import ProblemSpec
from SALib.sample import morris
import ast
import json

# 读取参数文件
file_path1 = 'C:/Users/11201/Desktop/sensities/morris/morris_parameter.txt'
with open(file_path1, 'r', encoding='utf-8') as f:
    parameter_system = json.load(f)


# 验证参数总数（12个参数）
print(f"参数总数验证：{len(parameter_system)}")

# ============== 步骤2：构建SALib问题 ==============
def build_salib_problem(params):
    """将自定义参数格式转换为SALib要求格式"""
    problem = {
        "num_vars": len(params),
        "names": [],
        "bounds": []
    }
    
    # 逐参数填充字段
    for p in params:
        problem["names"].append(p["symbol"])
        problem["bounds"].append(p["bounds"])
    
    return problem

# 生成SALib兼容的问题定义
salib_problem = build_salib_problem(parameter_system)
print("\n前3个参数定义：")
for i in range(3):
    print(f"{salib_problem['names'][i]}: {salib_problem['bounds'][i]}")

# ============== 步骤3：Morris采样 ==============
def generate_morris_samples(problem, trajectory_count):
    """使用优化轨迹法提高采样效率"""
    # 固定随机种子（关键修改）
    np.random.seed(123)  # 👈 任意整数均可
    
    samples = morris.sample(
        problem=problem,
        N=trajectory_count,
        num_levels=8,
        optimal_trajectories=None  
    )
    
    # 检查样本规范性
    print(f"\n样本矩阵维度：{samples.shape}")
    print("首样本验证：", samples[0, :3])
    
    return samples

# 生成样本（轨迹数）
morris_samples = generate_morris_samples(salib_problem, trajectory_count=12) 

# ============== 步骤4：C++兼容输出 ==============
def save_for_cpp(samples, params, filename):
    """生成C++可读取的带参数说明的样本文件"""
    
    # 生成头信息（参数顺序说明）
    headers = [p["symbol"] for p in params]
    header_line = "# " + "\t".join(headers) + "\n"
    
    # 生成单位行
    units = [p["unit"] for p in params]
    unit_line = "# Unit:\t" + "\t".join(units) + "\n"

    fixed_path = "C:/Users/11201/Desktop/sensities/morris/" + filename
    
      # 修正编码处理（使用utf-8编码）
    with open(fixed_path, 'w', encoding='utf-8') as f:  # 关键修改：指定编码
        f.write(header_line)
        f.write(unit_line)
        np.savetxt(
            fname=f,  # 传入文件对象
            X=samples,
            delimiter="\t",
            fmt="%.4e", 
            comments=''
        )

    
    print(f"\n文件已保存：{fixed_path} (最后维度验证: {samples.shape})")

save_for_cpp(
    samples=morris_samples,
    params=parameter_system,
    filename="morris_samples_cpp.txt"
)

# ============== 验证样本质量 ==============
#print("\n参数范围验证（第一参数K）：")
#print("理论范围：[1.5, 3.5]")
#print("实际范围：[{:.4f}, {:.4f}]".format(
#    morris_samples[:,0].min(),
#    morris_samples[:,0].max()
#))
