import numpy as np
import pandas as pd
from SALib.analyze import morris
import os
import re
import json
# 配置参数（需要用户根据实际情况修改）
# ======================================================================
PARAM_FILE = 'C:/Users/11201/Desktop/sensities/morris/morris_samples_cpp.txt'   # 参数矩阵文件路径
RESULT_DIR = 'C:/Users/11201/Desktop/sensities/Candle_To_Analysize/morris/mAccu'    # 结果文件目录路径
SELECTED_NODE = (0, 0)                 # 要提取的节点坐标 (x,y)
SELECTED_TIME_INDEX = 2               # 提取最后一个时间点的数据（也可设为具体索引）
OUTPUT_VARS = ['mLeav','mFrozen','mAccu']      # 需要分析的输出变量列表
PARAM_NAMES = ['K','RHO','CP','TM','H','MU','SIG','GA','VM','L','DH','AP','AN','ML','MT','TD']  # 参数名称
# ======================================================================

# 将自定义参数结构转换为SALib标准格式
def convert_to_salib_problem(custom_params):
    """
    参数转换函数
    custom_params: 用户自定义的参数字典列表
    返回SALib标准problem字典
    """
    problem = {
        'num_vars': len(custom_params),
        # 使用符号作为参数名（也可用'name'字段）
        'names': [p['symbol'] for p in custom_params],  
        # 提取每个参数的上下界
        'bounds': [p['bounds'] for p in custom_params],
        # 必须显式声明groups字段
        'groups': None  
    }
    
    # 验证参数范围有效性
    for i, b in enumerate(problem['bounds']):
        if len(b) != 2 or b[0] >= b[1]:
            raise ValueError(f"参数 {problem['names'][i]} 范围 {b} 无效")
    
    return problem


def read_parameter_matrix(file_path):
    """
    更稳健的参数矩阵读取方法
    返回:
    - param_values: 参数值矩阵 (num_samples × num_params)
    - param_names: 参数名称列表
    """
    # 使用pandas读取以处理复杂分隔符
    df = pd.read_csv(file_path, 
                    comment='#',
                    sep='\t+',  # 匹配任意空白分隔符
                    header=None,
                    engine='python')
    
    # 验证列数匹配
    expected_cols = len(PARAM_NAMES)
    if df.shape[1] != expected_cols:
        raise ValueError(f"参数数量不匹配，期望{expected_cols}列，实际{df.shape[1]}列")
    
    return df.values, PARAM_NAMES


def parse_result_file(file_path, target_node, time_index):
    """
    解析单个结果文件，提取指定节点和时间的数值
    参数:
    - file_path: 结果文件路径
    - target_node: 目标节点坐标元组 (x,y)
    - time_index: 时间点索引（支持负数索引）
    返回: 提取的数值
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # 解析时间行并验证索引
    time_points = list(map(int, lines[0].strip().split()[1:]))
    actual_time_index = time_index if time_index >=0 else len(time_points)+time_index
    
    # 遍历数据行查找目标节点
    for line in lines[1:]:
        parts = line.strip().split()
        var_info = parts[0]
        
        # 使用正则表达式解析变量名和节点坐标
        match = re.match(r'(\w+)\((\d+),(\d+)\)', var_info)
        if match:
            var_name = match.group(1)
            node_x = int(match.group(2))
            node_y = int(match.group(3))
            
            # 检查节点匹配
            if (node_x, node_y) == target_node:
                try:
                    return float(parts[1 + actual_time_index])  # 数据列从索引1开始
                except IndexError:
                    raise ValueError(f"时间索引 {time_index} 超出范围，文件: {file_path}")
    
    # 如果未找到目标节点
    raise ValueError(f"未找到节点 {target_node} 的数据，文件: {file_path}")

def process_all_results(param_values, result_dir):
    """
    处理所有结果文件并生成分析矩阵
    返回:
    - Y: 结果矩阵 (num_samples × num_output_vars)
    """
    # 获取结果文件列表并按样本序号排序
    files = [f for f in os.listdir(result_dir) if f.endswith('.txt')]
    files.sort(key=lambda x: int(re.search(r'^\d+', x).group()))  # 按开头的数字排序
    
    # 验证文件数量与参数样本数匹配
    if len(files) != param_values.shape[0]:
        raise ValueError(f"参数样本数({param_values.shape[0]})与结果文件数({len(files)})不匹配")
    
    # 初始化结果矩阵
    Y = np.zeros((len(files), len(OUTPUT_VARS)))
    
    for i, filename in enumerate(files):
        file_path = os.path.join(result_dir, filename)
        
        # 解析每个输出变量
        for var_idx, target_var in enumerate(OUTPUT_VARS):
            # 检查文件名是否包含目标变量
            if target_var in filename:
                Y[i, var_idx] = parse_result_file(file_path, SELECTED_NODE, SELECTED_TIME_INDEX)
    
    return Y

# 主程序流程
# ======================================================================
if __name__ == "__main__":
    # 1. 读取参数矩阵
    X, param_names = read_parameter_matrix(PARAM_FILE)
    
    # 2. 处理结果数据
    Y = process_all_results(X, RESULT_DIR)
    
    # 3. 执行Morris分析（每个输出变量单独分析）
    results = {}
    for var_idx, var_name in enumerate(OUTPUT_VARS):
        # 定义问题字典（SALib要求格式）
        

        # 读取参数文件
        file_path1 = 'C:/Users/11201/Desktop/sensities/morris/morris_parameter.txt'
        with open(file_path1, 'r', encoding='utf-8') as f:
            parameter_system = json.load(f)
        problem = convert_to_salib_problem(parameter_system)

        # 执行分析
        analysis_results = morris.analyze(
            problem, X, Y[:, var_idx],
            conf_level=0.95,  # 置信水平
            num_resamples=2000, # 重采样次数
            num_levels=12,  # 这里应该和生成样本的num_levels相同设置为8
            print_to_console=False
            )

        
        # 转换结果为DataFrame便于查看
        df = pd.DataFrame(analysis_results)
        df['parameter'] = param_names
        df.set_index('parameter', inplace=True)
        
        results[var_name] = df
    
    # 输出结果示例
    for var_name, df in results.items():
        print(f"\n{var_name} Morris分析结果:")
        print(df[['mu_star', 'sigma']].sort_values('mu_star', ascending=False))
