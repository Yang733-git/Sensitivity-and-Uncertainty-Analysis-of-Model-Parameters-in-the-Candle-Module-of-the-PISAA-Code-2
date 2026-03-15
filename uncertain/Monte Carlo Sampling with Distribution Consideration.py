import numpy as np
import pandas as pd
import json
import math
from scipy.stats import truncnorm, lognorm, triang

def load_core_params(json_path):
    """从JSON文件加载参数定义，确保符号映射正确"""
    with open(json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)
    
    SYMBOL_MAP = {'rho': 'ρ', 'gamma': 'γ', 'alph': 'α'}
    
    core_params = []
    for p in params:
        symbol = SYMBOL_MAP.get(p['symbol'], p['symbol'])
        unit = p['unit'].replace('(', '').replace(')', '')
        core_params.append((symbol, p['bounds'][0], p['bounds'][1], unit))
    
    return core_params

def format_float(x):
    """
    格式化浮点数，确保最大精度的同时去除不必要的零
    处理规则：
    1. 对于-0.0001 < x < 0.0001 的极小值，固定保留6位小数
    2. 其他情况：保留6位小数后去除尾部的零
    3. 确保整数部分不会丢失小数位
    """
    # 处理极小数的情况
    if abs(x) < 0.0001 and abs(x) > 0:
        return "{0:.6f}".format(x)
    
    # 标准处理：先格式化为字符串
    s = "{0:.6f}".format(x).rstrip('0').rstrip('.')
    
    # 确保整数部分保留小数位
    if '.' not in s:
        return s + ".0"  # 对于整数值添加小数位
    return s

def tn_params_from_bounds(L, U):
    mu = 0.5 * (L + U)
    sigma = (U - L) / (2 * 1.96)
    return mu, max(sigma, 1e-12)

def ln_params_from_bounds(L, U):
    L = max(L, 1e-16)
    U = max(U, 1e-16)
    mu_ln = 0.5 * math.log(L * U)
    sigma_ln = (math.log(U) - math.log(L)) / (2 * 1.96)
    return mu_ln, max(sigma_ln, 1e-12)

def ln_params_from_rel_unc(L, U, rel_unc_95):
    L = max(L, 1e-16)
    U = max(U, 1e-16)
    mu_ln = 0.5 * math.log(L * U)
    cv = rel_unc_95 / 1.96
    sigma_ln = math.sqrt(math.log(1.0 + cv * cv))
    return mu_ln, max(sigma_ln, 1e-12)

def truncated_rvs_of(dist, L, U, size):
    FL = dist.cdf(L)
    FU = dist.cdf(U)
    if not np.isfinite(FL):
        FL = 0.0
    if not np.isfinite(FU):
        FU = 1.0
    if abs(FU - FL) < 1e-14:
        return np.full(size, 0.5 * (L + U), dtype=float)
    u = np.random.uniform(0.0, 1.0, size)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return dist.ppf(FL + u * (FU - FL))

def normalize_symbol(symbol):
    s = str(symbol).strip()
    alias_map = {
        'ρ': 'RHO',
        'RHO': 'RHO',
        'rho': 'RHO',
        'μ': 'MU',
        'MU': 'MU',
        'mu': 'MU',
        'σ': 'SIG',
        'SIG': 'SIG',
        'sig': 'SIG',
        'sigma': 'SIG',
        'γ': 'GA',
        'GA': 'GA',
        'ga': 'GA',
        'gamma': 'GA',
        'K': 'K',
        'k': 'K',
        'CP': 'CP',
        'cp': 'CP',
        'TM': 'TM',
        'tm': 'TM',
        'H': 'H',
        'h': 'H',
        'VM': 'VM',
        'vm': 'VM',
        'L': 'L',
        'l': 'L',
        'DH': 'DH',
        'dh': 'DH',
        'AP': 'AP',
        'ap': 'AP',
        'AN': 'AN',
        'an': 'AN',
        'ML': 'ML',
        'ml': 'ML',
        'MT': 'MT',
        'mt': 'MT',
        'TD': 'TD',
        'td': 'TD',
    }
    return alias_map.get(s, s.upper())

def get_component_key(material_name):
    component_map = {
        'UO2': 'fuel',
        'CL': 'cladding',
        'CR': 'control_rod',
    }
    return component_map[material_name]

DIST_SPEC = {
    'fuel': {
        'K':   ('LN_TRUNC', {'rel_unc_95': 0.40}),
        'RHO': ('TN',       {}),
        'CP':  ('LN_TRUNC', {'rel_unc_95': 0.10}),
        'TM':  ('TN',       {}),
        'H':   ('TRI',      {'a': 1400000.0, 'c': 1505000.0, 'b': 1610000.0}),
        'MU':  ('LN_TRUNC', {'rel_unc_95': 0.50}),
        'TD':  ('U',        {'use_bounds': True}),
        'MT':  ('U',        {'use_bounds': True}),
    },
    'cladding': {
        'K':   ('TRI', {'a': 32.9, 'c': 36.55, 'b': 40.2}),
        'RHO': ('TN',  {}),
        'CP':  ('TRI', {'a': 382.0, 'c': 415.5, 'b': 449.0}),
        'TM':  ('TRI', {'a': 2100.0, 'c': 2128.0, 'b': 2156.0}),
        'H':   ('TRI', {'a': 890000.0, 'c': 945000.0, 'b': 1000000.0}),
        'MU':  ('TRI', {'a': 0.006, 'c': 0.0105, 'b': 0.015}),
        'TD':  ('U',   {'use_bounds': True}),
        'MT':  ('U',   {'use_bounds': True}),
    },
    'control_rod': {
        'K':   ('TRI', {'a': 47.7, 'c': 71.35, 'b': 95.0}),
        'RHO': ('TN',  {}),
        'CP':  ('TRI', {'a': 217.0, 'c': 241.0, 'b': 265.0}),
        'TM':  ('TN',  {}),
        'H':   ('TRI', {'a': 183000.0, 'c': 296000.0, 'b': 409000.0}),
        'MU':  ('TRI', {'a': 0.002, 'c': 0.004, 'b': 0.006}),
        'TD':  ('U',   {'use_bounds': True}),
        'MT':  ('U',   {'use_bounds': True}),
    },
    'common': {
        'ML':  ('U',   {'use_bounds': True}),
        'SIG': ('U',   {'a': 0.40, 'b': 0.60}),
        'GA':  ('TRI', {'a': 0.025, 'c': 0.030, 'b': 0.035}),
        'VM':  ('U',   {'a': 0.1, 'b': 0.16}),
        'L':   ('U',   {'a': 0.08, 'b': 0.12}),
        'DH':  ('U',   {'use_bounds': True}),
        'AP':  ('U',   {'use_bounds': True}),
        'AN':  ('U',   {'a': 0.3, 'b': 0.4}),
    }
}

def draw_samples_by_distribution(symbol, bounds, component, size):
    canonical_symbol = normalize_symbol(symbol)
    low = float(bounds[0])
    high = float(bounds[1])
    spec = DIST_SPEC.get(component, {}).get(canonical_symbol)
    
    if spec is None:
        spec = DIST_SPEC['common'].get(canonical_symbol)
    
    if spec is None:
        return np.random.uniform(low, high, size)
    
    dist_type, params = spec
    
    if dist_type == 'U':
        if params.get('use_bounds'):
            a = low
            b = high
        else:
            a = float(params['a'])
            b = float(params['b'])
        return np.random.uniform(a, b, size)
    
    if dist_type == 'TRI':
        a = float(params['a'])
        c_mode = float(params['c'])
        b = float(params['b'])
        c = (c_mode - a) / (b - a)
        dist = triang(c, loc=a, scale=(b - a))
        u = np.random.uniform(0.0, 1.0, size)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return dist.ppf(u)
    
    if dist_type == 'TN':
        mu, sigma = tn_params_from_bounds(low, high)
        a = (low - mu) / sigma
        b = (high - mu) / sigma
        dist = truncnorm(a, b, loc=mu, scale=sigma)
        u = np.random.uniform(0.0, 1.0, size)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        return dist.ppf(u)
    
    if dist_type == 'LN_TRUNC':
        if 'rel_unc_95' in params:
            mu_ln, sigma_ln = ln_params_from_rel_unc(low, high, float(params['rel_unc_95']))
        else:
            mu_ln, sigma_ln = ln_params_from_bounds(low, high)
        dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))
        return truncated_rvs_of(dist, low, high, size)
    
    return np.random.uniform(low, high, size)

# ================ 主要修改部分 ================
# 定义三种材料及其参数文件路径
materials = [
    ("UO2", "C:/Users/11201/Desktop/sensities/uncertain/uncertain_parameterUO2.txt"),
    ("CL", "C:/Users/11201/Desktop/sensities/uncertain/uncertain_parameterCL.txt"),
    ("CR", "C:/Users/11201/Desktop/sensities/uncertain/uncertain_parameterCR.txt")
]

# 固定样本数量
n_samples = 2000
np.random.seed(12345678)  # 设置全局随机种子保证可复现性

# 遍历处理三种材料
for material_name, json_path in materials:
    print(f"\n处理材料: {material_name}...")
    
    # 1. 加载当前材料参数
    core_params = load_core_params(json_path)
    
    # 2. 定义抽样问题
    problem = {
        'num_vars': len(core_params),
        'names': [p[0] for p in core_params],
        'bounds': [[p[1], p[2]] for p in core_params],
        'dists': ['unif'] * len(core_params)  # 均匀分布
    }
    
    # ================ 修正蒙特卡洛抽样方法 ================
    # 3. 直接实现蒙特卡洛抽样（简单随机抽样）
    num_vars = problem['num_vars']
    # 创建空数组存储样本
    samples = np.zeros((n_samples, num_vars))
    component_key = get_component_key(material_name)
    
    # 对每个变量独立抽样
    for i, bounds in enumerate(problem['bounds']):
        symbol = problem['names'][i]
        samples[:, i] = draw_samples_by_distribution(symbol, bounds, component_key, n_samples)
    
    df = pd.DataFrame(samples, columns=problem['names'])
    # ================ 修正结束 ================
    
    # 4. 格式化输出数据
    formatted_df = pd.DataFrame()
    for col in df.columns:
        formatted_df[col] = df[col].apply(format_float)
    
    data_lines = []
    for _, row in formatted_df.iterrows():
        line = '\t'.join(row.astype(str))
        data_lines.append(line.replace(' ', ''))
    data_content = '\n'.join(data_lines)
    
    # 5. 边界验证
    print(f"{material_name}参数边界验证:")
    param_bound_dict = {p[0]: (p[1], p[2]) for p in core_params}
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        expected_min, expected_max = param_bound_dict[col]
        print(f"  {col}: 样本范围[{min_val:.6f},{max_val:.6f}] | 设定范围[{expected_min},{expected_max}]")
    
    # 6. 准备文件头
    var_header = "# " + "\t".join(problem['names'])
    unit_header = "#Unit: " + "\t".join([p[3] for p in core_params])
    
    # 7. 生成输出文件路径（包含材料名）
    output_path = f'C:/Users/11201/Desktop/sensities/uncertain/uncertain_samples_{material_name}.txt'
    
    # 8. 写入文件
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(var_header + "\n")
        f.write(unit_header + "\n")
        f.write(data_content)
    
    # 9. 验证数据
    def validate_samples(data_frame):
        """增强型数据验证"""
        param_dict = {p[0]: (p[1], p[2]) for p in core_params}
        
        # 数值范围验证
        for col in data_frame.columns:
            min_val = data_frame[col].min()
            max_val = data_frame[col].max()
            expected_min, expected_max = param_dict[col]
            
            # 打印边界偏差警告
            if min_val < expected_min:
                print(f"[WARNING] {col}下限偏低: {min_val} < {expected_min}")
            if max_val > expected_max:
                print(f"[WARNING] {col}上限偏高: {max_val} > {expected_max}")
            
            # 执行严格断言
            assert min_val >= expected_min, f"{col}下限异常：{min_val}<{expected_min}"
            assert max_val <= expected_max, f"{col}上限异常：{max_val}>{expected_max}"
        
        # 检查文件行数是否正确
        with open(output_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == len(data_frame) + 2, "文件行数不匹配"
    
    validate_samples(df)
    
    # 10. 打印文件信息
    print(f"\n{material_name}生成文件首部示例：")
    with open(output_path) as f:
        print(f.readline().strip())  # 变量行
        print(f.readline().strip())  # 单位行
        print(f.readline().strip())  # 第一行数据
    
    print(f"成功生成{len(df)}个{material_name}样本，保存至：{output_path}\n")

print("\n所有材料样本生成完成！")