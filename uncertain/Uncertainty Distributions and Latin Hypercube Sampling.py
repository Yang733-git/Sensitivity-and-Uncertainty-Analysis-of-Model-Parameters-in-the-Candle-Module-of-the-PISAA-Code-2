# -*- coding: utf-8 -*-  # 文件编码：UTF-8，确保中文注释与字符串安全读取
# Python 3.10+  # 指定运行环境版本要求（3.10及以上）
# 功能：  # 顶层说明：脚本的整体功能说明块
#  1) 读取三类组件的参数文件（JSON 数组：[{symbol, name, unit, bounds}, ...]）  # 功能点1：输入数据结构为JSON数组，包含每个参数的符号/名称/单位/边界
#  2) “参数分布选取 & 参数值确定”严格按《参数分布选取依据》：  # 功能点2：严格遵循文档“参数分布选取依据”中的方法
#     - 乘性误差、右偏、给相对不确定度±x%（以95%覆盖带解释）→ 截断对数正态 LN_TRUNC  # 规则2.1：乘性误差采用截断对数正态
#         · μ_ln 取为几何中点 ln(√(L·U))，对应“经验式的中位数”近似  # 参数设置：μ_ln 的选取
#         · σ_ln = sqrt( ln( 1 + (x/1.96)^2 ) )  # 参数设置：σ_ln 的选取，x为95%覆盖意义下的相对不确定度
#         · 并在物理上下界 [L,U] 内截断  # 保证数值物理合理性
#     - 加性误差、围绕名义值窄幅波动 → 截断正态 TN，μ=0.5(L+U)，σ=(U-L)/(2*1.96)  # 规则2.2：加性误差采用截断正态
#     - 仅有最小-最可能-最大 → 三角分布 Tri(a,mode,b)  # 规则2.3：给出三点值时用三角分布
#     - 仅有区间、无偏好 → 均匀分布 U(a,b)  # 规则2.4：仅区间时用均匀分布
#  3) 输出与示例一致的 .tsv：两行表头 + 科学计数法数据行  # 功能点3：输出格式约定
#
# 仅修改了“参数分布选取与参数值确定”的实现，其余保持不变。  # 变更范围：仅分布选择逻辑被改动，其它保持一致

# ======================== 导入标准库与第三方库（功能块） ========================  # 说明：导入依赖库以支持JSON解析、数值计算与统计分布
import json  # Python 标准库：读写和解析 JSON 文件
import math  # Python 标准库：数学运算（对数、平方根等）
from pathlib import Path  # Python 标准库：文件路径拼接与处理
import numpy as np  # 数值计算：数组与矢量化操作
from scipy.stats import truncnorm, lognorm, triang, beta  # 统计分布：截断正态、对数正态、三角分布、Beta分布
from scipy.stats import qmc  # QMC：拉丁超立方抽样 LatinHypercube

# ======================== 配置区（根据需要改）（功能块） ========================  # 说明：输入/输出路径与全局配置
# 你的三份参数文件（与之前上传的控制棒文件同结构：JSON 数组）  # 三个组件的参数定义文件路径
PATH_CR = r"C:\Users\11201\Desktop\sensities\morris\morris_parameterCR.txt"  # 控制棒参数文件路径
PATH_CL = r"C:\Users\11201\Desktop\sensities\morris\morris_parameterCL.txt"  # 包壳（Cladding）参数文件路径
PATH_UO2 = r"C:\Users\11201\Desktop\sensities\morris\morris_parameterUO2.txt"  # 燃料（UO2）参数文件路径

# 输出文件路径（默认与输入同目录）  # 依据输入路径生成对应输出文件名
OUT_CR = str(Path(PATH_CR).with_name("CR_LHS_samples.txt"))  # 控制棒LHS输出文件路径
OUT_CL = str(Path(PATH_CL).with_name("CL_LHS_samples.txt"))  # 包壳LHS输出文件路径
OUT_UO2 = str(Path(PATH_UO2).with_name("UO2_LHS_samples.txt"))  # 燃料LHS输出文件路径

# 每个组件抽样数量  # LHS样本数
N_SAMPLES = 2000  # 拉丁超立方抽样的样本数量，默认500条

# 列顺序与单位行（与示例一致）  # 输出列的顺序与单位描述行
ORDER = ["K","RHO","CP","TM","H","MU","SIG","GA","VM","L","DH","AP","AN","ML","MT","TD"]  # 参数列顺序，必须与输入JSON的symbol匹配
UNITS_LINE = "# Unit:\tW/(m·K)\tkg/m³\tkJ/(kg·K)\tK\tJ/kg\tPa·s\tm\t°C/m\tm/s\tm\tm\tm²\tm²\tkg\tkg\tK"  # 单位行，与原输出格式一致
Z95 = 1.96  # 近似两侧 95% 分位  # 1.96 用于把95%覆盖映射为标准差倍数

# ======================== 分布参数工具（功能块） ================================  # 说明：将边界或相对不确定度转换为分布参数
def tn_params_from_bounds(L: float, U: float):  # 定义：由[L,U]推断截断正态的均值与标准差
    """把 [L,U] 当作 ~95% 覆盖区间，反推截断正态的 (mu, sigma)（随后在 [L,U] 截断）"""  # 文档字符串：说明推断方法
    mu = 0.5*(L+U)  # 取区间中点作为均值
    sigma = (U - L) / (2*Z95)  # 令两侧±1.96σ覆盖[L,U]，求得σ
    return mu, max(sigma, 1e-12)  # 返回μ与σ（σ下限防止为0）

def ln_params_from_bounds(L: float, U: float):  # 定义：由[L,U]推断对数正态参数
    """把 [L,U] 当作对数域的 ~95% 区间，反推对数正态的 (mu_ln, sigma_ln)"""  # 文档字符串：说明推断方法
    L = max(L, 1e-16)  # 保护性下限，避免对数域的非正值
    mu_ln = 0.5 * math.log(L*U)  # 几何中位数  # μ_ln 取 ln(√(L·U))
    sigma_ln = (math.log(U) - math.log(L)) / (2*Z95)  # 依据95%覆盖推导σ_ln
    return mu_ln, max(sigma_ln, 1e-12)  # 返回μ_ln与σ_ln（σ_ln下限保护）

def ln_params_from_rel_unc(L: float, U: float, rel_unc_95: float):  # 定义：依据相对不确定度±x%推断LN参数
    """
    依据“相对不确定度±x%按95%覆盖带解释”的原则：
      σ_ln = sqrt( ln(1 + (x/1.96)^2) ), μ_ln 取 ln(√(L·U)).
    """  # 文档字符串：提供公式与约定
    mu_ln = 0.5 * math.log(max(L, 1e-16) * max(U, 1e-16))  # μ_ln 仍取几何中点（保证正值安全）
    cv = rel_unc_95 / Z95  # 将95%相对不确定度换算为标准差等效的比例
    sigma_ln = (math.log(1.0 + cv*cv))**0.5  # 根据公式求 σ_ln
    return mu_ln, max(sigma_ln, 1e-12)  # 返回μ_ln与σ_ln（σ_ln下限保护）

def beta_params_symmetric_from_95_width(width: float) -> float:  # 定义：由95%宽度近似求对称Beta的α
    """
    用正态近似求对称 Beta(alpha, alpha) 的 alpha，使中央 95% 宽度约为 width（定义在 [0,1] 上）。
    """  # 文档字符串：说明近似方法
    half = width / 2.0  # 中央95%的一半宽度
    var = (half / 1.96) ** 2  # 正态近似下的方差估计
    # 对称 Beta 在 0.5 附近 var ≈ 1/(8*(2*alpha+1)) => alpha ≈ 1/(16*var) - 0.5  # 推导关系用于反推α
    alpha = (1.0 / (16.0 * var)) - 0.5  # 计算α
    return float(max(alpha, 2.0))  # 限制α的下限，避免过于平坦

SIG_ALPHA = beta_params_symmetric_from_95_width(0.2)  # 将 [0.4,0.6] 视为 95% 区间（仅保留以兼容旧逻辑，当前未使用）  # 对称Beta参数（兼容保留）

def truncated_ppf_of(dist, L: float, U: float):  # 定义：生成任意分布在[L,U]截断后的分位函数
    """给任意连续分布 dist 构造在 [L,U] 的截断分位函数（CDF 重标定）。"""  # 文档字符串：说明CDF重标定方法
    FL, FU = dist.cdf(L), dist.cdf(U)  # 计算原分布在L与U的CDF值
    if not np.isfinite(FL): FL = 0.0  # 数值保护：下界CDF异常时置0
    if not np.isfinite(FU): FU = 1.0  # 数值保护：上界CDF异常时置1
    if abs(FU - FL) < 1e-14:  # 若区间CDF几乎不变，说明尺度极小
        mid = 0.5*(L+U)  # 取区间中点作为近似
        return lambda u: np.full_like(u, mid, dtype=float)  # 返回常数分位函数以避免数值不稳定
    def ppf(u):  # 内部函数：映射U[0,1]到截断分布的分位
        u = np.clip(u, 1e-12, 1-1e-12)  # 裁剪u避免极端0或1导致的反函数溢出
        return dist.ppf(FL + u*(FU - FL))  # 线性重标定CDF后求ppf
    return ppf  # 返回构造好的分位函数

# ======================== “依据驱动”的分布选择表（功能块） =======================  # 说明：按“参数分布选取依据”只保留必要的选择信息
# 仅描述“选择依据”所需的信息：  # 该表不直接给出最终μ/σ，而是给出基于依据的参数化方式
#   - LN_TRUNC：通过 rel_unc_95 指定95%覆盖意义下的相对不确定度（x 以小数表示，如 0.40）  # 乘性误差按相对不确定度设定
#               μ_ln / σ_ln 将按 ln_params_from_rel_unc(...) 用 JSON 的 [L,U] 计算，并在 [L,U] 截断  # 调参函数由代码计算
#   - TN：若未给 μ/σ，则按 tn_params_from_bounds(...) 用 JSON 的 [L,U] 计算，并在 [L,U] 截断  # 加性误差按边界反推
#   - TRI/U：按给定三点或区间（若未给 a/b 则回退到 JSON bounds）  # 三角/均匀分布按三点值或边界使用
DIST_SPEC = {  # 分布依据总表
    "fuel": {  # 燃料（UO2）组件
        "K":   ("LN_TRUNC", {"rel_unc_95": 0.40}),  # ±40%（95%覆盖） # 导热率K：乘性偏差较大，采用LN截断
        "RHO": ("TN",        {}),                   # 加性窄幅，围绕名义值 # 密度：截断正态（未知μσ由[L,U]推）
        "CP":  ("LN_TRUNC", {"rel_unc_95": 0.10}),  # ±10%（95%覆盖） # 定压比热：乘性误差较小的LN
        "TM":  ("TN",        {}),                   # 熔点：截断正态
        "H":   ("TRI",       {"a": 1400., "c": 1505., "b": 1610.}),  # 三点估计：采用三角分布
        "MU":  ("LN_TRUNC", {"rel_unc_95": 0.50}),  # ±50%（95%覆盖） # 粘度：右偏乘性，采用LN截断
        "TD":  ("U",         {"use_bounds": True}),  # 仅区间：均匀分布（用JSON边界）
        "MT":  ("U",         {"use_bounds": True}),  # 仅区间：均匀分布（用JSON边界）
    },
    "cladding": {  # 包壳组件
        "K":   ("TRI",  {"a": 32.9, "c": 36.55, "b": 40.2}),  # 三角分布（给定三点）
        "RHO": ("TN",   {}),  # 截断正态（由[L,U]推μσ）
        "CP":  ("TRI",  {"a": 382., "c": 415.5, "b": 449.}),  # 三角分布
        "TM":  ("TRI",  {"a": 2100., "c": 2128., "b": 2156.}),  # 三角分布
        "H":   ("TRI",  {"a": 890., "c": 945., "b": 1000.}),  # 三角分布
        "MU":  ("TRI",  {"a": 0.006, "c": 0.0105, "b": 0.015}),  # 三角分布
        "TD":  ("U",    {"use_bounds": True}),  # 仅区间：均匀分布
        "MT":  ("U",    {"use_bounds": True}),  # 仅区间：均匀分布
    },
    "control_rod": {  # 控制棒组件
        "K":   ("TRI",  {"a": 47.7, "c": 71.35, "b": 95.}),  # 三角分布
        "RHO": ("TN",   {}),  # 截断正态
        "CP":  ("TRI",  {"a": 217., "c": 241., "b": 265.}),  # 三角分布
        "TM":  ("TN",   {}),  # 截断正态
        "H":   ("TRI",  {"a": 183., "c": 296., "b": 409.}),  # 三角分布
        "MU":  ("TRI",  {"a": 0.002, "c": 0.004, "b": 0.006}),  # 三角分布
        "TD":  ("U",    {"use_bounds": True}),  # 仅区间：均匀分布
        "MT":  ("U",    {"use_bounds": True}),  # 仅区间：均匀分布
    },
    # 三个组件通用项  # 所有组件共享的几何与工况参数分布
    "common": {
        "ML": ("U",   {"use_bounds": True}),  # 仅区间：均匀分布（质量下限-上限）
        "SIG":("U",   {"a": 0.40,   "b": 0.60}),  # 应力梯度：均匀分布于[0.4,0.6]
        "GA": ("TRI", {"a": 0.025,  "c": 0.030, "b": 0.035}),  # 三角分布（几何参数）
        "VM": ("U",   {"a": 0.1,   "b": 0.16}),  # 速度：均匀分布
        "L":  ("U",   {"a": 0.08,    "b": 0.12}),  # 长度：均匀分布
        "DH": ("U",   {"a": 0.014,  "b": 0.022}),  # 水力直径：均匀分布
        "AP": ("U",   {"a": 0.001,  "b": 0.008}),  # 面积参数：均匀分布
        "AN": ("U",   {"a": 0.3,    "b": 0.4}),  # 面积参数：均匀分布
    }
}  # 结束：分布依据总表

# ======================== 分布映射（按“依据”求参）（功能块） ====================  # 说明：根据DIST_SPEC生成各参数的分位函数
def make_ppf(symbol: str, bounds, component: str):  # 定义：创建符号参数的分位函数（用于U[0,1]→目标分布）
    """
    返回该参数的分位函数 ppf(u)，其中 u ~ U[0,1]。
    分布类型和参数值均按《参数分布选取依据》确定；
    - LN：用相对不确定度±x%（95%覆盖）求 σ_ln，并以 [L,U] 的几何中点定 μ_ln；最后在 [L,U] 截断。
    - TN：把 [L,U] 解读为 95% 覆盖区间求 μ、σ，并在 [L,U] 截断。
    - TRI/U：按表给定；若 U 指定 use_bounds，则直接用 JSON bounds。

    若表中某参数缺失，则回退旧规则，并严格限制在 JSON 的 [L,U]。
    """  # 文档字符串：详细说明映射逻辑与回退策略
    s = symbol.upper()  # 规范化符号为大写，便于与ORDER/DIST_SPEC对齐
    comp = component.lower()  # 组件名称统一为小写，便于字典索引
    L, U = float(bounds[0]), float(bounds[1])  # 从输入JSON拿到物理下/上界并转为浮点

    spec_table = DIST_SPEC.get(comp, {})  # 获取该组件的分布依据子表（若无返回空表）
    spec = spec_table.get(s) or DIST_SPEC["common"].get(s)  # 优先找组件专属项，否则找通用项

    if spec is None:  # 若在依据表中未匹配到该参数
        # 回退：保持兼容（严格限于 JSON 的 [L,U]）  # 使用旧规则但仍限定于[L,U]
        if s in {"L","DH","AP","AN","GA","VM"}:  # 对几何/流动参数采用三角分布近似
            a, b = L, U  # 区间上下界
            m = 0.5*(a+b)  # 模式近似取中点
            c = (m - a) / (b - a)  # 三角分布形状参数c
            dist = triang(c, loc=a, scale=(b-a))  # 构造三角分布对象
            return lambda u: dist.ppf(u)  # 返回分位函数
        if s == "SIG":  # 对SIG采用对称Beta（兼容旧逻辑）
            dist = beta(SIG_ALPHA, SIG_ALPHA, loc=L, scale=(U-L))  # Beta分布映射到[L,U]
            return lambda u: dist.ppf(u)  # 返回分位函数
        if s == "ML":  # ML采用Beta(2,2)（兼容旧逻辑）
            from scipy.stats import beta as beta_dist  # 局部导入避免全局命名冲突
            dist = beta_dist(2, 2, loc=L, scale=(U-L))  # Beta(2,2)映射到[L,U]
            return lambda u: dist.ppf(u)  # 返回分位函数
        if s in {"MT","TD"}:  # 温度相关量采用截断正态
            mu, sigma = tn_params_from_bounds(L, U)  # 由[L,U]反推μ、σ
            a, b = (L - mu)/sigma, (U - mu)/sigma  # 转为标准化截断区间
            dist = truncnorm(a, b, loc=mu, scale=sigma)  # 构造截断正态分布对象
            return lambda u: dist.ppf(u)  # 返回分位函数
        if s in {"K","MU"}:  # 导热率/粘度：右偏乘性→对数正态
            mu_ln, sigma_ln = ln_params_from_bounds(L, U)  # 由[L,U]反推μ_ln、σ_ln
            dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))  # 构造对数正态分布对象
            return truncated_ppf_of(dist, L, U)  # 返回在[L,U]截断后的分位函数
        if s == "H":  # 热焓H
            if comp == "fuel":  # 燃料：用对数正态（右偏）
                mu_ln, sigma_ln = ln_params_from_bounds(L, U)  # 反推LN参数
                dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))  # 构造LN分布
                return truncated_ppf_of(dist, L, U)  # 截断分位函数
            else:  # 其他组件：用三角分布
                a, b = L, U  # 区间上下界
                m = 0.5*(a+b)  # 模式取中点
                c = (m - a)/(b - a)  # 三角分布形状参数
                dist = triang(c, loc=a, scale=(b-a))  # 构造三角分布
                return lambda u: dist.ppf(u)  # 返回分位函数
        if s in {"RHO","CP","TM"}:  # 密度/比热/熔点：加性误差
            mu, sigma = tn_params_from_bounds(L, U)  # 由[L,U]反推μ、σ
            a, b = (L - mu)/sigma, (U - mu)/sigma  # 标准化截断区间
            dist = truncnorm(a, b, loc=mu, scale=sigma)  # 构造截断正态
            return lambda u: dist.ppf(u)  # 返回分位函数
        # 兜底  # 未覆盖到的情况仍回退TN
        mu, sigma = tn_params_from_bounds(L, U)  # 由[L,U]反推μ、σ
        a, b = (L - mu)/sigma, (U - mu)/sigma  # 标准化截断区间
        dist = truncnorm(a, b, loc=mu, scale=sigma)  # 构造截断正态
        return lambda u: dist.ppf(u)  # 返回分位函数

    # 按“依据”构造分布  # 如果在DIST_SPEC中找到了配置信息
    dist_type, params = spec  # 解包分布类型与参数
    if dist_type == "U":  # 均匀分布
        if params.get("use_bounds"):  # 若指定使用JSON边界
            a, b = L, U  # 使用[L,U]
        else:
            a, b = float(params["a"]), float(params["b"])  # 使用表内给定区间
        return lambda u: a + (b - a) * u  # 均匀分布的分位函数线性映射

    elif dist_type == "TRI":  # 三角分布
        a = float(params["a"]); c_mode = float(params["c"]); b = float(params["b"])  # 取三点值a, c(众数), b
        c = (c_mode - a) / (b - a)  # 计算scipy.triang所需的形状参数c
        dist = triang(c, loc=a, scale=(b - a))  # 构造三角分布
        return lambda u: dist.ppf(u)  # 返回分位函数

    elif dist_type == "TN":  # 截断正态
        if "mu" in params and "sigma" in params and "L" in params and "U" in params:  # 若显式给定μ、σ与截断上下界
            mu, sigma = float(params["mu"]), float(params["sigma"])  # 读取μ与σ
            Lp, Up = float(params["L"]), float(params["U"])  # 读取截断边界
            a, b = (Lp - mu)/sigma, (Up - mu)/sigma  # 标准化截断区间
            dist = truncnorm(a, b, loc=mu, scale=sigma)  # 构造截断正态对象
            return lambda u: dist.ppf(u)  # 返回分位函数
        else:  # 未显式给出μσ，则由[L,U]按95%覆盖推断
            mu, sigma = tn_params_from_bounds(L, U)  # 由边界求μσ
            a, b = (L - mu)/sigma, (U - mu)/sigma  # 标准化截断区间
            dist = truncnorm(a, b, loc=mu, scale=sigma)  # 构造截断正态
            return lambda u: dist.ppf(u)  # 返回分位函数

    elif dist_type == "LN_TRUNC":  # 截断对数正态
        if "rel_unc_95" in params:  # 若依据给出相对不确定度±x%（95%覆盖）
            mu_ln, sigma_ln = ln_params_from_rel_unc(L, U, float(params["rel_unc_95"]))  # 由x与[L,U]求μ_ln与σ_ln
            dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))  # 构造LN分布
            return truncated_ppf_of(dist, L, U)  # 返回在[L,U]截断后的分位函数
        elif all(k in params for k in ("mu_ln","sigma_ln","L","U")):  # 若显式给出μ_ln/σ_ln/截断区间
            mu_ln, sigma_ln = float(params["mu_ln"]), float(params["sigma_ln"])  # 读取μ_ln与σ_ln
            Lp, Up = float(params["L"]), float(params["U"])  # 读取截断区间
            dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))  # 构造LN分布
            return truncated_ppf_of(dist, Lp, Up)  # 返回在给定[Lp,Up]截断的分位函数
        else:  # 未显式给出x或μ_ln/σ_ln时，回退为由[L,U]当作95%区间推断
            mu_ln, sigma_ln = ln_params_from_bounds(L, U)  # 由边界推断对数正态参数
            dist = lognorm(s=sigma_ln, scale=math.exp(mu_ln))  # 构造LN分布
            return truncated_ppf_of(dist, L, U)  # 返回截断后的分位函数
    else:  # 未知分布类型
        raise ValueError(f"未知分布类型: {dist_type} for {component}.{s}")  # 抛出异常提示配置错误

# ======================== LHS 抽样与文件读写（功能块） ==========================  # 说明：LHS采样、文件加载与结果写出
def lhs_u01(n: int, d: int, seed: int = 2025) -> np.ndarray:  # 定义：生成n×d的U(0,1)拉丁超立方样本
    """生成 n×d 的拉丁超立方样本（[0,1]）。"""  # 文档字符串：函数说明
    engine = qmc.LatinHypercube(d=d, seed=seed)  # 构造LHS引擎，维度d，随机种子seed
    return engine.random(n)  # 生成n行样本，返回ndarray

def load_params_file(path: str):  # 定义：加载参数JSON文件
    """读取 JSON 数组结构的参数文件。"""  # 文档字符串：函数说明
    with open(path, "r", encoding="utf-8") as f:  # 打开文件，UTF-8编码读取
        try:
            data = json.load(f)  # 解析JSON为Python对象
        except json.JSONDecodeError as e:  # 捕获JSON解析错误
            raise RuntimeError(f"无法解析 JSON：{path}\n{e}")  # 抛出更友好的错误并给出文件路径
    return data  # 返回解析后的数据（list[dict]）

def sample_one_component(params_path: str, component: str, n_samples: int, out_path: str, seed: int):  # 定义：对单一组件进行LHS采样并写出
    params = load_params_file(params_path)  # 加载参数定义JSON
    # 从文件里拿到 bounds；确保 ORDER 中的每个符号都存在  # 检查与整理边界信息
    bounds_map = {p["symbol"].upper(): tuple(p["bounds"]) for p in params}  # 构建symbol→(L,U)映射
    missing = [s for s in ORDER if s not in bounds_map]  # 检查是否有ORDER中需要的symbol缺失
    if missing:  # 若存在缺失
        raise ValueError(f"{component} 文件缺少参数：{missing}")  # 终止并报错，标明缺失字段

    # 为每个参数构造分位函数（按“依据”自动选分布并求参数）  # 生成各列的ppf映射函数
    ppfs = [make_ppf(sym, bounds_map[sym], component) for sym in ORDER]  # 顺序与ORDER一致

    # 生成 LHS 的 U(0,1) 样本，并映射到对应分布  # 从U[0,1]样本映射至实际物理量
    U = lhs_u01(n_samples, len(ORDER), seed=seed)  # 生成n×d的超立方均匀样本
    X_cols = [ppfs[j](U[:, j]) for j in range(len(ORDER))]  # 对每一列应用对应ppf得到采样值
    X = np.column_stack(X_cols)  # 合并列为n×d矩阵

    # 写出 .tsv，表头两行与示例一致  # 输出文件写入阶段
    header = "# " + "\t".join(ORDER)  # 构造首行标题，# 开头并以tab分隔
    with open(out_path, "w", encoding="utf-8") as f:  # 以写模式打开输出文件
        f.write(header + "\n")  # 写入标题行
        f.write(UNITS_LINE + "\n")  # 写入单位行
        for row in X:  # 逐行写入数据
            f.write("\t".join(f"{v:.4e}" for v in row) + "\n")  # 科学计数法格式化每个值，并用tab分隔

    print(f"[OK] {component} -> {out_path} ({n_samples} rows)")  # 控制台回报生成成功与样本数

# ============================== 主流程（功能块） ================================  # 说明：依次处理三个组件，保持种子不同以可复现实验
if __name__ == "__main__":  # 仅当作为脚本直接运行时执行主流程
    # 三个组件各抽样 N_SAMPLES 条，可修改 seed 复现实验  # 逐组件调用采样函数
    sample_one_component(PATH_CR,  "control_rod", N_SAMPLES, OUT_CR, seed=1003)  # 控制棒：采样并输出
    sample_one_component(PATH_CL,  "cladding",    N_SAMPLES, OUT_CL, seed=1002)  # 包壳：采样并输出
    sample_one_component(PATH_UO2, "fuel",        N_SAMPLES, OUT_UO2, seed=1001)  # 燃料：采样并输出
