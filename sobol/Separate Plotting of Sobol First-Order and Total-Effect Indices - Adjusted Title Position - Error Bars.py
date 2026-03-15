# -*- coding: utf-8 -*-  # 指定源码编码，保证中文注释不乱码  # Python 3.10 兼容  # 无副作用
"""
功能（本版修正点）：
    - 读取样本与结果，进行 Sobol 敏感性分析，并分别绘制“一阶 S1 网格图”和“总效应 ST 网格图”（3行×3列：行=材料，列=输出）；
    - 关键修复：自动识别 Y 长度对应的抽样模式（含二阶 or 不含二阶），
      若 Y 长度为 N*(2D+2) 则按含二阶计算；若为 N*(D+2) 则按不含二阶计算；
      并据此自动传入 sobol.analyze(calc_second_order=...)，避免长度不匹配报错；
    - 每个子图横轴为全部输入变量，纵轴为相应敏感性指数，含基于 Sobol analyze 内部 bootstrap 重抽样得到的 95% 置信区间；
      若指数全为 NaN，则在子图内给出中文说明。

使用方式：
    1) 确认库版本：SALib==1.4.7, numpy==1.23.5, pandas==1.5.3, matplotlib(兼容版本)；
    2) 修改“用户可配置区”的路径、节点与时间选择；
    3) 运行脚本后，在 fig_save_dir 目录下得到两张图片：S1_grid.png、ST_grid.png；
    4) 控制台会打印：样本 D、变量名、样本行数、识别到的抽样模式（含/不含二阶）、以及 N 的值；
    5) 若看到 RuntimeWarning: invalid value encountered in divide，表示某个输出在所选时间/节点上为常数；该子图会提示 NaN。

注意：
    - 结果文件命名需满足“样本序号 + 材料 + 输出 + (可选扩展名)”的形式，例如：0UO2mLeav 或 15CRmFrozen.txt；
    - 本版允许两种长度公式：N*(2D+2) 或 N*(D+2)。若两者都能整除，默认优先按“含二阶”处理；
    - 若想强制某一种模式，可在 main() 里固定 calc_second_order_force（见“可选强制开关”）。
"""

from __future__ import annotations  # 注解前置兼容  # Python 3.10 可用  # 无副作用
import os  # 操作系统路径与目录  # 标准库
import re  # 正则解析  # 标准库
from pathlib import Path  # 跨平台路径  # 标准库
from typing import Dict, List, Tuple, Optional  # 类型注解  # 标准库

import numpy as np  # 数值计算  # 版本 1.23.5
import pandas as pd  # 表格与读取  # 版本 1.5.3
import matplotlib.pyplot as plt  # 画图  # 兼容版本

from SALib.analyze import sobol  # Sobol 分析  # 版本 1.4.7

# --------- 字体与渲染设置（在 Windows 上尽量避免中文缺字告警） ----------
plt.rcParams["font.family"] = ['Times New Roman']  # 依次尝试宋体/雅黑/DejaVu  # 尽量保证中文可见
plt.rcParams["axes.unicode_minus"] = False  # 允许坐标轴显示负号  # 防止负号变成方块

# ========================= 用户可配置区 =========================

sample_file = r"C:\Users\11201\Desktop\sensities\sobol\sobol_samples.txt"  # 样本文件路径  # 必须存在
result_root = r"C:\Users\11201\Desktop\sensities\Candle_To_Analysize\sobol"  # 结果根目录  # 必须存在

materials = ["UO2", "CL", "CR"]  # 三种材料  # 必须与目录名一致
outputs   = ["mLeav", "mFrozen", "mAccu"]  # 三种输出  # 必须与目录名与文件名一致

material_title_map = {"UO2": "Fuel pellet", "CL": "Fuel cladding", "CR": "Control rod"}  # 行标题显示映射（仅用于显示，不影响数据读取）  # 可修改
output_title_map = {"mLeav": "FOM1", "mFrozen": "FOM2", "mAccu": "FOM3"}  # 列标题显示映射（仅用于显示，不影响数据读取）  # 可修改

# 统一的节点与时间选择（固定一个节点与时间点抽取标量响应）
node_i, node_j = 0, 0  # 节点索引  # 可按需修改
time_select_mode = "index"  # 'index' 或 'value'  # 默认按列索引选
time_index = 1             # 当按索引选时的列号（0 表示第一个时间点）  # 可修改
time_value = 1.0           # 当按时间值选时的目标值（会取最近的列）  # 可修改

# 随机种子（影响置信区间计算中的随机过程）
RANDOM_SEED = 12345  # 固定随机种子  # 可修改

# bootstrap 误差条参数（基于已有 Y 的重抽样，不需要重新运行外部程序）
BOOTSTRAP_NUM_RESAMPLES = 1000  # bootstrap 重抽样次数  # 可修改
BOOTSTRAP_CONF_LEVEL = 0.95     # 置信水平  # 可修改
BOOTSTRAP_KEEP_RESAMPLES = False  # 是否保留所有重抽样结果  # 一般无需开启
PLOT_ERROR_BARS = True          # 是否在柱状图上绘制 bootstrap 误差条  # 可修改
ERROR_BAR_CAPSIZE = 3           # 误差条端帽长度  # 可修改

# 是否在运行时弹出图形窗口（脚本中通常 False，仅保存到文件）
SHOW_FIG = True  # 不弹窗  # 如需交互查看可改 True

# 输出图保存目录（建议为不带扩展名的目录）
fig_save_dir = Path(r"C:\Users\11201\Desktop\sensities\sobol\sobol_figures")  # 若不存在将自动创建

# 可选强制开关：如需强制某种模式，可将其设为 True/False；否则设为 None 让程序自动识别
calc_second_order_force: Optional[bool] = None  # None=自动识别；True=强制含二阶；False=强制不含二阶

y_label_map = {  # 仅用于显示，不影响数据读取
    "S1": r"$S_1$",     # 或 "First-order Sobol index"
    "ST": r"$S_T$",     # 或 "Total-effect Sobol index"
}

FIG_NOTE_TEXT = "$S_1$ is the Sobol first-order sensitivity indices."  # 设为空字符串则不添加图中说明


# ========================= 功能函数（读取/解析） =========================

def read_sample_header_and_matrix(sample_path: str) -> Tuple[List[str], np.ndarray]:
    """读取样本文件，返回变量名列表（按列顺序）与样本矩阵（float）。"""
    header_names: List[str] = []  # 存放变量名
    data_lines: List[str] = []    # 存放数值行
    tried_encodings = ["utf-8", "latin-1"]  # 逐个尝试编码
    last_err = None  # 记录错误
    for enc in tried_encodings:  # 遍历编码列表
        try:
            with open(sample_path, "r", encoding=enc) as f:  # 打开样本文件
                lines = f.readlines()  # 读取全部行
            last_err = None  # 打开成功
            break  # 不再尝试其他编码
        except Exception as e:  # 打开失败
            last_err = e  # 记录异常，继续下一编码
    if last_err is not None:  # 若两种编码都失败
        raise RuntimeError(f"无法读取样本文件：{sample_path}，错误：{last_err}")  # 抛错

    # 解析变量名（取第一条以 '#' 开头且不含 'Unit' 的行）
    for line in lines:  # 遍历各行
        if line.strip().startswith("#"):  # 仅处理注释行
            if "Unit" in line:  # 跳过单位行
                continue  # 继续找变量名行
            tokens = [t for t in re.split(r"[\s\t]+", line.strip().lstrip("#").strip()) if t]  # 分割并去空
            header_names = tokens  # 变量名列表
            break  # 已找到变量名行
    if not header_names:  # 若没找到变量名行
        raise ValueError("样本文件未解析到变量名表头（以 '#' 开头且不含 'Unit' 的那一行）。")  # 抛错

    # 收集非注释的数据行
    for line in lines:  # 遍历各行
        s = line.strip()  # 去空白
        if not s or s.startswith("#"):  # 空行或注释行
            continue  # 跳过
        data_lines.append(s)  # 收集到数据列表

    # 将数据行解析为二维数组
    try:
        data = np.array([[float(x) for x in re.split(r"[\s\t]+", ln)] for ln in data_lines], dtype=float)  # 数值化
    except Exception as e:  # 解析失败
        raise ValueError(f"样本文件数值解析失败：{e}")  # 抛错

    # 校验列数一致
    if data.shape[1] != len(header_names):  # 列数不等
        raise ValueError(f"样本列数({data.shape[1]})与变量名数({len(header_names)})不一致。")  # 抛错

    return header_names, data  # 返回变量名与样本矩阵


def parse_time_header(time_line: str) -> np.ndarray:
    """解析结果文件首行 'time 0 1 2 ...'，返回时间数组。"""
    tokens = [t for t in re.split(r"[\s\t]+", time_line.strip()) if t]  # 分割
    if not tokens:  # 空行
        raise ValueError("结果文件首行（时间轴）为空或格式异常。")  # 抛错
    if tokens[0].lower() != "time":  # 首 token 非 time
        raise ValueError(f"结果文件首行应以 'time' 开头，实际为：{tokens[0]}")  # 抛错
    try:
        times = np.array([float(x) for x in tokens[1:]], dtype=float)  # 转 float 数组
    except Exception as e:  # 转换失败
        raise ValueError(f"时间轴解析失败：{e}")  # 抛错
    return times  # 返回时间数组


def pick_time_index(times: np.ndarray, mode: str, t_index: int, t_value: float) -> Tuple[int, float]:
    """根据模式选择时间列：'index' 取 t_index；'value' 取与 t_value 最接近的列。返回 (列索引, 实际时间)。"""
    if mode == "index":  # 按索引
        if t_index < 0 or t_index >= len(times):  # 越界
            raise IndexError(f"时间索引 {t_index} 越界（共有 {len(times)} 个时间点）。")  # 抛错
        return t_index, float(times[t_index])  # 返回索引与时间值
    elif mode == "value":  # 按时间值
        idx = int(np.argmin(np.abs(times - t_value)))  # 最近的时间索引
        return idx, float(times[idx])  # 返回最近索引与实际时间
    else:  # 非法模式
        raise ValueError("time_select_mode 只能为 'index' 或 'value'。")  # 抛错


def extract_value_from_result_file(
    file_path: Path,
    var_name: str,
    node_i: int,
    node_j: int,
    time_mode: str,
    time_index: int,
    time_value: float,
) -> float:
    """在单个结果文件中，读取 'var_name(i,j)' 这一行的指定时间列值。"""
    tried_encodings = ["utf-8", "latin-1"]  # 尝试两种常见编码
    content: List[str] = []  # 文件内容
    last_err = None  # 记录异常
    for enc in tried_encodings:  # 逐个尝试
        try:
            with open(file_path, "r", encoding=enc) as f:  # 打开文件
                content = f.readlines()  # 读全部行
            last_err = None  # 成功
            break  # 停止尝试
        except Exception as e:  # 失败
            last_err = e  # 记录并继续
    if last_err is not None:  # 打开失败
        raise RuntimeError(f"无法读取结果文件：{file_path}，错误：{last_err}")  # 抛错
    if not content:  # 空文件
        raise ValueError(f"结果文件为空：{file_path}")  # 抛错

    times = parse_time_header(content[0])  # 解析首行时间轴
    col_idx, _t = pick_time_index(times, time_mode, time_index, time_value)  # 确定使用的时间列索引

    target_label = f"{var_name}({node_i},{node_j})"  # 目标变量标签，例如 mLeav(0,0)
    for line in content[1:]:  # 跳过首行
        s = line.strip()  # 去空白
        if not s:  # 空行
            continue  # 跳过
        if not s.startswith(target_label):  # 非目标行
            continue  # 跳过
        tokens = [t for t in re.split(r"[\s\t]+", s) if t]  # 分割
        if len(tokens) < 2:  # 只有标签无数据
            raise ValueError(f"结果文件行缺少数值：{file_path} 内的 '{s}'")  # 抛错
        values = np.array([float(x) for x in tokens[1:]], dtype=float)  # 数值数组
        if col_idx < 0 or col_idx >= len(values):  # 列索引越界
            raise IndexError(f"在文件 {file_path} 中，时间列索引 {col_idx} 越界（共有 {len(values)} 列）。")  # 抛错
        return float(values[col_idx])  # 返回标量值

    # 没有找到目标变量行
    raise ValueError(f"在文件 {file_path} 未找到变量行：'{target_label}'。")  # 抛错


def collect_Y_for_material_output(
    material: str,
    output_var: str,
    base_dir: str,
    node_i: int,
    node_j: int,
    time_mode: str,
    time_index: int,
    time_value: float,
) -> Tuple[np.ndarray, List[int], Tuple[int, float]]:
    """在 base_dir/material/output_var 目录下收集所有样本序号对应的 Y（一维数组）。"""
    subdir = Path(base_dir) / material / output_var  # 拼出目录
    if not subdir.exists() or not subdir.is_dir():  # 目录不存在
        raise FileNotFoundError(f"目录不存在：{subdir}")  # 抛错

    # 匹配文件名：以数字开头 + 材料名 + 输出名 + 可选扩展
    pattern = re.compile(r"^(\d+)" + re.escape(material) + re.escape(output_var) + r"$")  # 无扩展
    pattern_ext = re.compile(r"^(\d+)" + re.escape(material) + re.escape(output_var) + r"\.[A-Za-z0-9_.-]+$")  # 带扩展

    index_to_file: Dict[int, Path] = {}  # 样本序号 -> 文件路径
    for entry in os.listdir(subdir):  # 遍历目录
        p = subdir / entry  # 完整路径
        if not p.is_file():  # 仅处理文件
            continue  # 跳过
        m = pattern.match(entry)  # 尝试匹配无扩展
        if m is None:  # 不匹配
            m = pattern_ext.match(entry)  # 尝试匹配带扩展
        if m is None:  # 仍不匹配
            continue  # 跳过
        idx = int(m.group(1))  # 解析样本序号
        index_to_file[idx] = p  # 记录映射

    if not index_to_file:  # 没有符合的文件
        raise FileNotFoundError(f"目录 {subdir} 下未找到任何匹配 '{material}{output_var}' 模式的结果文件。")  # 抛错

    all_indices = sorted(index_to_file.keys())  # 已有样本序号列表
    expected_max = all_indices[-1]  # 最大序号
    expected_indices = list(range(expected_max + 1))  # 期望从 0 到 max
    missing = sorted(set(expected_indices) - set(all_indices))  # 缺失序号列表

    # 读取一次代表文件，确定时间列索引（避免每个文件重复寻找最近时间）
    repr_file = index_to_file[all_indices[0]]  # 取最小序号文件
    tried_encodings = ["utf-8", "latin-1"]  # 尝试编码
    lines = None  # 缓存
    last_err = None  # 异常
    for enc in tried_encodings:  # 逐个尝试
        try:
            with open(repr_file, "r", encoding=enc) as f:  # 打开代表文件
                lines = f.readlines()  # 读行
            last_err = None  # 成功
            break  # 停止尝试
        except Exception as e:  # 失败
            last_err = e  # 记录
    if last_err is not None:  # 全部失败
        raise RuntimeError(f"无法读取代表文件：{repr_file}，错误：{last_err}")  # 抛错
    if not lines:  # 空文件
        raise ValueError(f"代表文件为空：{repr_file}")  # 抛错

    times = parse_time_header(lines[0])  # 解析时间轴
    used_col_idx, used_time_val = pick_time_index(times, time_mode, time_index, time_value)  # 选取时间列

    Y = np.full(shape=(expected_max + 1,), fill_value=np.nan, dtype=float)  # 预分配 Y，先填 NaN
    for idx in all_indices:  # 遍历已有样本序号
        fp = index_to_file[idx]  # 文件路径
        val = extract_value_from_result_file(  # 解析标量响应
            fp, output_var, node_i, node_j, time_mode, used_col_idx, used_time_val
        )  # 统一用同一个时间列
        Y[idx] = val  # 写入对应位置

    return Y, missing, (used_col_idx, used_time_val)  # 返回 Y、缺失与时间信息


def build_sobol_problem(var_names: List[str]) -> Dict:
    """构造 SALib 的 problem 字典（bounds 在 analyze 阶段不使用，这里占位 [0,1]）。"""
    D = len(var_names)  # 变量数
    return {"num_vars": D, "names": var_names, "bounds": [[0.0, 1.0]] * D}  # 返回结构


def check_sobol_length_and_mode(Y_len: int, D: int, force: Optional[bool] = None) -> Tuple[int, bool]:
    """
    校验 Y 的长度并确定抽样模式，返回 (N, calc_second_order)。
    - 若 force 为 True/False，则强制采用对应模式并按该模式校验；
    - 若 force 为 None：先尝试含二阶（N*(2D+2)），再尝试不含二阶（N*(D+2)）；
      若两者都可整除则优先采用“含二阶”。
    """
    denom_2nd = 2 * D + 2  # 含二阶的分母
    denom_1st = D + 2      # 不含二阶的分母

    # 若强制模式不为空，直接按强制模式校验
    if force is True:  # 强制含二阶
        if Y_len % denom_2nd != 0:
            raise ValueError(f"Y 长度={Y_len} 不满足 N*(2D+2)，D={D} -> 分母={denom_2nd}（已强制含二阶）。")  # 抛错
        N = Y_len // denom_2nd  # 基数
        if N <= 0:
            raise ValueError("计算得到的基数 N<=0，请检查输入（强制含二阶）。")  # 抛错
        return N, True  # 返回
    if force is False:  # 强制不含二阶
        if Y_len % denom_1st != 0:
            raise ValueError(f"Y 长度={Y_len} 不满足 N*(D+2)，D={D} -> 分母={denom_1st}（已强制不含二阶）。")  # 抛错
        N = Y_len // denom_1st  # 基数
        if N <= 0:
            raise ValueError("计算得到的基数 N<=0，请检查输入（强制不含二阶）。")  # 抛错
        return N, False  # 返回

    # 自动识别：优先尝试“含二阶”
    if Y_len % denom_2nd == 0:  # 能整除含二阶
        N = Y_len // denom_2nd  # 计算 N
        if N <= 0:
            raise ValueError("计算得到的基数 N<=0，请检查输入。")  # 抛错
        return N, True  # 返回含二阶
    # 再尝试“不含二阶”
    if Y_len % denom_1st == 0:  # 能整除不含二阶
        N = Y_len // denom_1st  # 计算 N
        if N <= 0:
            raise ValueError("计算得到的基数 N<=0，请检查输入。")  # 抛错
        return N, False  # 返回不含二阶

    # 两种模式都不满足时，给出更友好的诊断
    r1 = Y_len % denom_2nd  # 含二阶余数
    r2 = Y_len % denom_1st  # 不含二阶余数
    raise ValueError(
        f"Y 长度={Y_len} 对 D={D} 既不满足含二阶 N*(2D+2)={denom_2nd}（余数 {r1}），也不满足不含二阶 N*(D+2)={denom_1st}（余数 {r2}）。"
        "请检查样本/结果是否一一对应或是否混用了不同抽样设置。"
    )  # 抛错


# ========================= 新版绘图（3×3：行=材料，列=输出） =========================

def plot_index_grid_by_material_output(
    results_all: Dict[str, Dict[str, Dict[str, np.ndarray]]],  # 全部 Sobol 指标
    var_names: List[str],                                     # 全部输入变量名（按列顺序）
    materials: List[str],                                     # 行标签
    outputs: List[str],                                       # 列标签
    index_key: str,                                           # "S1" 或 "ST"
    fig_title: str,                                           # 整体标题
    save_path: Path,                                          # 保存路径
    material_title_map: Optional[Dict[str, str]] = None,       # 行标题显示映射（仅用于显示，不影响数据读取）  # 可选
    output_title_map: Optional[Dict[str, str]] = None,         # 列标题显示映射（仅用于显示，不影响数据读取）  # 可选
) -> None:
    """绘制一幅 3×3 网格图：行=材料，列=输出；每个子图横轴为全部输入变量的 {index_key}。"""
    n_rows = len(materials)  # 行数（3）
    n_cols = len(outputs)    # 列数（3）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.4 * n_rows), constrained_layout=False)  # 建画布
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.05, top=0.95, wspace=0.18, hspace=0.20)

    # 统一把 axes 处理成二维数组（即使只有 1×n 或 n×1）
    if n_rows == 1 and n_cols == 1:  # 1×1 情况
        axes = np.array([[axes]])  # 转二维
    elif n_rows == 1:  # 1×n
        axes = np.array([axes])  # 转二维
    elif n_cols == 1:  # n×1
        axes = np.array([[ax] for ax in axes])  # 转二维

    material_title_map = {} if material_title_map is None else material_title_map  # 行标题映射兜底  # 不影响读取
    output_title_map = {} if output_title_map is None else output_title_map  # 列标题映射兜底  # 不影响读取

    x = np.arange(len(var_names))  # x 轴位置（0..D-1）
    for r, mat in enumerate(materials):  # 遍历材料（行）
        for c, out in enumerate(outputs):  # 遍历输出（列）
            ax = axes[r, c]  # 取当前子图
            panel_idx = r * n_cols + c  # 子图编号（0..8）  # 用于(a)-(i)
            panel_tag = f"({chr(ord('a') + panel_idx)})"  # 子图标注(a)-(i)  # 显示用
            mat_title = material_title_map.get(mat, mat)  # 行标题显示文本  # 不影响读取
            out_title = output_title_map.get(out, out)  # 列标题显示文本  # 不影响读取
            vals = results_all[mat][out].get(index_key, None)  # 取该材料-输出的 S1 或 ST
            conf = results_all[mat][out].get(f"{index_key}_conf", None)  # 取对应置信区间

            # 判空与 NaN 处理：如果 vals 不存在或全部为 NaN，给出显式文字，不再“空白”
            if vals is None or np.all(~np.isfinite(vals)):  # 都不是有限数（NaN/Inf）
                ax.text(0.5, 0.5,
                        f"{mat} - {out}\n该指数全为 NaN/无效\n(可能是该输出在所选时间/节点为常数)",
                        ha="center", va="center", fontsize=12)  # 居中文本说明
                ax.set_xticks([])  # 不显示刻度
                ax.set_yticks([])  # 不显示刻度
                ax.text(0.03, 0.97, panel_tag, transform=ax.transAxes, ha="left", va="top", fontsize=14)  # 子图内标注(a)-(i)
                ax.set_frame_on(True)  # 保留边框
                if r == n_rows - 1:  # 仅最底层子图添加横坐标名称
                    ax.set_xlabel("Input parameters", fontsize=12)  # 设置 x 轴名称（仅底行）
                continue  # 下一个子图

            # 将 NaN 替换为 0 以保证可以画条形，同时保留一个掩码用于标注
            vals = np.asarray(vals, dtype=float)  # 转 np 数组
            finite_mask = np.isfinite(vals)  # 有效值掩码
            vals_plot = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)  # NaN/Inf->0，保证能画
            yerr = None  # 误差条默认无
            if PLOT_ERROR_BARS and conf is not None:  # 有置信区间且允许绘制误差条
                conf = np.asarray(conf, dtype=float)  # 转数组
                conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)  # 清洗 NaN
                yerr = conf  # 使用同顺序的误差

            # 绘制单组条形（横轴为全部输入变量）
            ax.bar(x, vals_plot, yerr=yerr, capsize=ERROR_BAR_CAPSIZE)  # 单组条形，带误差帽

            # 对于原始为 NaN 的位置，在条形上方添加小注释“NaN”
            for xi, is_ok, yv in zip(x, finite_mask, vals_plot):  # 遍历每个变量
                if not is_ok:  # 原值是 NaN
                    ax.text(xi, yv + 0.02, "NaN", ha="center", va="bottom", fontsize=12, rotation=90)  # 标注 NaN

            # 坐标轴与标题（自适应上下界，允许出现负值）
            ax.set_xticks(x)  # 设置 x 刻度
            ax.set_xticklabels(var_names, rotation=0, ha="center", fontsize=12)  # 变量名倾斜显示
            if r == n_rows - 1:  # 仅最底层子图添加横坐标名称
                ax.set_xlabel("Input parameters", fontsize=12)  # 设置 x 轴名称（仅底行）
            finite_vals = vals[np.isfinite(vals)]  # 取有效值
            if finite_vals.size > 0:               # 至少有一个有效值
                vmin = float(np.nanmin(finite_vals))  # 有效值最小值
                vmax = float(np.nanmax(finite_vals))  # 有效值最大值
            else:                                  # 全无有效值时兜底
                vmin, vmax = 0.0, 1.0
            margin = 0.05 * (vmax - vmin if vmax > vmin else 1.0)  # 相对边距
            ymin = min(0.0, vmin - margin)  # 下界允许小于0
            ymax = max(1.0, vmax + margin)  # 上界至少不小于1.0
            ax.set_ylim(ymin, ymax)  # 设置上下界
            if c == 0:  # 第一列设置y轴标签
                ax.set_ylabel(y_label_map.get(index_key, index_key), fontsize=12)  # y 轴标签：S1 或 ST
            ax.text(0.03, 0.97, panel_tag, transform=ax.transAxes, ha="left", va="top", fontsize=14)  # 子图内标注(a)-(i)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.6)  # 加细网格

    # --------- 新增：按“行标题 + 列标题”的方式设置网格标题（不影响数据读取） ----------
    fig.canvas.draw()  # 触发布局计算，确保 get_position() 结果稳定  # 用于行列标题定位
    plt.tight_layout(rect=[0.00, 0.02, 0.00, 0.00], pad=0.0, w_pad=0.0, h_pad=0.0)
    for c, out in enumerate(outputs):  # 遍历列（输出）  # 顶部列标题
        out_title = output_title_map.get(out, out)  # 列标题显示文本  # 不影响读取
        pos = axes[0, c].get_position()  # 获取顶部子图位置  # figure 坐标系
        x_center = 0.5 * (pos.x0 + pos.x1)  # 计算该列中心 x 位置  # 用于居中放置标题
        y_top = pos.y1 + 0.01  # 在该列顶部略上方放置标题  # 与示例图一致
        fig.text(x_center, y_top, out_title, ha="center", va="bottom", fontsize=14)  # 添加列标题  # 不影响读取
    for r, mat in enumerate(materials):  # 遍历行（材料）  # 左侧行标题
        mat_title = material_title_map.get(mat, mat)  # 行标题显示文本  # 不影响读取
        pos = axes[r, 0].get_position()  # 获取该行左侧子图位置  # figure 坐标系
        y_center = 0.5 * (pos.y0 + pos.y1)  # 计算该行中心 y 位置  # 用于垂直居中放置标题
        x_left = pos.x0 - 0.055  # 在该行左侧略偏左放置标题  # 与示例图一致
        if x_left < 0.0:  # 若偏移后超出画布  # 防止标题被截断
            x_left = 0.005  # 退回到画布内侧  # 保证可见
            fig.text(x_left, y_center, mat_title, ha="left", va="center", fontsize=16)  # 添加行标题  # 不影响读取
        else:  # 正常情况下  # 采用右对齐贴近子图
            fig.text(x_left, y_center, mat_title, ha="right", va="center", fontsize=16)  # 添加行标题  # 不影响读取

    # 去掉整幅图的标题，不再显示
    # fig.suptitle(fig_title, fontsize=14)  # 整体标题（已注释，不显示）
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    _note_artist = None
    #if isinstance(FIG_NOTE_TEXT, str) and FIG_NOTE_TEXT.strip():
    #    _note_artist = fig.text(0.5, 0.02, FIG_NOTE_TEXT.strip(), ha="center", va="bottom", fontsize=10)
    if _note_artist is None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight")  # 保存到文件
    else:
        fig.savefig(save_path, dpi=600, bbox_inches="tight", bbox_extra_artists=[_note_artist])  # 保存到文件
    if SHOW_FIG:  # 如需弹出显示
        plt.show()  # 展示
    plt.close(fig)  # 关闭图释放内存


# ========================= 主流程（读取 → 分析 → 绘图） =========================

def main() -> None:
    """读取样本 → 组装 Y → 自动识别 Sobol 抽样模式 → 分析 → 画两张 3×3 图（S1 与 ST）。"""
    # 1) 读取样本文件，得到输入变量名与样本矩阵（用于规模校验）
    var_names, X = read_sample_header_and_matrix(sample_file)  # 返回变量名与矩阵
    D = len(var_names)  # 输入维数
    print(f"[INFO] 样本文件已读取：D={D}，变量名={var_names}，样本行数={X.shape[0]}")  # 日志

    # 2) 构建 SALib problem（bounds 在 analyze 阶段不使用，这里占位即可）
    problem = build_sobol_problem(var_names)  # problem 结构
    print(f"[INFO] Bootstrap 设置：num_resamples={BOOTSTRAP_NUM_RESAMPLES}，conf_level={BOOTSTRAP_CONF_LEVEL}，plot_error_bars={PLOT_ERROR_BARS}")  # 日志
    results_all: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}  # 存放各材料-输出的 Sobol 结果

    # 3) 收集各材料/输出的 Y，做长度校验（自动识别模式），然后调用 sobol.analyze
    used_time_info_global = None  # 记录一次时间列与值
    for mat in materials:  # 遍历材料
        results_all[mat] = {}  # 初始化该材料的字典
        for out in outputs:  # 遍历输出
            Y, missing, used_time_info = collect_Y_for_material_output(  # 收集 Y
                mat, out, result_root, node_i, node_j, time_select_mode, time_index, time_value
            )  # 返回 Y、缺失、实际时间信息
            if used_time_info_global is None:  # 仅记录一次时间信息
                used_time_info_global = used_time_info  # 赋值
            if missing:  # 若有缺失样本
                raise ValueError(f"[ERROR] {mat}-{out} 缺失样本序号：{missing}，请补齐后再分析。")  # 抛错

            # 自动识别/或强制设置 抽样模式，并计算 N
            N, calc_second = check_sobol_length_and_mode(len(Y), D, force=calc_second_order_force)  # 自动识别
            mode_str = "含二阶（需要 N*(2D+2)）" if calc_second else "不含二阶（需要 N*(D+2)）"  # 文字说明
            print(f"[INFO] 材料={mat} 输出={out}：Y 长度={len(Y)} → N={N}，识别为{mode_str}")  # 打印日志

            # 与识别结果一致地调用 SALib
            Si = sobol.analyze(  # 调用 SALib 分析
                problem=problem,         # 变量定义
                Y=Y,                     # 模型输出（标量）
                calc_second_order=calc_second,  # 是否计算二阶（与长度公式一致）
                num_resamples=BOOTSTRAP_NUM_RESAMPLES,  # bootstrap 重抽样次数
                conf_level=BOOTSTRAP_CONF_LEVEL,  # bootstrap 置信水平
                keep_resamples=BOOTSTRAP_KEEP_RESAMPLES,  # 是否保留全部 bootstrap 结果
                print_to_console=False,  # 不打印到控制台
                seed=RANDOM_SEED,        # 固定随机种子
            )  # 返回包含 S1/ST/S2 及其 conf 的字典（若不含二阶，S2 相关键可能缺失）

            # 存入结果（统一为 numpy 数组，便于索引）
            Dloc = D  # 局部变量方便书写
            results_all[mat][out] = {
                "S1":      np.asarray(Si.get("S1",      np.full(Dloc,    np.nan))),  # 一阶
                "S1_conf": np.asarray(Si.get("S1_conf", np.full(Dloc,    np.nan))),  # 一阶置信区间
                "ST":      np.asarray(Si.get("ST",      np.full(Dloc,    np.nan))),  # 总效应
                "ST_conf": np.asarray(Si.get("ST_conf", np.full(Dloc,    np.nan))),  # 总效应置信区间
                "S2":      np.asarray(Si.get("S2",      np.full((Dloc,Dloc),np.nan))),  # 二阶（若未计算则为 NaN 矩阵）
                "S2_conf": np.asarray(Si.get("S2_conf", np.full((Dloc,Dloc),np.nan))),  # 二阶置信区间（同上）
            }  # 完成存储

    # 统一打印时间选择信息
    if used_time_info_global is not None:  # 已获取
        col_idx, t_val = used_time_info_global  # 解包
        mode_desc = f"按索引 time[{col_idx}]" if time_select_mode == "index" else f"按最接近时间值 {time_value}（实际用 {t_val}）"
        print(f"[INFO] 统一时间选择：{mode_desc}")  # 日志

    # 4) 仅绘制两幅图：S1 与 ST（3×3：行=材料，列=输出）
    fig_save_dir.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在
    plot_index_grid_by_material_output(  # 绘制 S1
        results_all=results_all, var_names=var_names, materials=materials, outputs=outputs,
        index_key="S1", fig_title="一阶敏感性指数 S1（行=材料，列=输出；横轴=全部输入变量）",
        save_path=fig_save_dir / "S1_grid.png",
        material_title_map=material_title_map, output_title_map=output_title_map,
    )  # 保存到 S1_grid.png

    plot_index_grid_by_material_output(  # 绘制 ST
        results_all=results_all, var_names=var_names, materials=materials, outputs=outputs,
        index_key="ST", fig_title="总效应敏感性指数 ST（行=材料，列=输出；横轴=全部输入变量）",
        save_path=fig_save_dir / "ST_grid.png",
        material_title_map=material_title_map, output_title_map=output_title_map,
    )  # 保存到 ST_grid.png

    print(f"[INFO] 图像已保存到：{fig_save_dir.resolve()}  (S1_grid.png, ST_grid.png)")  # 结束提示


# ------------------------------- 入口 --------------------------------
if __name__ == "__main__":  # 脚本入口
    main()  # 执行主流程