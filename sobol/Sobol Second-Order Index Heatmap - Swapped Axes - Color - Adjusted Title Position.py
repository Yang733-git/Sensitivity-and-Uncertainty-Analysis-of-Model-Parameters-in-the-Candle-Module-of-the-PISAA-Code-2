# -*- coding: utf-8 -*-  # 文件编码声明，确保中文正常显示
import numpy as np  # 数值计算库
import pandas as pd  # 表格数据处理库
import os  # 文件与路径操作库
import matplotlib.pyplot as plt  # 绘图主接口
from matplotlib.gridspec import GridSpec  # 精细网格布局
from SALib.analyze import sobol  # Sobol 敏感性分析
import seaborn as sns  # 热力图等高级绘图
import warnings  # 警告控制
warnings.filterwarnings('ignore')  # 关闭非关键警告，避免终端干扰
import matplotlib as mpl  # Matplotlib 全局配置

# 设置中文字体支持，防止坐标/标题中文乱码（Python 3.10 兼容）
mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置中文优先字体
mpl.rcParams['axes.unicode_minus'] = False  # 处理负号显示

# === 配置全局参数（保持原有路径与含义不变） ===
SAMPLE_PATH = "C:/Users/11201/Desktop/sensities/sobol/sobol_samples.txt"  # Sobol 样本文件路径
RESULT_BASE = "C:/Users/11201/Desktop/sensities/Candle_To_Analysize/sobol"  # 仿真结果根目录
MATERIALS = ["UO2", "CL", "CR"]  # 材料列表（顺序即三行的顺序）
OUTPUT_VARS = ["mLeav", "mFrozen", "mAccu"]  # 输出变量列表（顺序即三列的顺序）
NODE_INDEX = (0, 0)   # 固定节点索引：(行,列)  # 选取仿真输出的网格点
TIME_INDEX = -1        # 最后时间点  # 取时间序列末尾的值

ROW_TITLES = ["Fuel pellet", "Fuel cladding", "Control rod"]  # 行标题（仅用于显示；可修改且不影响数据读取）
COL_TITLES = ["FOM1", "FOM2", "FOM3"]  # 列标题（仅用于显示；可修改且不影响数据读取）

# === 读取样本文件 ===
def read_sample_file(sample_path):  # 定义读取样本文件的函数
    """安全读取样本文件，返回参数名称和数据矩阵"""  # 功能说明
    data = np.loadtxt(sample_path, skiprows=2)  # 读取数值体，跳过前两行（参数名与单位行）
    with open(sample_path, 'r', encoding='utf-8') as f:  # 打开文件用于读取表头
        param_header = next(f).strip()  # 第一行：参数名行（以#开头）
        unit_header = next(f).strip()  # 第二行：单位行（本处不使用）
    param_header = param_header[1:].strip()  # 移除开头的 # 并去空白
    param_names = param_header.split('\t')  # 以制表符切割得到参数名列表
    return param_names, data  # 返回参数名与样本矩阵

# === 解析结果文件 ===
def parse_output_file(file_path, time_index=-1, node_index=(0,0)):  # 定义解析单个输出文件函数
    """
    解析单个输出文件，返回指定位置的值
    格式:
    time 0 1 2 3...
    mLeav(0,0) 1.0 2.0 3.0...
    """  # 文档字符串，描述文件格式
    try:  # 异常捕获，避免单文件损坏中断整体流程
        with open(file_path, 'r') as f:  # 打开文件
            lines = f.readlines()  # 读取所有行
        if not lines:  # 若文件为空
            return np.nan  # 返回 NaN
        time_line = lines[0].split()  # 第一行时间轴
        time_points = [float(t) for t in time_line[1:]]  # 跳过首列标签，转为浮点时间
        if not time_points:  # 若无时间点
            return np.nan  # 返回 NaN
        if time_index < 0:  # 允许负索引（从末尾计数）
            time_index = len(time_points) + time_index  # 转为正索引
        if time_index < 0 or time_index >= len(time_points):  # 越界处理
            time_index = len(time_points) - 1  # 回退到最后一个时间点
        node_str = f"({node_index[0]},{node_index[1]})"  # 生成节点字符串表示
        for line in lines[1:]:  # 遍历数据行
            if not line.strip():  # 跳过空行
                continue  # 进入下一行
            parts = line.split()  # 按空白切分
            if len(parts) < 2:  # 非法行保护
                continue  # 进入下一行
            if node_str in parts[0]:  # 若本行是目标节点
                return float(parts[1 + time_index])  # 取对应时间点的数值并返回
    except Exception as e:  # 捕获文件解析异常
        print(f"解析错误: {file_path} - {e}")  # 打印错误信息
    return np.nan  # 无匹配或出错时返回 NaN

# === 读取材料结果 ===
def read_material_results(material_path, output_vars, sample_count):  # 定义读取某材料全部结果的函数
    """读取特定材料文件夹下的所有结果"""  # 文档说明
    results = {var: np.full(sample_count, np.nan) for var in output_vars}  # 初始化每个输出量的结果数组
    for var in output_vars:  # 遍历输出变量
        var_dir = os.path.join(material_path, var)  # 该输出量对应的文件夹
        if not os.path.exists(var_dir):  # 文件夹不存在则提示
            print(f"文件夹不存在: {var_dir}")  # 打印提示
            continue  # 跳过该输出量
        for file_name in os.listdir(var_dir):  # 遍历输出量目录下的所有文件
            file_path = os.path.join(var_dir, file_name)  # 拼接完整路径
            sample_id = -1  # 默认无效样本编号
            for i, char in enumerate(file_name):  # 从文件名前缀提取纯数字样本编号
                if not char.isdigit():  # 碰到非数字即前缀结束
                    if i > 0:  # 确保有数字
                        try:
                            sample_id = int(file_name[:i])  # 转为整数样本编号
                        except ValueError:
                            pass  # 忽略异常，保持为 -1
                    break  # 结束提取
            if sample_id < 0 or sample_id >= sample_count:  # 若样本编号无效或越界
                continue  # 跳过该文件
            value = parse_output_file(file_path, TIME_INDEX, NODE_INDEX)  # 解析该文件取值
            results[var][sample_id] = value  # 写入对应样本位置
    for var in output_vars:  # 读取完后做缺失统计
        nan_count = np.isnan(results[var]).sum()  # 统计 NaN 数量
        if nan_count > 0:  # 如有缺失
            print(f"警告: {os.path.basename(material_path)}/{var} 有 {nan_count}/{sample_count} 个缺失值")  # 打印警告
    return results  # 返回该材料的全部输出结果

# === Sobol分析函数 ===
def analyze_sobol(param_names, param_values, output_values):  # 定义 Sobol 二阶分析封装
    """执行Sobol分析，返回结果字典"""  # 文档说明
    valid_idx = ~np.isnan(output_values)  # 过滤 NaN
    X = param_values[valid_idx]  # 有效样本的参数矩阵
    Y = output_values[valid_idx]  # 有效样本的输出向量
    if len(Y) < 100:  # 样本量保护
        print(f"警告: 只有 {len(Y)} 个有效样本")  # 打印提示
        return None  # 样本不足时跳过分析
    problem = {  # 构建 SALib 问题定义
        'num_vars': X.shape[1],  # 参数维度
        'names': param_names,  # 参数名称
        'bounds': [[min(X[:, i]), max(X[:, i])] for i in range(X.shape[1])]  # 各参数取值范围
    }  # 结束 problem 定义
    try:  # 调用 SALib 执行分析
        result = sobol.analyze(problem, Y, calc_second_order=True)  # 计算含二阶项
        return {  # 整理输出
            'S1': result['S1'],  # 一阶指数
            'ST': result['ST'],  # 总效应指数
            'S2': result['S2']   # 二阶指数矩阵
        }  # 返回字典
    except Exception as e:  # 捕获计算异常
        print(f"Sobol分析错误: {e}")  # 打印错误
        return None  # 返回空结果

# === 保存二阶指数到CSV文件（保持不变） ===
def save_second_order_results(sobol_results, materials, output_vars, param_names, save_dir):  # 定义保存函数
    """
    将Sobol二阶敏感性指数保存到CSV文件
    参数:
    - sobol_results: {材料: {输出变量: {S2: matrix}}}
    - materials: 材料列表
    - output_vars: 输出变量列表
    - param_names: 参数名称列表
    - save_dir: 保存目录
    """  # 文档说明
    if not os.path.exists(save_dir):  # 若保存目录不存在
        os.makedirs(save_dir)  # 创建目录
        print(f"已创建目录: {save_dir}")  # 打印提示
    for material in materials:  # 遍历材料
        for output_var in output_vars:  # 遍历输出量
            result = sobol_results[material].get(output_var, None)  # 取该组合的分析结果
            if result is None or 'S2' not in result:  # 无结果则跳过
                print(f"跳过{material}/{output_var}: 无S2分析结果")  # 打印提示
                continue  # 进入下一项
            S2_matrix = result['S2']  # 取二阶矩阵
            abs_S2_matrix = np.abs(np.where(np.isfinite(S2_matrix), S2_matrix, 0))  # 取绝对值并把非数置零
            df = pd.DataFrame(  # 组装为数据框
                abs_S2_matrix,  # 数据
                index=param_names,  # 行名
                columns=param_names  # 列名
            )  # 结束 DataFrame
            file_name = f"{material}_{output_var}_S2_matrix.csv"  # 输出文件名
            file_path = os.path.join(save_dir, file_name)  # 完整路径
            try:  # 写文件
                df.to_csv(file_path, encoding='utf-8')  # 保存 CSV
                print(f"Sobol二阶指数已保存: {file_path}")  # 成功提示
            except Exception as e:  # 写入异常
                print(f"保存{file_path}时出错: {e}")  # 打印错误

# === 可视化函数：一次性绘制三种材料（3×3），并改用红色热力图 ===
def visualize_sobol_heatmaps_all(param_names, results_dict, save_path=None):  # 定义绘制总览 3×3 热力图的函数
    """
    创建 3×3 图：三行对应 MATERIALS（三种材料），三列对应 OUTPUT_VARS（三个输出变量）。
    仅做两处语义改动：
    1) cmap 改为 'Reds'；2) 多材料 3×3 布局。
    保持：每格标注、坐标轴“中心对齐”（+0.5 偏移）与中文标签。
    """  # 文档说明
    rows, cols = len(MATERIALS), len(OUTPUT_VARS)  # 行列数
    row_titles = ROW_TITLES if len(ROW_TITLES) == rows else MATERIALS  # 行标题（显示用，长度不匹配则回退到材料名）
    col_titles = COL_TITLES if len(COL_TITLES) == cols else OUTPUT_VARS  # 列标题（显示用，长度不匹配则回退到输出量名）
    fig = plt.figure(figsize=(16, 4 * rows))  # 创建画布，高度为原来的3倍
    gs = GridSpec(rows, cols)  # 设置网格布局
    col_prefixes = ['(a)', '(b)', '(c)']  # 每列的注记前缀（保持与旧版一致）
    axes_grid = [[None for _ in range(cols)] for _ in range(rows)]  # 缓存子图对象以便后续放置行标题
    subplot_prefixes = [f"({chr(ord('a') + i)})" for i in range(rows * cols)]  # 子图内 (a)–(i) 标注（行优先）

    # === 预计算同一列统一热力范围（vmin=0，vmax=同一列最大热力值） ===  # 【改动2】同一列统一热力范围
    col_vmax = {}  # 存放每一列（每个输出变量）的最大热力值  # 【改动2】列最大值容器
    for output_var in OUTPUT_VARS:  # 遍历每一列输出变量  # 【改动2】逐列统计最大值
        vmax_value = 0.0  # 初始化该列最大值（最小从0开始）  # 【改动2】最大值初值
        for material in MATERIALS:  # 遍历三行材料以获取该列的全局最大值  # 【改动2】跨材料取最大值
            result_tmp = results_dict.get(material, {}).get(output_var, None)  # 取对应组合的Sobol结果  # 【改动2】取结果
            if result_tmp is None or 'S2' not in result_tmp:  # 若无数据则跳过  # 【改动2】空结果保护
                continue  # 跳过  # 【改动2】继续下一个材料
            S2_tmp = result_tmp['S2']  # 取二阶矩阵  # 【改动2】取S2
            abs_tmp = np.abs(np.where(np.isfinite(S2_tmp), S2_tmp, 0))  # 绝对值并屏蔽非数  # 【改动2】得到热力矩阵
            vmax_value = max(vmax_value, float(np.max(abs_tmp)))  # 更新该列最大值  # 【改动2】更新最大值
        col_vmax[output_var] = vmax_value  # 记录该列统一最大值  # 【改动2】写入字典

    for r, material in enumerate(MATERIALS):  # 遍历材料 -> 行
        for c, output_var in enumerate(OUTPUT_VARS):  # 遍历输出量 -> 列
            ax = plt.subplot(gs[r, c])  # 取得子图坐标轴
            axes_grid[r][c] = ax  # 保存当前子图对象引用
            if r == 0:  # 仅第一行显示列标题（对应示例图上侧列标题）
                ax.set_title(str(col_titles[c]), fontsize=16, pad=10)  # 设置列标题（显示用，不影响数据读取）
            subplot_idx = r * cols + c  # 计算子图全局序号（行优先）
            ax.text(  # 在子图内添加 (a)–(i) 标注（对应示例图子图内标签）
                0.02,  # x 方向靠左（轴坐标）
                1.05,  # y 方向靠上（轴坐标）
                subplot_prefixes[subplot_idx],  # (a)–(i)
                fontsize=14,  # 字号
                ha='left',  # 水平左对齐
                va='top',  # 垂直上对齐
                transform=ax.transAxes  # 使用轴坐标变换
            )  # 完成子图内标注

            result = results_dict.get(material, {}).get(output_var, None)  # 取对应的分析结果
            if result is None or 'S2' not in result:  # 若无数据
                ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', fontsize=12)  # 子图中给出提示
                ax.set_xlabel('Input parameters' if r == rows - 1 else '', fontsize=14, labelpad=14)  # 仍给出轴名
                ax.set_ylabel('Input parameters' if c == 0 else '', fontsize=14, labelpad=14)  # 仍给出轴名
                ax.set_title("", loc='left', fontsize=13)  # 左侧标题给出材料名和输出量
                ax.set_xticks([])  # 无数据则不显示刻度
                ax.set_yticks([])  # 无数据则不显示刻度
                continue  # 进入下一幅
            S2_matrix = result['S2']  # 取二阶矩阵
            abs_matrix = np.abs(np.where(np.isfinite(S2_matrix), S2_matrix, 0))  # 绝对值并屏蔽非数
            mask = np.tril(np.ones_like(abs_matrix, dtype=bool))  # 【改动1】遮盖下三角与对角线，仅展示上三角并留空另一半
            sns.heatmap(  # 绘制热力图
                abs_matrix,  # 数据矩阵
                ax=ax,  # 放在当前子图
                cmap="Reds",  # 【改动1】颜色表改为红色系
                annot=True,  # 在格子里标注数值
                fmt=".2f",  # 数值格式
                cbar_kws={'label': 'interaction strength'},  # 颜色条标题
                linewidths=0.5,  # 网格线宽
                square=True,  # 保持方格比例
                mask=mask,  # 【改动1】仅显示上三角（对角线与下三角留空）
                vmin=0,  # 【改动2】同一列统一最小热力值为0
                vmax=col_vmax.get(output_var, 0.0)  # 【改动2】同一列统一最大热力值为该列全局最大值
            )  # 完成热力图
            ax.set_xlabel('input parameters' if r == rows - 1 else '', fontsize=14)  # 设置横轴名称
            ax.set_ylabel('input parameters' if c == 0 else '', fontsize=14)  # 设置纵轴名称

            # 坐标刻度“中心对齐”：在整数索引基础上 +0.5，使刻度对齐到每个网格中心（保持旧版修正）
            n_params = len(param_names)  # 参数个数
            step = max(1, n_params // 5)  # 每隔多少个参数显示一个刻度
            idx_positions = np.arange(0, n_params, step, dtype=int)  # 需要显示标签的整型索引
            tick_positions = idx_positions + 0.5  # 关键：中心对齐（偏移 0.5）
            tick_labels = [f"{param_names[i]}" for i in idx_positions]  # 标签文本（索引+换行+名称）
            ax.set_xticks(tick_positions)  # 设横轴刻度位置（中心）
            ax.set_xticklabels(tick_labels)  # 设横轴刻度文本
            ax.set_yticks(tick_positions)  # 设纵轴刻度位置（中心）
            ax.set_yticklabels(tick_labels)  # 设纵轴刻度文本
            ax.plot([0, n_params], [0, n_params], color='lightgrey', linewidth=0.5, zorder=3)  # 【改动1】对角线置空并用浅灰细线分隔

            #ax.set_title(f"{material} - {output_var}", loc='center', fontsize=13)  # 在每幅图左上角给出材料名和输出量
            ax.text(  # 在 x 轴标签下方添加列注记（与旧版一致）
                0.5,  # x 方向居中（轴坐标）
                -0.2,  # 位于坐标轴下方
                "",  # 注记内容
                fontsize=14,  # 字号
                ha='center',  # 水平居中
                va='top',  # 垂直顶部对齐
                transform=ax.transAxes  # 使用轴坐标变换
            )  # 完成子图注记

    plt.subplots_adjust(left=0.13, right=0.98, bottom=0.05, top=0.96, wspace=0.14, hspace=0.18)  # 紧凑布局，避免元素遮挡
    for r in range(rows):  # 遍历每一行以放置左侧行标题
        if axes_grid[r][0] is None:  # 保护：若子图不存在则跳过
            continue  # 跳过该行
        pos = axes_grid[r][0].get_position()  # 获取该行第一列子图在画布中的位置
        y_center = (pos.y0 + pos.y1) / 2.0  # 计算该行的垂直中心位置
        fig.text(  # 在左侧添加行标题（对应示例图左侧行标题）
            0.05,  # x 位置（画布坐标，预留左边距）
            y_center,  # y 位置（该行中心）
            str(row_titles[r]),  # 行标题文本（显示用，不影响数据读取）
            fontsize=16,  # 字号
            ha='center',  # 对齐
            va='center'  # 垂直居中
        )  # 完成行标题绘制

    if save_path:  # 如指定保存路径
        plt.savefig(save_path, dpi=600)  # 高分辨率保存图片
        print(f"图表已保存: {save_path}")  # 提示保存成功
        plt.show()  # 直接显示
    else:  # 未指定保存路径
        plt.show()  # 直接显示

# === 主流程（数据读取/分析/保存保持不变，仅将绘图改为一次性输出3×3） ===
def main():  # 定义主入口
    print("=== Sobol敏感性分析开始 ===")  # 起始提示
    print("\n[步骤1/4] 读取样本文件...")  # 步骤提示
    param_names, param_data = read_sample_file(SAMPLE_PATH)  # 读取样本
    n_samples, n_params = param_data.shape  # 获取样本规模
    print(f"样本统计: {n_samples}个样本, {n_params}个参数")  # 打印规模
    print(f"参数名称: {', '.join(param_names)}")  # 打印参数名

    print("\n[步骤2/4] 读取结果文件...")  # 步骤提示
    all_results = {mat: {} for mat in MATERIALS}  # 初始化存放所有材料结果的字典
    for material in MATERIALS:  # 遍历材料
        print(f"处理材料: {material}...")  # 提示当前材料
        mat_path = os.path.join(RESULT_BASE, material)  # 材料目录路径
        if not os.path.exists(mat_path):  # 若不存在
            print(f"警告: 路径不存在 - {mat_path}")  # 打印警告
            continue  # 跳过
        mat_results = read_material_results(mat_path, OUTPUT_VARS, n_samples)  # 读取该材料所有输出
        all_results[material] = mat_results  # 写入汇总

    print("\n[步骤3/4] 执行Sobol分析...")  # 步骤提示
    sobol_results = {mat: {} for mat in MATERIALS}  # 初始化分析结果字典
    for material in MATERIALS:  # 遍历材料
        for output_var in OUTPUT_VARS:  # 遍历输出变量
            Y = all_results[material].get(output_var, np.full(n_samples, np.nan))  # 取该输出的样本向量
            valid_count = np.sum(~np.isnan(Y))  # 统计有效样本数
            if valid_count < 100:  # 阈值保护
                print(f"警告: {material}/{output_var} 只有 {valid_count} 个有效样本，跳过分析")  # 打印提示
                sobol_results[material][output_var] = None  # 标记为空
                continue  # 跳过
            print(f"分析 {material}/{output_var}...")  # 过程提示
            sobol_result = analyze_sobol(param_names, param_data, Y)  # 执行 Sobol 分析
            sobol_results[material][output_var] = sobol_result  # 写入结果

    print("\n[步骤4/5] 保存二阶指数结果到CSV文件...")  # 步骤提示
    save_dir = "C:/Users/11201/Desktop/sensities/sobol/S2_matrices/"  # CSV 输出目录
    save_second_order_results(  # 调用保存函数
        sobol_results,  # 全部分析结果
        MATERIALS,  # 材料列表
        OUTPUT_VARS,  # 输出变量列表
        param_names,  # 参数名
        save_dir  # 保存目录
    )  # 完成 CSV 写出

    print("\n[步骤5/5] 创建三种材料的二阶交互作用 3×3 热力图...")  # 步骤提示
    visualize_sobol_heatmaps_all(  # 调用一次性 3×3 绘图
        param_names,  # 参数名
        sobol_results,  # 分析结果
        save_path="C:/Users/11201/Desktop/sensities/sobol/AllMaterials_second_order_heatmaps.png"  # 输出图片路径
    )  # 完成绘图与保存
    print("\n=== 分析完成 ===")  # 结束提示

if __name__ == "__main__":  # 主模块入口
    main()  # 调用主函数
