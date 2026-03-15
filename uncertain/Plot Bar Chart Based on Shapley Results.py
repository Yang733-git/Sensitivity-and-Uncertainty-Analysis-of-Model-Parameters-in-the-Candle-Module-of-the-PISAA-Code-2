# -*- coding: utf-8 -*-
"""
根据 Shapley 结果文件，针对三类组件 UO2 / CL / CR
和三种输出 mLeav、mFrozen、mAccu 绘制条形图。

本版本：
- 读取 CSV 文件 shapley_results.csv
- 直接使用原始 Shapley 值（保留正负号，不做归一化）
- 在一张图中画出 3x3 个子图：
    行：UO2 / CL / CR（同一行是同种组件）
    列：mLeav / mFrozen / mAccu（同一列是同种输出）
- 每个子图是竖直条形图：
    横轴：输入变量（param_name），按原始顺序（param_index），带变量名
    纵轴：Shapley value（可正可负）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
#plt.rcParams['font.family'] = 'SimSun'
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.serif'] = 'Times New Roman'  # 指定衬线字体为Times New Roman

def bootstrap_mean_error(values, n_boot=1000, ci=95.0, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, 0.0, 0.0

    center = float(values.mean())

    if values.size == 1:
        return center, 0.0, 0.0

    if rng is None:
        rng = np.random.default_rng(0)

    boot_means = np.empty(n_boot, dtype=float)
    n = values.size

    for k in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[k] = sample.mean()

    alpha = (100.0 - ci) / 2.0
    low, high = np.percentile(boot_means, [alpha, 100.0 - alpha])

    return center, max(center - low, 0.0), max(high - center, 0.0)

def main():
    # ===== 1. 读取 Shapley 结果 =====
    # 注意路径要用 raw string 或者双反斜杠，避免转义问题
    shapley_path = r"C:\Users\11201\Desktop\sensities\Candle_To_Analysize\shapley_results.csv"

    # 读取 CSV 文件
    df = pd.read_csv(shapley_path)

    # 简单检查一下列名（可选）
    expected_cols = {"material", "output", "param_index", "param_name", "shapley_value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"下面这些列在 CSV 里没找到，请检查列名或修改代码: {missing}")

    # ===== 2. 使用原始带符号的 Shapley 值（不做任何归一化） =====
    # 直接把原始 Shapley 值作为绘图用列
    df["shap_to_plot"] = df["shapley_value"]

    # 为了画图顺序统一，设置组件和输出的顺序
    materials_order = ["UO2", "CL", "CR"]
    outputs_order = ["mLeav", "mFrozen", "mAccu"]  # 如需调整顺序可在这里改

    #materials_titles = ["Fuel pellet", "Fuel cladding", "Control rod"]
    materials_titles = ["Fuel pellet", "Fuel cladding", "Control rod"]
    outputs_titles = ["FOM1", "FOM2", "FOM3"]

    bootstrap_n = 1000
    bootstrap_ci = 95.0
    rng = np.random.default_rng(20240314)

    # ===== 3. 准备输出图像保存目录 =====
    output_dir = os.path.join(os.path.dirname(shapley_path), "fig_shapley_bar")
    os.makedirs(output_dir, exist_ok=True)

    # ===== 4. 创建 3x3 子图：行=组件，列=输出 =====
    n_rows = len(materials_order)
    n_cols = len(outputs_order)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 2.4 * n_rows)  # 尺寸可根据需要微调
        # 不再 sharey=True，每个子图独立 y 轴
    )

    # 统一处理 axes 为二维数组
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, mat in enumerate(materials_order):
        for j, out in enumerate(outputs_order):
            ax = axes[i, j]

            sub_label = f"({chr(ord('a') + i * n_cols + j)})"
            ax.text(0.02, 0.98, sub_label, transform=ax.transAxes, va="top", ha="left")

            # 取出当前 (material, output) 下的数据
            sub = df[(df["output"] == out) & (df["material"] == mat)].copy()

            if sub.empty:
                # 如果某个组件/输出组合没有数据，则跳过
                ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center")
                ax.axis("off")
                continue

            # 按“原始顺序”排序，而不是按 Shapley 大小排序
            plot_rows = []
            for (param_index, param_name), group in sub.groupby(["param_index", "param_name"]):
                center, err_low, err_high = bootstrap_mean_error(
                    group["shap_to_plot"].values,
                    n_boot=bootstrap_n,
                    ci=bootstrap_ci,
                    rng=rng
                )
                plot_rows.append(
                    {
                        "param_index": param_index,
                        "param_name": param_name,
                        "shap_to_plot": center,
                        "err_low": err_low,
                        "err_high": err_high
                    }
                )
            sub = pd.DataFrame(plot_rows)
            sub = sub.sort_values("param_index", ascending=True)

            x_pos = np.arange(len(sub))
            yerr = np.vstack([sub["err_low"].values, sub["err_high"].values])

            # 竖直条形图：横轴是输入变量，纵轴是原始 Shapley 值（可正可负）
            ax.bar(x_pos, sub["shap_to_plot"].values, yerr=yerr, capsize=3)

            # 设置横轴为参数名，并旋转避免重叠
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['TH' if x == 'SIG' else x for x in sub["param_name"]], rotation=55, ha="center")
            if i == n_rows - 1:
                ax.set_xlabel("Input parameter", fontsize=12)
            else:
                #ax.set_xticklabels([])
                ax.set_xlabel("")

            # 每个子图都设置 y 轴标签
            if j == 0:
                ax.set_ylabel("Shapley index")
            else:
                ax.set_ylabel("")

            # 每个子图标题里写 组件 + 输出
            # 不再手动设置 ylim，让 Matplotlib 自己根据数据自适应纵坐标范围
            # （保留空行占位，不改其它结构）

    for j, title in enumerate(outputs_titles):
        axes[0, j].set_title(title)

    # 整体大标题
    #fig.suptitle("Shapley values (3 materials × 3 outputs)", fontsize=14)

    # 调整布局，留出标题空间
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.09, top=0.96, wspace=0.20, hspace=0.24)  # 调整子图间距

    for i, title in enumerate(materials_titles):
        bbox = axes[i, 0].get_position()
        y_center = bbox.y0 + bbox.height / 2.0
        fig.text(0.07, y_center, title, va="center", ha="center", fontsize=14)

    # 保存单一图像（包含 3x3 子图）
    out_file = os.path.join(output_dir, "shapley_bar_3x3.png")
    plt.savefig(out_file, dpi=600)
    plt.close(fig)

    print(f"Saved figure: {out_file}")


if __name__ == "__main__":
    main()