import os  # 导入操作系统相关库
import re  # 导入正则表达式库
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据处理库
import matplotlib.pyplot as plt  # 导入绘图库
import scipy.stats as stats  # 导入统计库
from matplotlib.ticker import AutoMinorLocator  # 导入副刻度定位器
from scipy.stats import gaussian_kde  # 导入核密度估计
from itertools import combinations  # 导入组合工具

"""
功能：蒙特卡洛不确定性分析及概率密度分布图绘制
使用方式：
1. 运行脚本后，输入要分析的输入变量名称（如'mLiq'）
2. 脚本会读取指定路径下的数据文件
3. 计算统计指标并绘制三行三列9个子图
4. 结果保存在指定路径的PNG文件中

布局说明：
- 行：表示不同组件类型（燃料芯块、包壳、控制棒）
- 列：表示不同输出变量（迁移量、凝固量、积累量）
- 每个子图显示单一组件类型和单一输出变量的概率密度分布
"""

# 设置全局字体为宋体（SimSun）并确保负号正确显示
plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
#plt.rcParams['font.family'] = 'SimSun'
#plt.rcParams['axes.unicode_minus'] = False
#plt.rcParams['font.serif'] = 'Times New Roman'  # 指定衬线字体为Times New Roman


def _safe_js_divergence(p, q):
    """
    计算离散概率分布之间的 Jensen-Shannon divergence
    说明：
    1. 输入 p、q 可以是频数或概率
    2. 函数内部会自动归一化
    3. 返回值使用自然对数底
    """
    p = np.asarray(p, dtype=float)  # 转为浮点数组
    q = np.asarray(q, dtype=float)  # 转为浮点数组
    p_sum = np.sum(p)  # 计算 p 总和
    q_sum = np.sum(q)  # 计算 q 总和
    if p_sum <= 0 or q_sum <= 0:  # 若总和非法则返回 NaN
        return np.nan
    p = p / p_sum  # 归一化 p
    q = q / q_sum  # 归一化 q
    m = 0.5 * (p + q)  # 构造中间分布 m

    def _kl_divergence(a, b):
        mask = (a > 0) & (b > 0)  # 仅在有效位置计算
        if not np.any(mask):  # 若无有效位置则返回 0
            return 0.0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))  # 计算 KL 散度

    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)  # 返回 JS 散度


def _normalize_to_unit_interval(values):
    """
    将一维样本归一化到 [0, 1]
    说明：
    1. 自动去除非有限值
    2. 若所有值相同，则统一映射为 0.5
    3. 返回归一化数组、最小值、最大值、极差
    """
    values = np.asarray(values, dtype=float)  # 转为浮点数组
    values = values[np.isfinite(values)]  # 保留有限值
    if values.size == 0:  # 无有效值时返回空结果
        return values, np.nan, np.nan, np.nan
    value_min = np.min(values)  # 计算最小值
    value_max = np.max(values)  # 计算最大值
    value_range = value_max - value_min  # 计算极差
    if np.isclose(value_range, 0.0):  # 若极差近似为 0
        return np.full(values.shape, 0.5, dtype=float), value_min, value_max, value_range
    return (values - value_min) / value_range, value_min, value_max, value_range  # 线性归一化


def _compute_1d_additional_statistics(values, uniform_bins=10):
    """
    计算一维样本的附加统计指标
    指标包括：
    - CV
    - variance ratio
    - KS statistic / pvalue
    - Wasserstein distance
    - Jensen-Shannon divergence
    - 分层覆盖均匀性指标
    - 边缘分布分箱频数偏差
    """
    extra_stats = {}  # 初始化结果字典
    normalized_values, value_min, value_max, value_range = _normalize_to_unit_interval(values)  # 归一化
    n = len(normalized_values)  # 样本数

    if n == 0:  # 无有效样本时全部置为 NaN
        extra_stats['cv'] = np.nan
        extra_stats['variance_ratio'] = np.nan
        extra_stats['ks_statistic'] = np.nan
        extra_stats['ks_pvalue'] = np.nan
        extra_stats['wasserstein_distance'] = np.nan
        extra_stats['jensen_shannon_divergence'] = np.nan
        extra_stats['strata_bin_count_cv'] = np.nan
        extra_stats['strata_bin_count_min'] = np.nan
        extra_stats['strata_bin_count_max'] = np.nan
        extra_stats['strata_empty_bin_ratio'] = np.nan
        extra_stats['marginal_bin_abs_dev_mean'] = np.nan
        extra_stats['marginal_bin_abs_dev_max'] = np.nan
        extra_stats['marginal_bin_rel_abs_dev_mean'] = np.nan
        extra_stats['marginal_bin_rel_abs_dev_max'] = np.nan
        return extra_stats

    original_mean = np.mean(values)  # 原始样本均值
    original_std = np.std(values)  # 原始样本标准差（与原脚本保持一致，ddof=0）
    original_var = np.var(values, ddof=1) if len(values) > 1 else 0.0  # 原始样本方差（样本方差）

    if not np.isclose(original_mean, 0.0):  # 均值不为 0 时计算 CV
        extra_stats['cv'] = original_std / original_mean
    else:
        extra_stats['cv'] = np.nan  # 均值为 0 时 CV 不定义

    if value_range > 0.0:  # 极差大于 0 时才计算均匀性比较指标
        uniform_variance_same_range = (value_range ** 2) / 12.0  # 同范围均匀分布方差
        extra_stats['variance_ratio'] = original_var / uniform_variance_same_range if uniform_variance_same_range > 0.0 else np.nan  # 方差比
        ks_result = stats.kstest(normalized_values, 'uniform', args=(0.0, 1.0))  # 对 U(0,1) 做 KS 检验
        extra_stats['ks_statistic'] = ks_result.statistic  # KS 统计量
        extra_stats['ks_pvalue'] = ks_result.pvalue  # KS 检验 p 值
        uniform_reference = (np.arange(n, dtype=float) + 0.5) / n  # 构造离散均匀参考样本
        extra_stats['wasserstein_distance'] = stats.wasserstein_distance(normalized_values, uniform_reference)  # Wasserstein 距离
        hist_counts, _ = np.histogram(normalized_values, bins=uniform_bins, range=(0.0, 1.0))  # 对 [0,1] 分箱
        expected_count = n / float(uniform_bins)  # 每箱期望频数
        uniform_probs = np.full(uniform_bins, 1.0 / uniform_bins, dtype=float)  # 理想均匀分布概率
        observed_probs = hist_counts / float(n)  # 实际分箱概率
        extra_stats['jensen_shannon_divergence'] = _safe_js_divergence(observed_probs, uniform_probs)  # JS 散度

        if uniform_bins > 1 and expected_count > 0.0:  # 计算分层覆盖频数 CV
            extra_stats['strata_bin_count_cv'] = np.std(hist_counts, ddof=1) / expected_count
        else:
            extra_stats['strata_bin_count_cv'] = 0.0

        extra_stats['strata_bin_count_min'] = np.min(hist_counts)  # 分层最小频数
        extra_stats['strata_bin_count_max'] = np.max(hist_counts)  # 分层最大频数
        extra_stats['strata_empty_bin_ratio'] = np.mean(hist_counts == 0)  # 空箱占比

        abs_deviation = np.abs(hist_counts - expected_count)  # 与期望频数的绝对偏差
        extra_stats['marginal_bin_abs_dev_mean'] = np.mean(abs_deviation)  # 平均绝对偏差
        extra_stats['marginal_bin_abs_dev_max'] = np.max(abs_deviation)  # 最大绝对偏差
        extra_stats['marginal_bin_rel_abs_dev_mean'] = np.mean(abs_deviation / expected_count) if expected_count > 0.0 else np.nan  # 平均相对绝对偏差
        extra_stats['marginal_bin_rel_abs_dev_max'] = np.max(abs_deviation / expected_count) if expected_count > 0.0 else np.nan  # 最大相对绝对偏差
    else:  # 退化样本时相关指标不定义
        extra_stats['variance_ratio'] = np.nan
        extra_stats['ks_statistic'] = np.nan
        extra_stats['ks_pvalue'] = np.nan
        extra_stats['wasserstein_distance'] = np.nan
        extra_stats['jensen_shannon_divergence'] = np.nan
        extra_stats['strata_bin_count_cv'] = np.nan
        extra_stats['strata_bin_count_min'] = np.nan
        extra_stats['strata_bin_count_max'] = np.nan
        extra_stats['strata_empty_bin_ratio'] = np.nan
        extra_stats['marginal_bin_abs_dev_mean'] = np.nan
        extra_stats['marginal_bin_abs_dev_max'] = np.nan
        extra_stats['marginal_bin_rel_abs_dev_mean'] = np.nan
        extra_stats['marginal_bin_rel_abs_dev_max'] = np.nan

    return extra_stats  # 返回附加指标


def _compute_quantile_confidence_intervals(values, confidence_level=0.95, n_bootstrap=2000, random_seed=0):
    """
    计算 p5、p95 及 p95-p5 的 95% 置信区间
    说明：
    1. 采用非参数 Bootstrap 百分位法
    2. 返回样本量以及各统计量对应的区间上下限
    3. 固定随机种子以保证结果可复现
    """
    values = np.asarray(values, dtype=float)  # 转为浮点数组
    values = values[np.isfinite(values)]  # 保留有限值
    result = {
        'sample_count': len(values),
        'p5_ci_low': np.nan,
        'p5_ci_high': np.nan,
        'p95_ci_low': np.nan,
        'p95_ci_high': np.nan,
        'p95_minus_p5': np.nan,
        'p95_minus_p5_ci_low': np.nan,
        'p95_minus_p5_ci_high': np.nan
    }  # 初始化结果字典

    if len(values) == 0:  # 无有效样本时直接返回
        return result

    result['p95_minus_p5'] = np.percentile(values, 95) - np.percentile(values, 5)  # 计算 p95-p5 的点估计

    if len(values) == 1:  # 单一样本时区间退化为点值
        result['p5_ci_low'] = values[0]
        result['p5_ci_high'] = values[0]
        result['p95_ci_low'] = values[0]
        result['p95_ci_high'] = values[0]
        result['p95_minus_p5_ci_low'] = 0.0
        result['p95_minus_p5_ci_high'] = 0.0
        return result

    rng = np.random.default_rng(random_seed)  # 固定随机种子
    alpha = 1.0 - confidence_level  # 显著性水平
    lower_q = 100.0 * (alpha / 2.0)  # 下侧分位点
    upper_q = 100.0 * (1.0 - alpha / 2.0)  # 上侧分位点
    bootstrap_p5_list = []  # 存储 Bootstrap 的 p5
    bootstrap_p95_list = []  # 存储 Bootstrap 的 p95
    bootstrap_spread_list = []  # 存储 Bootstrap 的 p95-p5
    chunk_size = min(500, n_bootstrap)  # 分块计算，避免一次性占用过多内存
    remaining = n_bootstrap  # 剩余 Bootstrap 次数

    while remaining > 0:  # 分块执行 Bootstrap
        current_chunk = min(chunk_size, remaining)  # 当前分块大小
        bootstrap_indices = rng.integers(0, len(values), size=(current_chunk, len(values)))  # 生成 Bootstrap 抽样下标
        bootstrap_samples = values[bootstrap_indices]  # 构造 Bootstrap 样本
        current_p5 = np.percentile(bootstrap_samples, 5, axis=1)  # 计算当前分块的 p5
        current_p95 = np.percentile(bootstrap_samples, 95, axis=1)  # 计算当前分块的 p95
        bootstrap_p5_list.append(current_p5)  # 保存当前分块的 p5
        bootstrap_p95_list.append(current_p95)  # 保存当前分块的 p95
        bootstrap_spread_list.append(current_p95 - current_p5)  # 保存当前分块的 p95-p5
        remaining -= current_chunk  # 更新剩余次数

    bootstrap_p5 = np.concatenate(bootstrap_p5_list)  # 拼接所有 Bootstrap p5 结果
    bootstrap_p95 = np.concatenate(bootstrap_p95_list)  # 拼接所有 Bootstrap p95 结果
    bootstrap_spread = np.concatenate(bootstrap_spread_list)  # 拼接所有 Bootstrap p95-p5 结果
    result['p5_ci_low'] = np.percentile(bootstrap_p5, lower_q)  # p5 的置信区间下限
    result['p5_ci_high'] = np.percentile(bootstrap_p5, upper_q)  # p5 的置信区间上限
    result['p95_ci_low'] = np.percentile(bootstrap_p95, lower_q)  # p95 的置信区间下限
    result['p95_ci_high'] = np.percentile(bootstrap_p95, upper_q)  # p95 的置信区间上限
    result['p95_minus_p5_ci_low'] = np.percentile(bootstrap_spread, lower_q)  # p95-p5 的置信区间下限
    result['p95_minus_p5_ci_high'] = np.percentile(bootstrap_spread, upper_q)  # p95-p5 的置信区间上限
    return result  # 返回分位数区间统计结果


def _compute_histogram_bootstrap_band(values, bin_edges, confidence_level=0.95, n_bootstrap=2000, random_seed=0):
    """
    计算直方图概率密度估计的 Bootstrap 不确定度带
    说明：
    1. 使用与原始直方图完全相同的分箱边界
    2. 对样本做非参数 Bootstrap 重采样
    3. 返回每个 bin 密度的置信区间上下限
    """
    values = np.asarray(values, dtype=float)  # 转为浮点数组
    values = values[np.isfinite(values)]  # 保留有限值
    bin_edges = np.asarray(bin_edges, dtype=float)  # 转为浮点数组
    result = {
        'density_ci_low': np.full(len(bin_edges) - 1, np.nan, dtype=float),
        'density_ci_high': np.full(len(bin_edges) - 1, np.nan, dtype=float)
    }  # 初始化结果字典

    if len(values) == 0 or len(bin_edges) < 2:  # 无有效样本或分箱非法时直接返回
        return result

    if len(values) == 1:  # 单一样本时区间退化为点值
        single_hist, _ = np.histogram(values, bins=bin_edges, density=True)  # 计算单一样本直方图密度
        result['density_ci_low'] = single_hist  # 下限即点估计
        result['density_ci_high'] = single_hist  # 上限即点估计
        return result

    rng = np.random.default_rng(random_seed)  # 固定随机种子
    alpha = 1.0 - confidence_level  # 显著性水平
    lower_q = 100.0 * (alpha / 2.0)  # 下侧分位点
    upper_q = 100.0 * (1.0 - alpha / 2.0)  # 上侧分位点
    bootstrap_hist_list = []  # 存储 Bootstrap 直方图密度
    chunk_size = min(200, n_bootstrap)  # 分块大小，避免一次性占用过多内存
    remaining = n_bootstrap  # 剩余 Bootstrap 次数

    while remaining > 0:  # 分块执行 Bootstrap
        current_chunk = min(chunk_size, remaining)  # 当前分块大小
        bootstrap_indices = rng.integers(0, len(values), size=(current_chunk, len(values)))  # 生成 Bootstrap 抽样下标
        bootstrap_samples = values[bootstrap_indices]  # 构造 Bootstrap 样本
        current_hist_list = []  # 存储当前分块的直方图密度
        for bootstrap_sample in bootstrap_samples:  # 遍历当前分块中的每个 Bootstrap 样本
            current_hist, _ = np.histogram(bootstrap_sample, bins=bin_edges, density=True)  # 使用固定分箱边界计算密度直方图
            current_hist_list.append(current_hist)  # 保存当前 Bootstrap 密度
        bootstrap_hist_list.append(np.asarray(current_hist_list, dtype=float))  # 保存当前分块结果
        remaining -= current_chunk  # 更新剩余次数

    bootstrap_hist_matrix = np.vstack(bootstrap_hist_list)  # 拼接所有 Bootstrap 直方图密度结果
    result['density_ci_low'] = np.percentile(bootstrap_hist_matrix, lower_q, axis=0)  # 每个 bin 密度下限
    result['density_ci_high'] = np.percentile(bootstrap_hist_matrix, upper_q, axis=0)  # 每个 bin 密度上限
    return result  # 返回直方图 Bootstrap 不确定度带


def _normalize_matrix_to_unit_cube(sample_matrix):
    """
    将多维样本矩阵逐维归一化到 [0,1]^d
    说明：
    1. 每一列单独归一化
    2. 常量列统一映射为 0.5
    """
    if sample_matrix.size == 0:  # 空矩阵直接返回
        return sample_matrix, np.array([]), np.array([]), np.array([])
    min_values = np.min(sample_matrix, axis=0)  # 每列最小值
    max_values = np.max(sample_matrix, axis=0)  # 每列最大值
    ranges = max_values - min_values  # 每列极差
    normalized_matrix = np.empty_like(sample_matrix, dtype=float)  # 初始化归一化矩阵
    constant_mask = np.isclose(ranges, 0.0)  # 判断常量列
    if np.any(~constant_mask):  # 对非常量列进行线性归一化
        normalized_matrix[:, ~constant_mask] = (sample_matrix[:, ~constant_mask] - min_values[~constant_mask]) / ranges[~constant_mask]
    if np.any(constant_mask):  # 对常量列置为 0.5
        normalized_matrix[:, constant_mask] = 0.5
    return normalized_matrix, min_values, max_values, ranges  # 返回结果


def _approximate_star_discrepancy(sample_matrix_unit):
    """
    近似计算星差异（star discrepancy）
    说明：
    1. 这里采用以样本点为锚框的近似方法
    2. 返回的是近似值，不是高维精确解
    """
    if sample_matrix_unit.size == 0:  # 空矩阵返回 NaN
        return np.nan
    max_discrepancy = 0.0  # 初始化最大差异
    for point in sample_matrix_unit:  # 遍历每个样本点作为锚点
        box_volume = np.prod(point)  # 以该点为右上角的超矩形体积
        empirical_cdf = np.mean(np.all(sample_matrix_unit <= point, axis=1))  # 经验分布函数值
        current_discrepancy = abs(empirical_cdf - box_volume)  # 当前差异
        if current_discrepancy > max_discrepancy:  # 更新最大差异
            max_discrepancy = current_discrepancy
    return max_discrepancy  # 返回近似星差异


def _centered_l2_discrepancy(sample_matrix_unit):
    """
    计算 centered L2 discrepancy
    说明：
    1. 使用标准 closed-form 公式
    2. 对样本数较大时计算量会增加，但不改变原流程
    """
    if sample_matrix_unit.size == 0:  # 空矩阵返回 NaN
        return np.nan
    n_samples, n_dims = sample_matrix_unit.shape  # 获取样本数和维度数
    term1 = (13.0 / 12.0) ** n_dims  # 第一项常数项
    centered_distance = np.abs(sample_matrix_unit - 0.5)  # 与中心点 0.5 的距离
    term2 = (2.0 / n_samples) * np.sum(
        np.prod(1.0 + 0.5 * centered_distance - 0.5 * centered_distance ** 2, axis=1)
    )  # 第二项
    term3_sum = 0.0  # 初始化第三项累加和
    for i in range(n_samples):  # 双重求和的外层循环
        current_term = (
            1.0
            + 0.5 * np.abs(sample_matrix_unit[i] - 0.5)[None, :]
            + 0.5 * np.abs(sample_matrix_unit - 0.5)
            - 0.5 * np.abs(sample_matrix_unit[i][None, :] - sample_matrix_unit)
        )  # 当前 i 对所有样本的乘积项
        term3_sum += np.sum(np.prod(current_term, axis=1))  # 累加第三项
    term3 = term3_sum / float(n_samples ** 2)  # 第三项归一化
    discrepancy_squared = term1 - term2 + term3  # 计算平方值
    return np.sqrt(max(discrepancy_squared, 0.0))  # 返回 centered L2 discrepancy


def _minimum_spacing(sample_matrix_unit):
    """
    计算样本点最小间距
    说明：
    1. 使用欧氏距离
    2. 返回所有点对之间的最小距离
    """
    if sample_matrix_unit.shape[0] < 2:  # 少于两个点无法计算
        return np.nan
    min_distance = np.inf  # 初始化最小距离
    for i in range(sample_matrix_unit.shape[0] - 1):  # 遍历点对
        distances = np.sqrt(np.sum((sample_matrix_unit[i + 1:] - sample_matrix_unit[i]) ** 2, axis=1))  # 计算当前点到后续点的距离
        if distances.size > 0:  # 若当前有有效距离
            local_min = np.min(distances)  # 当前局部最小距离
            if local_min < min_distance:  # 更新全局最小距离
                min_distance = local_min
    return min_distance if np.isfinite(min_distance) else np.nan  # 返回最小间距


def _approximate_fill_distance(sample_matrix_unit, n_reference_points=2000, random_seed=0):
    """
    近似计算 fill distance（填充性）
    说明：
    1. 在单位超立方体中随机撒参考点
    2. 对每个参考点求其到最近样本点的距离
    3. 取这些最近距离中的最大值作为近似 fill distance
    """
    if sample_matrix_unit.size == 0:  # 空矩阵返回 NaN
        return np.nan
    rng = np.random.default_rng(random_seed)  # 固定随机种子以保证复现
    n_dims = sample_matrix_unit.shape[1]  # 获取维度数
    reference_points = rng.random((n_reference_points, n_dims))  # 生成参考点
    max_nearest_distance = 0.0  # 初始化最大最近距离
    chunk_size = 200  # 分块大小，避免一次性占用过多内存
    for start_idx in range(0, n_reference_points, chunk_size):  # 分块遍历参考点
        end_idx = min(start_idx + chunk_size, n_reference_points)  # 当前分块结束下标
        chunk = reference_points[start_idx:end_idx]  # 取当前分块
        diff = chunk[:, None, :] - sample_matrix_unit[None, :, :]  # 构造距离差
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # 计算欧氏距离
        nearest_distance = np.min(distances, axis=1)  # 每个参考点到最近样本点距离
        chunk_max = np.max(nearest_distance)  # 当前分块的最大最近距离
        if chunk_max > max_nearest_distance:  # 更新全局最大最近距离
            max_nearest_distance = chunk_max
    return max_nearest_distance  # 返回近似 fill distance


def _compute_2d_projection_uniformity(sample_matrix_unit, projection_bins=8):
    """
    计算二维投影均匀性指标
    指标包括：
    - 所有二维投影的相对 RMSE 平均值
    - 所有二维投影的相对 RMSE 最大值
    - 所有二维投影的空箱比例平均值
    - 所有二维投影的 JS divergence 平均值
    """
    result = {
        'pair_count': 0.0,
        'proj2d_rel_rmse_mean': np.nan,
        'proj2d_rel_rmse_max': np.nan,
        'proj2d_empty_cell_ratio_mean': np.nan,
        'proj2d_js_divergence_mean': np.nan
    }  # 初始化结果字典
    n_samples, n_dims = sample_matrix_unit.shape  # 获取样本数和维度数
    if n_dims < 2 or n_samples == 0:  # 维度不足或无样本时直接返回
        return result

    rel_rmse_list = []  # 存储相对 RMSE
    empty_ratio_list = []  # 存储空箱比例
    js_list = []  # 存储 JS 散度

    for dim_i, dim_j in combinations(range(n_dims), 2):  # 遍历所有二维投影组合
        hist2d, _, _ = np.histogram2d(
            sample_matrix_unit[:, dim_i],
            sample_matrix_unit[:, dim_j],
            bins=projection_bins,
            range=((0.0, 1.0), (0.0, 1.0))
        )  # 二维投影分箱
        expected_count = n_samples / float(projection_bins ** 2)  # 每个二维网格的期望频数
        if expected_count <= 0.0:  # 防御性检查
            continue
        rel_rmse = np.sqrt(np.mean(((hist2d - expected_count) / expected_count) ** 2))  # 相对 RMSE
        empty_ratio = np.mean(hist2d == 0)  # 空网格比例
        observed_probs = hist2d.ravel() / float(n_samples)  # 观测概率
        uniform_probs = np.full(hist2d.size, 1.0 / hist2d.size, dtype=float)  # 理想均匀概率
        js_value = _safe_js_divergence(observed_probs, uniform_probs)  # JS 散度
        rel_rmse_list.append(rel_rmse)  # 保存相对 RMSE
        empty_ratio_list.append(empty_ratio)  # 保存空箱比例
        js_list.append(js_value)  # 保存 JS 散度

    if rel_rmse_list:  # 若存在有效二维投影结果
        result['pair_count'] = float(len(rel_rmse_list))  # 投影对数
        result['proj2d_rel_rmse_mean'] = float(np.mean(rel_rmse_list))  # 平均相对 RMSE
        result['proj2d_rel_rmse_max'] = float(np.max(rel_rmse_list))  # 最大相对 RMSE
        result['proj2d_empty_cell_ratio_mean'] = float(np.mean(empty_ratio_list))  # 平均空箱比例
        result['proj2d_js_divergence_mean'] = float(np.mean(js_list))  # 平均 JS 散度

    return result  # 返回二维投影均匀性结果


def _build_joint_sample_statistics(sample_value_maps, materials, output_vars):
    """
    构造联合样本并计算多维均匀性指标
    说明：
    1. 按共同样本序号对齐 3×3 共 9 个维度
    2. 共同样本不足时，返回 NaN 而不影响原流程
    """
    dimension_keys = []  # 存储联合样本的维度键
    common_indices = None  # 存储所有维度共同拥有的样本序号

    for material in materials:  # 遍历材料
        for output_var in output_vars:  # 遍历输出变量
            current_value_map = sample_value_maps[material][output_var]  # 当前维度的样本映射
            current_indices = set(current_value_map.keys())  # 当前维度的样本序号集合
            dimension_keys.append((material, output_var))  # 记录维度信息
            if common_indices is None:  # 初始化共同样本集合
                common_indices = current_indices
            else:
                common_indices &= current_indices  # 取交集

    joint_stats = {
        'joint_common_sample_count': 0.0,
        'joint_dimension_count': float(len(dimension_keys)),
        'joint_min_spacing': np.nan,
        'joint_fill_distance': np.nan,
        'joint_star_discrepancy': np.nan,
        'joint_centered_l2_discrepancy': np.nan,
        'joint_marginal_bin_rel_abs_dev_mean': np.nan,
        'joint_marginal_bin_rel_abs_dev_max': np.nan,
        'joint_strata_bin_count_cv_mean': np.nan,
        'joint_proj2d_rel_rmse_mean': np.nan,
        'joint_proj2d_rel_rmse_max': np.nan,
        'joint_proj2d_empty_cell_ratio_mean': np.nan,
        'joint_proj2d_js_divergence_mean': np.nan
    }  # 初始化联合样本统计结果

    if not common_indices:  # 若没有共同样本序号
        return joint_stats

    sorted_indices = sorted(common_indices)  # 对共同样本序号排序
    sample_matrix = np.array(
        [[sample_value_maps[material][output_var][sample_idx] for material, output_var in dimension_keys] for sample_idx in sorted_indices],
        dtype=float
    )  # 构造联合样本矩阵
    valid_rows = np.all(np.isfinite(sample_matrix), axis=1)  # 检查每一行是否全为有限值
    sample_matrix = sample_matrix[valid_rows]  # 仅保留有效行

    if sample_matrix.shape[0] == 0:  # 若没有有效联合样本
        return joint_stats

    joint_stats['joint_common_sample_count'] = float(sample_matrix.shape[0])  # 记录共同有效样本数

    sample_matrix_unit, _, _, _ = _normalize_matrix_to_unit_cube(sample_matrix)  # 归一化到单位超立方体
    joint_stats['joint_min_spacing'] = _minimum_spacing(sample_matrix_unit)  # 计算最小间距
    joint_stats['joint_fill_distance'] = _approximate_fill_distance(sample_matrix_unit, n_reference_points=2000, random_seed=0)  # 计算近似 fill distance
    joint_stats['joint_star_discrepancy'] = _approximate_star_discrepancy(sample_matrix_unit)  # 计算近似星差异
    joint_stats['joint_centered_l2_discrepancy'] = _centered_l2_discrepancy(sample_matrix_unit)  # 计算 centered L2 discrepancy

    marginal_rel_abs_dev_mean_list = []  # 存储各维平均相对绝对偏差
    marginal_rel_abs_dev_max_list = []  # 存储各维最大相对绝对偏差
    marginal_bin_cv_list = []  # 存储各维分箱频数 CV

    for dim_idx in range(sample_matrix_unit.shape[1]):  # 遍历每个维度
        dim_hist, _ = np.histogram(sample_matrix_unit[:, dim_idx], bins=10, range=(0.0, 1.0))  # 当前维度分箱
        expected_count = sample_matrix_unit.shape[0] / 10.0  # 当前维度每箱期望频数
        abs_deviation = np.abs(dim_hist - expected_count)  # 绝对偏差
        if expected_count > 0.0:  # 防御性检查
            marginal_rel_abs_dev_mean_list.append(np.mean(abs_deviation / expected_count))  # 保存平均相对偏差
            marginal_rel_abs_dev_max_list.append(np.max(abs_deviation / expected_count))  # 保存最大相对偏差
            if len(dim_hist) > 1:  # 计算分箱频数 CV
                marginal_bin_cv_list.append(np.std(dim_hist, ddof=1) / expected_count)
            else:
                marginal_bin_cv_list.append(0.0)

    if marginal_rel_abs_dev_mean_list:  # 若存在有效边缘统计
        joint_stats['joint_marginal_bin_rel_abs_dev_mean'] = float(np.mean(marginal_rel_abs_dev_mean_list))  # 各维平均相对绝对偏差的平均值
        joint_stats['joint_marginal_bin_rel_abs_dev_max'] = float(np.max(marginal_rel_abs_dev_max_list))  # 各维最大相对绝对偏差的最大值
        joint_stats['joint_strata_bin_count_cv_mean'] = float(np.mean(marginal_bin_cv_list))  # 各维分层覆盖 CV 的平均值

    projection_stats = _compute_2d_projection_uniformity(sample_matrix_unit, projection_bins=8)  # 计算二维投影均匀性
    joint_stats['joint_proj2d_rel_rmse_mean'] = projection_stats['proj2d_rel_rmse_mean']  # 平均二维投影相对 RMSE
    joint_stats['joint_proj2d_rel_rmse_max'] = projection_stats['proj2d_rel_rmse_max']  # 最大二维投影相对 RMSE
    joint_stats['joint_proj2d_empty_cell_ratio_mean'] = projection_stats['proj2d_empty_cell_ratio_mean']  # 平均二维投影空箱率
    joint_stats['joint_proj2d_js_divergence_mean'] = projection_stats['proj2d_js_divergence_mean']  # 平均二维投影 JS 散度

    return joint_stats  # 返回联合样本统计结果


def monte_carlo_uncertainty_analysis():
    """
    执行蒙特卡洛不确定性分析，绘制分布直方图，并计算详细的统计指标
    修改为：三行三列9个子图，每行对应一种组件类型，每列对应一种输出变量
    """
    
    # 用户输入要分析的变量名
    target_var = input("请输入要分析的输入变量名称 (例如 'mLiq'): ").strip()  # 获取用户输入变量
    print(f"开始分析变量: {target_var}")  # 打印正在分析的变量
    
    # 基础路径设置
    base_path = r'C:\Users\11201\Desktop\sensities\Candle_To_Analysize\uncertain'  # 数据文件的基础路径
    output_path = r'C:\Users\11201\Desktop\sensities\Candle_To_Analysize\uncertain'  # 输出文件路径
    
    # 定义分析所需的常量
    materials = ['UO2', 'CL', 'CR']  # 三种材料类型（原：'燃料芯块','包壳','控制棒'）
    output_vars = ['mLeav', 'mFrozen', 'mAccu']  # 三种输出变量
    material_colors = {
        'UO2': '#FF9999',  # 红色调（原键：'燃料芯块'）
        'CL': '#99FF99',   # 绿色调（原键：'包壳'）
        'CR': '#9999FF'    # 蓝色调（原键：'控制棒'）
    }
    num_bins = 80  # 直方图分箱数量
    
    # 初始化结果数据结构
    data_matrix = {mat: {ov: [] for ov in output_vars} for mat in materials}  # 初始化数据存储
    sample_counts = {mat: [] for mat in materials}  # 存储有效样本计数
    sample_value_maps = {mat: {ov: {} for ov in output_vars} for mat in materials}  # 按样本序号存储数值，用于联合样本统计
    
    # 定义原始材料名称到中文名称的映射（用于读取文件夹）
    material_folder_map = {
        'UO2': 'UO2',
        'CL': 'CL',
        'CR': 'CR'
    }

    # 遍历所有材料文件夹
    for material in materials:
        folder_name = material_folder_map[material]  # 获取原始文件夹名称
        for output_var in output_vars:
            dir_path = os.path.join(base_path, folder_name, output_var)  # 获取每个材料的文件夹路径
            
            if not os.path.exists(dir_path):  # 检查路径是否存在
                print(f"警告: 目录不存在: {dir_path}")
                continue
            
            file_list = os.listdir(dir_path)  # 获取文件夹内所有数据文件
            if not file_list:
                print(f"警告: 文件夹为空: {dir_path}")
                continue
            
            valid_files = []  # 用于存储有效文件
            for file_name in file_list:
                match = re.match(r'^(\d+)', file_name)  # 按样本序号排序文件
                if match:
                    try:
                        sample_idx = int(match.group(1))
                        valid_files.append((sample_idx, file_name))
                    except:
                        continue
            
            valid_files.sort(key=lambda x: x[0])  # 按样本索引排序
            sample_counts[material].append(len(valid_files))
            
            # 读取每个文件的数据
            for sample_idx, file_name in valid_files:
                file_path = os.path.join(dir_path, file_name)
                
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()  # 读取文件内容
                    
                    if len(lines) < 2:
                        print(f"警告: 文件内容不足: {file_path}")
                        continue
                    
                    node_idx = 1  # 第一行数据节点
                    time_idx = 2  # 第一个时间点
                    data_line = lines[node_idx].split()  # 获取数据行
                    
                    if len(data_line) <= time_idx:
                        print(f"警告: 文件格式异常: {file_path}")
                        continue
                    
                    try:
                        value = float(data_line[time_idx])  # 尝试将数据转换为浮动值
                    except ValueError:
                        print(f"警告: 无法解析数值: {file_path} -> {data_line[time_idx]}")
                        continue
                    
                    data_matrix[material][output_var].append(value)  # 存储数据值
                    sample_value_maps[material][output_var][sample_idx] = value  # 按样本序号存储数据值
                    
                except Exception as e:
                    print(f"错误: 处理文件出错 {file_path}: {str(e)}")
    
    statistics = {mat: {ov: {} for ov in output_vars} for mat in materials}  # 创建统计数据结构
    joint_space_statistics = _build_joint_sample_statistics(sample_value_maps, materials, output_vars)  # 计算联合样本空间统计指标
    
    # 修改点：创建三行三列的子图布局（9个子图）
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))  # 创建3x3子图布局
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.06, top=0.96, wspace=0.16, hspace=0.20)  # 调整子图间距
    
    # 定义输出变量的中文名称映射
    output_var_names = {
        'mLeav': 'Migration',     # 原：'迁移量'
        'mFrozen': 'Frozen', # 原：'凝固量'
        'mAccu': 'Accumulation'      # 原：'积累量'
    }

    # 定义行标题（仅用于显示，不影响数据读取）
    row_title_names = {
        'UO2': 'Fuel Pellet',
        'CL': 'Cladding',
        'CR': 'Control Rod'
    }

    # 定义列标题（仅用于显示，不影响数据读取）
    col_title_names = {
        'mLeav': 'FOM1',
        'mFrozen': 'FOM2',
        'mAccu': 'FOM3'
    }

    # 定义子图标题前缀的小写字母
    subplot_letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']

    # 设置列标题（放在上侧，仅在第一行显示）
    for col_idx, output_var in enumerate(output_vars):  # 遍历输出变量设置列标题
        axes[0, col_idx].set_title(col_title_names.get(output_var, output_var), fontsize=14)  # 列标题

    # 修改点：遍历所有材料（行）和所有输出变量（列）
    letter_idx = 0  # 子图字母索引
    for row_idx, material in enumerate(materials):  # 行索引对应材料类型
        for col_idx, output_var in enumerate(output_vars):  # 列索引对应输出变量
            ax = axes[row_idx, col_idx]  # 获取当前子图
            
            # 在子图左上角内侧添加编号标签 (a)–(i)
            ax.text(0.02, 1.08, subplot_letters[letter_idx], transform=ax.transAxes, fontsize=12, va='top', ha='left')  # 子图编号
            letter_idx += 1  # 递增字母索引
            
            # 检查当前组合是否有数据
            if not data_matrix[material][output_var]:  # 如果没有数据，跳过
                print(f"警告: {material}/{output_var} 无可用数据")
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')  # 显示无数据
                continue
                
            data_vals = np.array(data_matrix[material][output_var])
            clean_vals = data_vals[np.isfinite(data_vals)]  # 移除非数值和无效数据
            
            if len(clean_vals) < 10:  # 有效数据不足检查
                print(f"警告: {material}/{output_var} 有效数据不足({len(clean_vals)})")
                ax.text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center')  # 显示数据不足
                continue
            
            # 计算统计指标
            stats_dict = {}
            stats_dict['sample_count'] = len(clean_vals)  # 样本量
            stats_dict['min'] = np.min(clean_vals)  # 最小值
            stats_dict['max'] = np.max(clean_vals)  # 最大值
            stats_dict['mean'] = np.mean(clean_vals)  # 均值
            stats_dict['std'] = np.std(clean_vals)  # 标准差
            stats_dict['var'] = np.var(clean_vals, ddof=1)  # 方差（样本方差，ddof=1）
            stats_dict['skew'] = stats.skew(clean_vals)  # 偏度
            stats_dict['kurtosis'] = stats.kurtosis(clean_vals)  # 峰度
            stats_dict['p5'] = np.percentile(clean_vals, 5)  # 5%分位数
            stats_dict['p95'] = np.percentile(clean_vals, 95)  # 95%分位数
            quantile_ci_stats = _compute_quantile_confidence_intervals(clean_vals, confidence_level=0.95, n_bootstrap=2000, random_seed=0)  # 计算分位数及差值的95%置信区间
            stats_dict['p5_ci_low'] = quantile_ci_stats['p5_ci_low']  # p5 的95%置信区间下限
            stats_dict['p5_ci_high'] = quantile_ci_stats['p5_ci_high']  # p5 的95%置信区间上限
            stats_dict['p95_ci_low'] = quantile_ci_stats['p95_ci_low']  # p95 的95%置信区间下限
            stats_dict['p95_ci_high'] = quantile_ci_stats['p95_ci_high']  # p95 的95%置信区间上限
            stats_dict['p95_minus_p5'] = quantile_ci_stats['p95_minus_p5']  # p95 减 p5 的差值
            stats_dict['p95_minus_p5_ci_low'] = quantile_ci_stats['p95_minus_p5_ci_low']  # p95-p5 的95%置信区间下限
            stats_dict['p95_minus_p5_ci_high'] = quantile_ci_stats['p95_minus_p5_ci_high']  # p95-p5 的95%置信区间上限
            stats_dict['p25'] = np.percentile(clean_vals, 25)  # 25%分位数
            stats_dict['p50'] = np.percentile(clean_vals, 50)  # 中位数
            stats_dict['p75'] = np.percentile(clean_vals, 75)  # 75%分位数
            stats_dict.update(_compute_1d_additional_statistics(clean_vals, uniform_bins=10))  # 增加一维统计与均匀性指标
            
            confidence_level = 0.95  # 95%置信区间
            n = len(clean_vals)
            sem = stats_dict['std'] / np.sqrt(n)  # 标准误差
            ci = stats.t.interval(confidence_level, n-1, loc=stats_dict['mean'], scale=sem)
            stats_dict['ci_low'] = ci[0]  # 置信区间下限
            stats_dict['ci_high'] = ci[1]  # 置信区间上限
            
            statistics[material][output_var] = stats_dict  # 保存统计结果
            
            print(f"\n{material}/{output_var} 统计结果:")
            for key, value in stats_dict.items():
                if key not in ['hist_bins', 'hist_density']:
                    print(f"  {key:<10}: {value:.6f}")
            
            # 绘制当前子图的概率密度分布
            hist, edges = np.histogram(clean_vals, bins=num_bins, density=True)  # 获取直方图数据
            bootstrap_band = _compute_histogram_bootstrap_band(clean_vals, edges, confidence_level=0.95, n_bootstrap=2000, random_seed=0)  # 计算直方图概率密度的 Bootstrap 不确定度带
            bootstrap_low = bootstrap_band['density_ci_low']  # 获取每个 bin 密度的下置信限
            bootstrap_high = bootstrap_band['density_ci_high']  # 获取每个 bin 密度的上置信限
            bootstrap_low_step = np.append(bootstrap_low, bootstrap_low[-1])  # 扩展为 step='post' 所需长度
            bootstrap_high_step = np.append(bootstrap_high, bootstrap_high[-1])  # 扩展为 step='post' 所需长度
            ax.fill_between(
                edges,
                bootstrap_low_step,
                bootstrap_high_step,
                step='post',
                color=material_colors[material],
                alpha=0.35,
                linewidth=0
            )  # 绘制 Bootstrap 不确定度带
            ax.step(
                edges[:-1], 
                hist, 
                where='post',
                linewidth=1.5,
                linestyle='-',  # 统一使用实线
                color='black',
                alpha=1
            )
            #ax.text(0.98, 0.98, f"Var={stats_dict['var']:.6f}", transform=ax.transAxes, fontsize=10, va='top', ha='right')  # 在子图右上角显示方差
            
            # KDE拟合部分
            #kde = gaussian_kde(clean_vals)  # 创建KDE对象
            #kde_x = np.linspace(np.min(clean_vals), np.max(clean_vals), 1000)  # 生成拟合曲线的x值
            #kde_y = kde(kde_x)  # 计算KDE值
            #ax.plot(kde_x, kde_y, color='black', linestyle='--', linewidth=1.5)  # 绘制KDE拟合曲线
            
            # 设置轴标签（只在最左侧和最下方子图显示）
            if col_idx == 0:  # 第一列设置y轴标签
                ax.set_ylabel('Probability Density', fontsize=12)
            if row_idx == 2:  # 最后一行设置x轴标签
                ax.set_xlabel('Output Value (kg)', fontsize=12)
            
            # 添加网格和副刻度
            #ax.grid(True, which='major', linestyle='--', alpha=0.4)
            #ax.xaxis.set_minor_locator(AutoMinorLocator())

    print("\n联合样本空间统计结果:")
    for key, value in joint_space_statistics.items():
        print(f"  {key:<30}: {value:.6f}")

    # 添加整体标题
    #plt.suptitle(f'输入变量 "{target_var}" 的不确定性分析', fontsize=14, y=0.98)
    
    # 保存和显示结果
    #plt.subplots_adjust(left=0.18, right=0.98, bottom=0.05, top=0.96, wspace=0.20, hspace=0.20)  # 紧凑布局，避免元素遮挡

    # 设置行标题（放在左侧）
    for row_idx, material in enumerate(materials):  # 遍历材料设置行标题
        pos = axes[row_idx, 0].get_position()  # 获取当前行第一个子图位置
        y_center = (pos.y0 + pos.y1) / 2  # 计算行标题y轴位置
        fig.text(0.12, y_center, row_title_names.get(material, material), va='center', ha='right', fontsize=14)  # 行标题

    image_path = os.path.join(output_path, f'{target_var}_uncertainty_analysis_9subplots.png')  # 保存图像路径
    plt.savefig(image_path, dpi=600)  # 保存为文件
    plt.show()  # 显示图像

    print("\n分析完成! 所有结果已保存到目标目录")
    print(f"图像路径: {image_path}")

# 执行分析
if __name__ == "__main__":
    monte_carlo_uncertainty_analysis()