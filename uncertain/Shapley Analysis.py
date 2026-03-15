"""
功能概述：
---------
本脚本用于基于已有的抽样输入与结果文件，计算
  - 三种材料（UO2, CL, CR）
  - 三种输出（mLeav, mFrozen, mAccu）
在给定的某一个空间节点 (node_i, node_j) 和某一个时间点（时间索引或时间值）
处的标量响应，对 16 个输入变量（或样本文件中的所有列）的 Shapley 效应。

核心步骤：
1. 对每个材料：
   - 从对应的 LHS+Iman–Conover 输入样本文件中读取输入矩阵 X（自动识别样本数和变量数）。
2. 对每个输出：
   - 扫描对应结果目录（例如 UO2/mLeav/），自动识别有多少个样本结果文件。
   - 每个结果文件名开头的数字即样本序号（例如 "0UO2mLeav" → 样本 0）。
   - 解析结果文件，提取统一设定的节点和时间的一个标量 y。
   - 与输入矩阵的对应行组合，得到 (X_used, y_used)。
3. 基于 (X_used, y_used)，使用 k 近邻 (kNN) + 随机排列 的 Monte Carlo 算法近似 Shapley 效应：
   - 对很多个随机变量排列 π：
     - 依次扩展变量子集 S，利用 kNN 估计 V(S) = Var(E[Y | X_S])。
     - 对每个变量 j，在其加入前后 V(S) 的增量作为一次“边际贡献”。
   - 对边际贡献跨所有排列求平均，并归一化为输出方差的比例。

使用方式：
---------
1. 修改下方“用户配置区”中的路径为你本机的实际路径：
   - INPUT_SAMPLE_PATHS 中的三个 txt 文件路径。
   - OUTPUT_ROOT_DIR 为结果根目录 (uncertain)，里面有 UO2/ CL/ CR/。
2. 修改节点与时间选择：
   - TARGET_NODE_I, TARGET_NODE_J：空间节点索引，例如 (0, 0)。
   - TARGET_TIME_INDEX：时间索引（0 表示 time 行中的第一个时间值）。
     若你更希望用物理时间值，可将 TARGET_TIME_INDEX 设为 None，
     并设定 TARGET_TIME_VALUE 为一个浮点数（例如 3600.0 秒），脚本会寻找最接近的时间列。
3. 运行脚本：
   - 在命令行中执行：python shapley_from_results.py
4. 运行结束后：
   - 在 OUTPUT_SHAPLEY_CSV 路径下得到一份 CSV 文件，包含所有材料和输出的 Shapley 结果：
     列为：material, output, param_index, param_name, shapley_value。

依赖：
-----
- Python 3.10
- numpy >= 1.23.5
- pandas >= 1.5.3
- scikit-learn（用于 kNN）
"""

# ==============================
# 用户配置区（请根据实际情况修改）
# ==============================

import os  # 操作文件路径和目录
import re  # 用于正则表达式解析文件名中的样本序号
import io  # 用于把字符串包装成类文件对象给 pandas 读取
import numpy as np  # 数值计算
import pandas as pd  # 数据表处理
from sklearn.neighbors import NearestNeighbors  # k 近邻实现，用于条件期望估计

# 三种材料的输入样本文件路径（请修改为你的实际路径）                      # noqa: E501
INPUT_SAMPLE_PATHS = {  # 字典映射材料名 → 输入样本文件路径
    "UO2": r"C:\Users\11201\Desktop\sensities\uncertain\LHS抽样-混合分布-有相关性-样本\UO2_LHS_transformed.txt",  # UO2 样本
    "CL":  r"C:\Users\11201\Desktop\sensities\uncertain\LHS抽样-混合分布-有相关性-样本\CL_LHS_transformed.txt",   # CL 样本
    "CR":  r"C:\Users\11201\Desktop\sensities\uncertain\LHS抽样-混合分布-有相关性-样本\CR_LHS_transformed.txt",   # CR 样本
    #"UO2": r"C:\Users\11201\Desktop\sensities\uncertain\MC抽样-混合分布-没有相关性-样本\uncertain_samples_UO2.txt",  # UO2 样本
    #"CL":  r"C:\Users\11201\Desktop\sensities\uncertain\MC抽样-混合分布-没有相关性-样本\uncertain_samples_CL.txt",   # CL 样本
    #"CR":  r"C:\Users\11201\Desktop\sensities\uncertain\MC抽样-混合分布-没有相关性-样本\uncertain_samples_CR.txt",   # CR 样本

}

# 结果文件根目录（下面有 UO2/ CL/ CR/ 各材料子目录）                          # noqa: E501
OUTPUT_ROOT_DIR = r"C:\Users\11201\Desktop\sensities\Candle_To_Analysize\uncertain"  # 结果根目录

# 材料与输出名称列表（一般不需要改）                                          # noqa: E501
MATERIALS = ["UO2", "CL", "CR"]  # 三种材料
OUTPUT_NAMES = ["mLeav", "mFrozen", "mAccu"]  # 三种输出变量

# 节点与时间选择：统一对所有材料和输出使用相同的节点与时间点                    # noqa: E501
TARGET_NODE_I = 0  # 目标节点的第一个索引（例如 0）
TARGET_NODE_J = 0  # 目标节点的第二个索引（例如 0）

# 方式一：用时间索引（优先使用，如果不是 None）                              # noqa: E501
TARGET_TIME_INDEX = 1  # 整数索引：0 表示 time 行中第一个时间点，1 表示第二个，以此类推  # noqa: E501

# 方式二：用物理时间值（当 TARGET_TIME_INDEX 为 None 时使用）                 # noqa: E501
TARGET_TIME_VALUE = None  # 浮点时间值，例如 3600.0；若为 None 则不用

# Shapley 计算参数                                                             # noqa: E501
M_PERMUTATIONS = 1024  # 随机变量排列次数（越大越精确，但计算越慢）
N_NEIGHBORS = None    # kNN 中的邻居数；若为 None，则使用 sqrt(n_samples) 近似

# 输出结果 CSV 路径                                                            # noqa: E501
OUTPUT_SHAPLEY_CSV = r"C:\Users\11201\Desktop\sensities\shapley_results.csv"  # 保存 Shapley 结果的 CSV

# 随机种子（保证结果可复现）                                                   # noqa: E501
RANDOM_SEED = 123  # 随机数种子


# ============================================
# 功能块 1：读取输入样本文件（LHS+Iman–Conover）
# ============================================

def load_input_samples(file_path: str):
    """
    从带有注释行的 LHS 输入样本文件中读取数据。
    文件格式示例：
        # VM L AN AP ML MT TD
        # Unit: °C m m m² kg kg K
        0.1002 0.0900 0.3954 ...
        0.1546 0.0900 0.3954 ...
    返回：
        X : numpy.ndarray, 形状 (n_samples, n_features)
        param_names : list[str], 每个输入变量的名称
    """
    # 读取整个文件为字符串列表                                                   # noqa: E501
    with open(file_path, "r") as f:  # 以只读方式打开输入样本文件
        lines = f.readlines()  # 读取所有行，得到一个字符串列表

    header_names = None  # 初始化参数名列表为 None，后续尝试从注释中解析
    data_lines = []  # 用于存储真正的数据行（非注释行）

    # 遍历每一行，分别处理注释行与数据行                                       # noqa: E501
    for line in lines:  # 逐行遍历文件内容
        stripped = line.strip()  # 去掉每行首尾空白字符
        if not stripped:  # 如果这一行是空行
            continue  # 直接跳过
        if stripped.startswith("#"):  # 如果是注释行（以 # 开头）
            maybe_header = stripped[1:].strip()  # 去掉 # 后再去除首尾空白，可能是参数名行
            # 只在不是“Unit”行时，把它视作参数名行                               # noqa: E501
            if maybe_header and not maybe_header.lower().startswith("unit"):  # 非空且不以 unit 开头
                header_names = maybe_header.split()  # 按空白切分得到参数名列表
            continue  # 注释行不进入数据行
        # 剩余的是数据行                                                        # noqa: E501
        data_lines.append(stripped)  # 将数据行加入列表

    if not data_lines:  # 如果没有任何数据行
        raise ValueError(f"输入样本文件中没有数据行：{file_path}")  # 抛出错误提示

    # 将数据行拼成一个大字符串，用 StringIO 让 pandas 当成文件读取                 # noqa: E501
    data_str = "\n".join(data_lines)  # 用换行符拼接所有数据行
    buffer = io.StringIO(data_str)  # 用 StringIO 封装成文件对象

    # 使用 pandas 读取数据，按任意空白分隔，且没有表头                         # noqa: E501
    df = pd.read_csv(buffer, delim_whitespace=True, header=None)  # 读入为 DataFrame

    X = df.values  # 提取底层 numpy 数组作为输入矩阵
    n_features = X.shape[1]  # 变量个数 = 列数

    # 处理参数名：如果从注释中解析到了且数量匹配，则使用；否则自动生成           # noqa: E501
    if header_names is None or len(header_names) != n_features:  # 若没有参数名或数量不匹配
        param_names = [f"X{i+1}" for i in range(n_features)]  # 生成 X1, X2, ... 这样的名称
    else:
        param_names = header_names  # 使用注释中提供的参数名

    return X, param_names  # 返回输入矩阵和参数名列表


# ====================================================
# 功能块 2：解析单个结果文件，提取指定节点和时间的标量
# ====================================================

def extract_scalar_from_result_file(
    file_path: str,
    target_node_i: int,
    target_node_j: int,
    target_time_index: int = None,
    target_time_value: float = None,
):
    """
    从一个结果文件中提取指定节点 (target_node_i, target_node_j) 和指定时间点
    对应的输出标量值。

    文件格式示例：
        time 0 1 2 ...
        mLeav(0,0) 2.07 2.08 2.09 ...
        mLeav(0,1) ...
        ...

    参数：
        file_path : 结果文件路径
        target_node_i, target_node_j : 节点索引 (i, j)
        target_time_index : 时间索引（优先使用），例如 0 表示第一个时间点
        target_time_value : 物理时间值，如果 target_time_index 为 None，则使用
                            与该值最接近的时间列。

    返回：
        value : float, 对应的标量输出
    """
    # 读取文件所有非空行                                                         # noqa: E501
    with open(file_path, "r") as f:  # 以只读模式打开结果文件
        lines = [line.strip() for line in f if line.strip()]  # 去掉空行并 strip，每行是字符串

    if not lines:  # 若结果文件为空
        raise ValueError(f"结果文件为空：{file_path}")  # 抛出错误

    # 找到以 "time" 开头的那一行                                                # noqa: E501
    time_line_idx = None  # 初始化时间行索引
    for idx, line in enumerate(lines):  # 遍历每一行
        parts = line.split()  # 按空白切分为单词
        if parts and parts[0].lower() == "time":  # 若首个单词为 time（大小写不敏感）
            time_line_idx = idx  # 记录时间行位置
            break  # 找到后即可退出循环

    if time_line_idx is None:  # 若没找到时间行
        raise ValueError(f"未找到时间行 'time'：{file_path}")  # 抛出错误

    # 解析时间向量                                                               # noqa: E501
    time_tokens = lines[time_line_idx].split()  # 时间行拆分为单词列表
    if len(time_tokens) <= 1:  # 至少应有 "time" + 1 个时间值
        raise ValueError(f"时间行格式不正确：{lines[time_line_idx]}")  # 抛出错误

    time_values = [float(tok) for tok in time_tokens[1:]]  # 跳过第一个 'time'，剩下的转为 float

    # 确定使用哪一列时间：优先使用索引，否则使用物理时间值                        # noqa: E501
    if target_time_index is not None:  # 如果指定了时间索引
        if target_time_index < 0 or target_time_index >= len(time_values):  # 检查索引范围
            raise IndexError(
                f"target_time_index={target_time_index} 越界，时间点数量为 {len(time_values)}"
            )
        col_idx = target_time_index  # 使用给定的时间索引
    else:
        if target_time_value is None:  # 若既没有索引也没有物理时间值
            raise ValueError("需要指定 target_time_index 或 target_time_value 至少一个")  # 抛错
        # 找到与 target_time_value 最接近的时间点索引                             # noqa: E501
        diffs = [abs(tv - target_time_value) for tv in time_values]  # 计算每个时间值与目标值的差
        col_idx = int(np.argmin(diffs))  # 取差值最小的索引

    # 搜索目标节点对应的那一行                                                   # noqa: E501
    target_value = None  # 初始化返回值
    for line in lines[time_line_idx + 1:]:  # 从时间行下一行开始遍历
        parts = line.split()  # 按空白切分
        if len(parts) < 2:  # 至少需要变量名 + 一个数据
            continue  # 格式不对则跳过
        name = parts[0]  # 第一个字段是变量名，例如 mLeav(0,0)

        # 尝试从变量名中解析出 (i,j) 节点索引                                   # noqa: E501
        if "(" in name and ")" in name:  # 仅当包含括号时才解析
            inside = name[name.find("(") + 1:name.find(")")]  # 括号内的内容，例如 "0,0"
            idx_parts = inside.split(",")  # 用逗号分割成两个索引
            if len(idx_parts) == 2:  # 确保有两个索引
                try:
                    i_val = int(idx_parts[0])  # 解析第一个索引
                    j_val = int(idx_parts[1])  # 解析第二个索引
                except ValueError:
                    continue  # 若不能转为整数，则跳过该行

                if i_val == target_node_i and j_val == target_node_j:  # 如果是目标节点
                    # 这一行后面的都是该节点在各时间点的数据                     # noqa: E501
                    data_tokens = parts[1:]  # 去掉变量名，剩下的是数值字符串
                    if col_idx >= len(data_tokens):  # 检查时间索引是否越界
                        raise IndexError(
                            f"时间列索引 {col_idx} 超出数据列数 {len(data_tokens)}：文件 {file_path}"
                        )
                    target_value = float(data_tokens[col_idx])  # 取出目标时间点的值
                    break  # 找到后可以结束搜索

    if target_value is None:  # 若没找到目标节点行
        raise ValueError(
            f"在文件 {file_path} 中未找到节点 ({target_node_i},{target_node_j}) 的数据行"
        )

    return target_value  # 返回提取到的标量值


# ==============================================
# 功能块 3：汇总一个“材料+输出”的数据集 (X, y)
# ==============================================

def collect_dataset_for_material_output(
    material: str,
    output_name: str,
    input_samples_path: str,
    output_root_dir: str,
    target_node_i: int,
    target_node_j: int,
    target_time_index: int = None,
    target_time_value: float = None,
):
    """
    对于指定的材料和输出，读取：
      - 输入样本文件（LHS+Iman–Conover 结果）
      - 结果目录下的所有结果文件（例如 UO2/mLeav/ 下的所有文件）
    自动识别样本数量，并为每个样本提取一个标量输出，组合得到 (X_used, y_used)。

    参数：
        material : 材料名，例如 "UO2"
        output_name : 输出变量名，例如 "mLeav"
        input_samples_path : 该材料对应的输入样本文件路径
        output_root_dir : 结果根目录
        target_node_i, target_node_j : 目标节点索引
        target_time_index : 时间索引（优先）
        target_time_value : 时间值（可选）

    返回：
        X_used : numpy.ndarray, 形状 (n_used, n_features)，参与 Shapley 的输入样本
        y_used : numpy.ndarray, 形状 (n_used,)，对应标量输出
        param_names : list[str]，输入变量名称
    """
    # 读取该材料的输入样本矩阵 X 以及变量名                                     # noqa: E501
    X_all, param_names = load_input_samples(input_samples_path)  # 调用前面的函数读取样本
    n_all_samples, n_features = X_all.shape  # 输入样本的总行数和变量个数

    # 构造该材料 + 输出的结果目录路径，例如 uncertain/UO2/mLeav                 # noqa: E501
    result_dir = os.path.join(output_root_dir, material, output_name)  # 拼接结果目录路径

    if not os.path.isdir(result_dir):  # 检查目录是否存在
        raise FileNotFoundError(f"结果目录不存在：{result_dir}")  # 如果不存在则抛错

    # 列出该目录下的所有文件                                                    # noqa: E501
    all_files = sorted(os.listdir(result_dir))  # 按文件名排序，方便重现性

    X_list = []  # 用于存放参与计算的输入样本行
    y_list = []  # 用于存放对应的标量输出
    used_indices = []  # 存放使用到的样本序号，便于检查

    # 遍历结果目录中的所有文件                                                 # noqa: E501
    for fname in all_files:  # 对每一个文件名进行处理
        if fname.startswith("."):  # 跳过隐藏文件（例如 .DS_Store）
            continue  # 不处理隐藏文件

        base_name, _ext = os.path.splitext(fname)  # 分离文件名与后缀（如果没有后缀，ext 为空）

        # 粗筛：文件名中应包含材料名和输出名，防止混入其它文件                     # noqa: E501
        if material not in base_name or output_name not in base_name:  # 如果文件名不包含材料或输出名
            continue  # 跳过该文件

        # 从文件名开头解析样本序号，例如 "0UO2mLeav" → 0                          # noqa: E501
        m = re.match(r"(\d+)", base_name)  # 用正则匹配文件名开头的一串数字
        if not m:  # 如果没有匹配到数字
            continue  # 跳过该文件
        sample_idx = int(m.group(1))  # 将匹配到的数字字符串转为整数

        if sample_idx < 0 or sample_idx >= n_all_samples:  # 检查样本序号是否在输入样本行数范围内
            # 若样本序号超出输入样本行数，说明数据不匹配，这里选择跳过                     # noqa: E501
            print(f"[警告] 文件 {fname} 的样本序号 {sample_idx} 超出输入样本行数 {n_all_samples}，已跳过")  # 提示用户
            continue  # 跳过该文件

        file_path = os.path.join(result_dir, fname)  # 构造该结果文件的完整路径

        # 从结果文件中提取目标节点和时间的标量输出                               # noqa: E501
        try:
            y_val = extract_scalar_from_result_file(
                file_path=file_path,
                target_node_i=target_node_i,
                target_node_j=target_node_j,
                target_time_index=target_time_index,
                target_time_value=target_time_value,
            )
        except Exception as e:
            # 若单个文件解析失败，给出提示并跳过该文件                             # noqa: E501
            print(f"[警告] 提取结果文件 {file_path} 时出错：{e}，已跳过该样本")  # 打印警告
            continue  # 跳过该文件

        # 将输入样本的对应行和输出标量加入列表                                   # noqa: E501
        X_list.append(X_all[sample_idx, :])  # 加入样本矩阵的第 sample_idx 行
        y_list.append(y_val)  # 加入对应的输出标量
        used_indices.append(sample_idx)  # 记录使用过的样本序号

    if not X_list:  # 若没有成功收集到任何样本
        raise ValueError(f"材料 {material}, 输出 {output_name} 没有可用的结果文件")  # 抛出错误

    # 将列表合并为 numpy 数组                                                   # noqa: E501
    X_used = np.vstack(X_list)  # 纵向堆叠成 (n_used, n_features) 数组
    y_used = np.array(y_list, dtype=float)  # 输出标量转为 (n_used,) 数组

    print(
        f"[信息] 材料 {material}, 输出 {output_name}: 使用了 {X_used.shape[0]} 个样本（自动识别）"
    )  # 打印提示信息

    return X_used, y_used, param_names  # 返回该材料+输出对应的数据集和参数名列表


# ====================================================
# 功能块 4：kNN 估计 V(S) = Var(E[Y | X_S]) 的辅助函数
# ====================================================

def estimate_var_conditional_knn(
    X: np.ndarray,
    y: np.ndarray,
    subset: frozenset,
    n_neighbors: int,
):
    """
    使用 k 近邻方式估计 V(S) = Var(E[Y | X_S])。

    步骤：
      1. 取出 X 的子集列 X_S（只保留 subset 中的变量索引）。
      2. 用 NearestNeighbors 在 X_S 上拟合 kNN。
      3. 对每个样本 i，找其在子空间中的 k 个最近邻样本索引 neighbors_i。
      4. 对这些邻居的 y 值求平均，得到 E[Y | X_S = X_S^{(i)}] 的估计 m_i。
      5. 对所有 m_i 求样本方差，作为 V(S) 的估计。

    参数：
        X : 输入样本矩阵 (n_samples, n_features)
        y : 输出向量 (n_samples,)
        subset : 一个 frozenset，表示变量索引集合（例如 frozenset({0,1,2})）
        n_neighbors : kNN 的邻居个数

    返回：
        var_est : float, 对 V(S) 的估计
    """
    n_samples, n_features = X.shape  # 获取样本数和特征数
    if not subset:  # 如果子集 S 为空
        return 0.0  # V(空集) = Var(E[Y]) = Var(常数) = 0

    # 将子集索引转换为排序后的列表，保证一致性                                   # noqa: E501
    cols = sorted(list(subset))  # 将 frozenset 转成排好序的列表

    # 在 X 中取出这些列，得到子集特征矩阵 X_S                                    # noqa: E501
    X_sub = X[:, cols]  # 取子集特征

    # 对输出 y 做一份扁平化拷贝                                                 # noqa: E501
    y = np.asarray(y).reshape(-1)  # 确保 y 是一维数组

    # 构造 kNN 模型                                                              # noqa: E501
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")  # 使用默认算法构建 kNN
    nn.fit(X_sub)  # 在子空间数据上拟合 kNN

    # 对每个样本点，找到其在子空间上的 k 个最近邻索引                            # noqa: E501
    neighbors_indices = nn.kneighbors(return_distance=False)  # 返回邻居索引矩阵 (n_samples, k)

    # 计算每个样本点的条件期望估计 m_i                                           # noqa: E501
    m_vals = np.zeros(n_samples, dtype=float)  # 初始化条件期望数组
    for i in range(n_samples):  # 遍历每个样本
        idxs = neighbors_indices[i]  # 第 i 个样本的 k 个邻居的索引
        m_vals[i] = float(np.mean(y[idxs]))  # 邻居的 y 值求平均作为 m_i

    # 对 m_i 求样本方差，作为 V(S) 的估计                                        # noqa: E501
    var_est = float(np.var(m_vals, ddof=1))  # 使用无偏估计 (ddof=1)

    return var_est  # 返回估计值


# ==========================================
# 功能块 5：基于 kNN 的 Shapley 效应估计主函数
# ==========================================

def compute_shapley_effects_knn(
    X: np.ndarray,
    y: np.ndarray,
    m_permutations: int = 128,
    n_neighbors: int = None,
    random_state: int = None,
):
    """
    使用随机变量排列 + kNN 条件方差估计的方式，近似计算 Shapley 效应。

    算法概要：
      - 对 m_permutations 次：
        1. 随机生成一个变量排列 π（例如 [2,5,1,...]）。
        2. 从空集 S = {} 开始，依次加入 π 中的变量 j：
           - 计算 V(S) 和 V(S ∪ {j})，V(S) = Var(E[Y|X_S])。
           - 对变量 j 的一次边际贡献为 Δ_j = V(S ∪ {j}) - V(S)。
           - 将 Δ_j 累加到该变量的 Shapley 累积值中。
           - 更新 S ← S ∪ {j}。
      - 最后将各变量的累积贡献除以 (m_permutations * Var(Y))，得到归一化的 Shapley 效应。

    参数：
        X : 输入样本矩阵 (n_samples, n_features)
        y : 输出向量 (n_samples,)
        m_permutations : 随机排列次数，越大结果越稳定
        n_neighbors : kNN 中的邻居数，如果为 None，则使用 floor(sqrt(n_samples)) 并截断到 [2, n_samples]
        random_state : 随机数种子，保证结果可复现

    返回：
        shapley_values : numpy.ndarray, 形状 (n_features,)，各变量的 Shapley 效应（和约为 1）
    """
    X = np.asarray(X)  # 确保 X 为 numpy 数组
    y = np.asarray(y).reshape(-1)  # 确保 y 为一维数组

    n_samples, n_features = X.shape  # 获取样本数和特征数

    # 计算输出的总方差 Var(Y)，用于归一化                                      # noqa: E501
    var_y = float(np.var(y, ddof=1))  # 无偏方差估计
    if var_y <= 0.0:  # 若方差为 0 或负（数值问题）
        raise ValueError("输出方差为零，无法计算 Shapley 效应")  # 抛出错误

    # 设置默认的邻居数：sqrt(n_samples)，并裁剪到 [2, n_samples]                 # noqa: E501
    if n_neighbors is None:  # 若用户未指定邻居数
        approx = int(np.sqrt(n_samples))  # 取样本数平方根的整数部分
        n_neighbors = max(2, min(approx, n_samples))  # 限制在至少 2，最多 n_samples
    else:
        # 确保用户给的邻居数在合理范围内                                      # noqa: E501
        n_neighbors = max(2, min(int(n_neighbors), n_samples))  # 截断到 [2, n_samples]

    # 初始化随机数生成器                                                      # noqa: E501
    rng = np.random.default_rng(random_state)  # 使用 NumPy 的 Generator

    # 每个变量的累积边际贡献                                                  # noqa: E501
    phi = np.zeros(n_features, dtype=float)  # 形状 (n_features,)

    # 缓存 V(S) 的结果，避免对子集重复计算                                       # noqa: E501
    V_cache = {frozenset(): 0.0}  # 空集的 V(S) = 0，先存入缓存

    # 主循环：随机生成若干变量排列                                             # noqa: E501
    for m in range(m_permutations):  # 对每一次排列
        perm = rng.permutation(n_features)  # 随机生成一个 0..(n_features-1) 的排列
        S_before = frozenset()  # 初始子集 S = 空集

        for idx in perm:  # 按排列顺序依次加入变量
            S_with = frozenset(set(S_before) | {idx})  # 新的子集 S ∪ {idx}

            # 如果缓存中没有 V(S_before)，则用 kNN 估计并缓存                     # noqa: E501
            if S_before not in V_cache:  # 检查缓存
                V_cache[S_before] = estimate_var_conditional_knn(
                    X=X,
                    y=y,
                    subset=S_before,
                    n_neighbors=n_neighbors,
                )

            # 如果缓存中没有 V(S_with)，则也进行估计并缓存                        # noqa: E501
            if S_with not in V_cache:  # 检查缓存
                V_cache[S_with] = estimate_var_conditional_knn(
                    X=X,
                    y=y,
                    subset=S_with,
                    n_neighbors=n_neighbors,
                )

            V_before = V_cache[S_before]  # 取出 V(S_before)
            V_with = V_cache[S_with]      # 取出 V(S_with)

            contrib = V_with - V_before  # 边际贡献 Δ_j = V(S ∪ {j}) - V(S)

            phi[idx] += contrib  # 将边际贡献累加到该变量的总贡献中

            S_before = S_with  # 更新 S，为下一个变量加入做准备

    # 将累积贡献除以 (m_permutations * Var(Y)) 得到归一化 Shapley 效应             # noqa: E501
    shapley_values = phi / (m_permutations * var_y)  # 归一化

    return shapley_values  # 返回每个变量的 Shapley 值（一维数组）


# ===========================
# 功能块 6：主流程（整合所有部分）
# ===========================

def main():
    """
    主函数：对三种材料 × 三种输出，循环执行：
      1. 汇总数据集 (X, y)；
      2. 基于 kNN 计算 Shapley 效应；
      3. 汇总所有结果到一个 DataFrame 并保存为 CSV。
    """
    # 用于汇总各材料和输出的 Shapley 结果                                      # noqa: E501
    results_records = []  # 列表，每个元素是一行结果的字典

    # 遍历每一个材料                                                           # noqa: E501
    for material in MATERIALS:  # 对 UO2, CL, CR 逐个处理
        if material not in INPUT_SAMPLE_PATHS:  # 确保有对应的样本文件路径
            print(f"[警告] 未在 INPUT_SAMPLE_PATHS 中找到材料 {material} 的样本路径，已跳过")  # 打印警告
            continue  # 跳过该材料

        input_path = INPUT_SAMPLE_PATHS[material]  # 取出该材料的样本文件路径

        # 遍历每一个输出变量                                                   # noqa: E501
        for output_name in OUTPUT_NAMES:  # 对 mLeav, mFrozen, mAccu 逐个处理
            print(f"\n[信息] 正在处理 材料={material}, 输出={output_name} ...")  # 打印当前处理信息

            # 收集该材料 + 输出 的数据集 (X_used, y_used)                       # noqa: E501
            try:
                X_used, y_used, param_names = collect_dataset_for_material_output(
                    material=material,
                    output_name=output_name,
                    input_samples_path=input_path,
                    output_root_dir=OUTPUT_ROOT_DIR,
                    target_node_i=TARGET_NODE_I,
                    target_node_j=TARGET_NODE_J,
                    target_time_index=TARGET_TIME_INDEX,
                    target_time_value=TARGET_TIME_VALUE,
                )
            except Exception as e:
                print(f"[错误] 收集数据集失败：材料={material}, 输出={output_name}, 错误信息：{e}")  # 打印错误信息
                continue  # 跳过该组，继续下一组

            # 计算该 (材料, 输出) 的 Shapley 效应                               # noqa: E501
            try:
                shapley_vals = compute_shapley_effects_knn(
                    X=X_used,
                    y=y_used,
                    m_permutations=M_PERMUTATIONS,
                    n_neighbors=N_NEIGHBORS,
                    random_state=RANDOM_SEED,
                )
            except Exception as e:
                print(f"[错误] 计算 Shapley 效应失败：材料={material}, 输出={output_name}, 错误信息：{e}")  # 打印错误信息
                continue  # 跳过该组

            # 将结果写入汇总列表，每个变量一行                                   # noqa: E501
            n_features = X_used.shape[1]  # 变量个数
            if len(param_names) != n_features:  # 参数名数量与特征数不一致时发出警告
                print(f"[警告] 参数名数量 ({len(param_names)}) 与特征数 ({n_features}) 不一致，将自动截断/补全")  # 提示用户
                # 截断或补全参数名长度                                            # noqa: E501
                if len(param_names) < n_features:
                    # 补充缺失的参数名                                            # noqa: E501
                    extra = [f"X{len(param_names) + i + 1}" for i in range(n_features - len(param_names))]  # 生成额外变量名
                    param_names = param_names + extra  # 合并
                else:
                    # 截断多余的参数名                                            # noqa: E501
                    param_names = param_names[:n_features]  # 保留前 n_features 个

            # 遍历每一个变量，将 Shapley 值记录下来                             # noqa: E501
            for j in range(n_features):  # 对每一个变量索引
                record = {  # 构造一条结果记录
                    "material": material,  # 材料名
                    "output": output_name,  # 输出名
                    "param_index": j,  # 变量索引（从 0 开始）
                    "param_name": param_names[j],  # 变量名（例如 VM, L 等）
                    "shapley_value": float(shapley_vals[j]),  # 该变量的 Shapley 效应
                }
                results_records.append(record)  # 将该记录加入列表

    # 将所有结果记录转换为 DataFrame                                           # noqa: E501
    if results_records:  # 若列表非空
        results_df = pd.DataFrame(results_records)  # 转为 DataFrame

        # 打印简单统计信息，例如每个材料+输出组合的 Shapley 值是否近似和为 1         # noqa: E501
        for (material, output_name), g in results_df.groupby(["material", "output"]):  # 按 (材料, 输出) 分组
            ssum = g["shapley_value"].sum()  # 各变量 Shapley 值求和
            print(f"[检查] 材料={material}, 输出={output_name}, Shapley 和 ≈ {ssum:.4f}")  # 打印检查信息

        # 保存到 CSV 文件                                                       # noqa: E501
        os.makedirs(os.path.dirname(OUTPUT_SHAPLEY_CSV), exist_ok=True)  # 确保输出目录存在
        results_df.to_csv(OUTPUT_SHAPLEY_CSV, index=False)  # 将结果 DataFrame 保存为 CSV（不写行索引）

        print(f"\n[完成] Shapley 结果已保存至：{OUTPUT_SHAPLEY_CSV}")  # 提示用户保存路径
    else:
        print("[警告] 未生成任何 Shapley 结果，请检查数据路径和配置")  # 若没有任何结果则打印警告


# =======================
# 脚本入口
# =======================

if __name__ == "__main__":  # 确保仅在直接运行脚本时执行 main()
    main()  # 调用主函数
