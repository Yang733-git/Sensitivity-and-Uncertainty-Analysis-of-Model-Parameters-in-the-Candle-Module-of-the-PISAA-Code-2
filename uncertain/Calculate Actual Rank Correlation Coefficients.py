"""
功能：                                                                 # 代码功能说明
1. 自动扫描指定目录下的样本文件（支持 .txt、.csv、.tsv、.dat、.xlsx、.xls）。  # 功能1
2. 将“一个样本文件”视为“一次完整抽样实现”。                                   # 功能2
3. 对每个样本文件中的全部参数列计算 Spearman 秩相关矩阵。                      # 功能3
4. 跨多个样本文件，对每一对参数的实际样本秩相关系数统计均值和标准差。         # 功能4
5. 输出三类结果：                                                             # 功能5
   - 每个文件各自的 Spearman 秩相关矩阵；                                     # 输出1
   - 所有参数对的 achieved sample rank-correlation (mean ± SD) 长表；         # 输出2
   - 所有参数的均值矩阵和标准差矩阵。                                         # 输出3

使用方式：                                                                   # 使用方式说明
1. 修改本脚本最前面的 SAMPLE_DIR 为你的样本目录路径。                         # 步骤1
2. 如需递归扫描子目录，将 RECURSIVE 设为 True；否则设为 False。                # 步骤2
3. 直接运行脚本。                                                             # 步骤3
4. 结果会自动写入 OUTPUT_DIR 指定目录。                                       # 步骤4

关于“均值 ± 标准差”的统计口径：                                              # 统计口径说明
- 一个文件内部的 2000 个样本点，只能算出“这一次实现”的一套 Spearman 相关矩阵。 # 说明1
- 若目录下有多个样本文件，则把这些文件视为多次独立实现，跨文件统计 mean ± SD。 # 说明2
- 若目录下只有 1 个样本文件，则 mean 可以计算，但 SD 在统计上没有定义。        # 说明3
"""                                                                             # 文档字符串结束

from __future__ import annotations                                               # 兼容 Python 3.10 的类型注解前向引用支持

import re                                                                       # 用于正则表达式处理
from pathlib import Path                                                        # 用于跨平台路径处理
from typing import Dict, List, Optional, Sequence, Tuple                        # 用于类型注解

import numpy as np                                                              # 用于数值计算
import pandas as pd                                                             # 用于数据读取与表格处理

# ========================= 1. 用户可修改的配置区 =========================     # 配置区开始
SAMPLE_DIR = r"C:\Users\11201\Desktop\sensities\uncertain\批量抽样样本"                                  # 样本目录路径，改成你的真实目录
OUTPUT_DIR = r"C:\Users\11201\Desktop\sensities\uncertain\批量抽样样本"                # 输出目录路径，结果文件会保存到这里
RECURSIVE = False                                                               # 是否递归扫描子目录，True=扫描子目录，False=只扫当前目录
SUPPORTED_PATTERNS = ["*.txt", "*.csv", "*.tsv", "*.dat", "*.xlsx", "*.xls"]   # 支持扫描的文件模式
PAIR_TABLE_DECIMALS = 6                                                         # 输出表格中相关系数保留的小数位数
# ========================= 1. 用户可修改的配置区结束 =====================     # 配置区结束


# ========================= 2. 基础工具函数 ===============================     # 工具函数区开始
def is_number_like(value: object) -> bool:                                      # 判断一个值是否“像数字”
    text = str(value).strip()                                                   # 先把值转成字符串并去掉首尾空白
    if text == "":                                                              # 如果是空字符串
        return False                                                            # 返回 False
    try:                                                                        # 尝试转成浮点数
        float(text)                                                             # 执行浮点转换
        return True                                                             # 能成功转为数字则返回 True
    except (TypeError, ValueError):                                             # 若抛出类型或数值异常
        return False                                                            # 返回 False


def detect_delimiter_from_line(line: str) -> str:                               # 从一行文本中自动识别分隔符
    stripped = line.strip().lstrip("#").strip()                                 # 去掉首尾空白和可能存在的井号
    candidates = [                                                              # 准备若干候选分隔符及其切分后的字段数
        ("\t", len(stripped.split("\t"))),                                      # 候选1：制表符
        (",", len(stripped.split(","))),                                        # 候选2：逗号
        (";", len(stripped.split(";"))),                                        # 候选3：分号
        (r"\s+", len(re.split(r"\s+", stripped)) if stripped else 0),           # 候选4：任意空白符
    ]                                                                           # 候选结束
    candidates = [(sep, cnt) for sep, cnt in candidates if cnt > 1]            # 只保留能分出多个字段的候选项
    if not candidates:                                                          # 如果没有任何可用候选项
        raise ValueError(f"无法识别分隔符，请检查表头行：{line!r}")              # 抛出异常提示用户检查文件
    candidates.sort(key=lambda item: item[1], reverse=True)                     # 优先选择切分字段数最多的分隔符
    return candidates[0][0]                                                     # 返回最优分隔符


def clean_column_names(columns: Sequence[object]) -> List[str]:                 # 清洗列名并避免重复
    cleaned: List[str] = []                                                     # 初始化清洗后的列名列表
    seen: Dict[str, int] = {}                                                   # 记录每个列名出现次数以处理重名
    for col in columns:                                                         # 遍历原始列名
        text = str(col).strip().lstrip("#").strip()                             # 去掉首尾空白和可能存在的井号
        text = re.sub(r"\s+", "_", text)                                        # 把连续空白替换成下划线
        if text == "":                                                          # 若清洗后为空
            text = "EMPTY"                                                      # 用 EMPTY 占位
        if text in seen:                                                        # 如果列名已经出现过
            seen[text] += 1                                                     # 重复计数加一
            text = f"{text}_{seen[text]}"                                       # 在列名后追加编号
        else:                                                                   # 如果列名首次出现
            seen[text] = 0                                                      # 初始化重复计数
        cleaned.append(text)                                                    # 追加到结果列表
    return cleaned                                                              # 返回清洗后的列名列表


def format_mean_sd(mean_value: float, sd_value: float, decimals: int) -> str:   # 把均值和标准差格式化成“mean ± SD”字符串
    mean_text = f"{mean_value:.{decimals}f}"                                    # 格式化均值文本
    if np.isnan(sd_value):                                                      # 如果标准差是 NaN
        return f"{mean_text} ± NA"                                              # 返回“均值 ± NA”
    return f"{mean_text} ± {sd_value:.{decimals}f}"                             # 返回“均值 ± 标准差”
# ========================= 2. 基础工具函数结束 ===========================     # 工具函数区结束


# ========================= 3. 文件发现与读取函数 =========================     # 文件读取区开始
def discover_sample_files(sample_dir: str, recursive: bool, output_dir: str) -> List[Path]:  # 自动发现目录下的样本文件，并排除输出目录
    directory = Path(sample_dir).expanduser().resolve()                         # 标准化样本目录路径
    output_path = Path(output_dir).expanduser().resolve()                       # 标准化输出目录路径
    if not directory.exists():                                                  # 如果目录不存在
        raise FileNotFoundError(f"样本目录不存在：{directory}")                  # 抛出目录不存在异常
    if not directory.is_dir():                                                  # 如果路径不是目录
        raise NotADirectoryError(f"给定路径不是目录：{directory}")               # 抛出不是目录异常

    discovered: List[Path] = []                                                 # 初始化发现到的文件列表
    batch_file_name_pattern = re.compile(r".+_LHS_samples_\d+$", re.IGNORECASE)  # 按截图中的文件名模式识别批量样本文件主名
    for pattern in SUPPORTED_PATTERNS:                                          # 遍历所有支持的文件模式
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)  # 根据递归开关选择扫描方式
        for path in iterator:                                                   # 遍历当前模式匹配到的路径
            if not path.is_file():                                              # 如果不是文件
                continue                                                        # 跳过
            resolved_path = path.resolve()                                      # 解析为绝对路径
            if not batch_file_name_pattern.fullmatch(resolved_path.stem):       # 如果文件主名不符合截图中的批量样本命名方式
                continue                                                        # 跳过，避免把结果文件或其他无关文件当成输入样本
            discovered.append(resolved_path)                                    # 追加到发现列表

    discovered = sorted(set(discovered), key=lambda p: p.name.lower())          # 去重并按文件名排序，保证结果稳定
    if not discovered:                                                          # 如果没有找到任何文件
        raise FileNotFoundError(f"目录下未找到符合批量样本命名方式的样本文件：{directory}")      # 抛出异常提示用户
    return discovered                                                           # 返回样本文件列表


def extract_header_from_text_file(file_path: Path) -> Tuple[Optional[List[str]], Optional[str]]:  # 从文本文件中提取注释表头
    preview_lines: List[str] = []                                               # 用于保存前若干行预览内容
    with file_path.open("r", encoding="utf-8", errors="replace") as file_obj:   # 以 UTF-8 打开文本文件并容错替换坏字符
        for _ in range(30):                                                     # 最多读取前 30 行用于判断结构
            line = file_obj.readline()                                          # 读取一行
            if not line:                                                        # 如果已到文件末尾
                break                                                           # 跳出循环
            preview_lines.append(line.rstrip("\n"))                             # 去掉换行符后保存到预览列表

    for line in preview_lines:                                                  # 逐行检查预览内容
        stripped = line.strip()                                                 # 去掉首尾空白
        if stripped == "":                                                      # 如果是空行
            continue                                                            # 跳过空行
        if not stripped.startswith("#"):                                        # 如果不是注释行
            continue                                                            # 跳过，因为附件格式的表头在注释行中
        content = stripped.lstrip("#").strip()                                  # 去掉开头井号，保留注释内容
        if content == "":                                                       # 如果去掉井号后还是空内容
            continue                                                            # 跳过
        delimiter = detect_delimiter_from_line(content)                         # 自动识别这行的分隔符
        tokens = re.split(r"\s+", content) if delimiter == r"\s+" else [token.strip() for token in content.split(delimiter)]  # 按识别到的分隔符切分字段
        non_numeric_count = sum(not is_number_like(token) for token in tokens)  # 统计非数字字段个数
        if len(tokens) >= 2 and non_numeric_count >= max(1, len(tokens) // 2):  # 若字段数足够且大部分更像“列名”
            return tokens, delimiter                                            # 认定为表头并返回
    return None, None                                                           # 如果没找到注释表头则返回空


def maybe_drop_units_row(df: pd.DataFrame) -> pd.DataFrame:                     # 尝试删除 Excel 中可能存在的单位行
    if df.empty:                                                                # 如果表为空
        return df                                                               # 直接返回
    first_row = df.iloc[0]                                                      # 取第一行做判断
    number_like_count = sum(is_number_like(value) for value in first_row.tolist())  # 统计第一行中像数字的单元格个数
    if number_like_count <= max(1, len(df.columns) // 3):                       # 如果第一行大多数不像数字，则很可能是单位行
        return df.iloc[1:].reset_index(drop=True)                               # 删除第一行并重建行索引
    return df                                                                   # 否则保留原表不变


def load_single_sample_file(file_path: Path) -> pd.DataFrame:                   # 读取单个样本文件并返回纯数值 DataFrame
    suffix = file_path.suffix.lower()                                           # 获取文件后缀并转成小写

    if suffix in {".txt", ".csv", ".tsv", ".dat"}:                              # 如果是常见文本样本文件
        header_names, delimiter = extract_header_from_text_file(file_path)       # 先尝试提取注释表头
        if header_names is not None and delimiter is not None:                   # 若成功识别到附件这类“注释表头”格式
            df = pd.read_csv(                                                    # 读取文本数据
                file_path,                                                       # 输入文件路径
                sep=delimiter,                                                   # 使用自动识别到的分隔符
                engine="python",                                                 # 使用 python 引擎以支持正则分隔符
                header=None,                                                     # 数据区不再使用文件内表头
                names=header_names,                                              # 手动指定列名为注释表头
                comment="#",                                                     # 所有以 # 开头的行都忽略，包括单位行
            )                                                                   # 读取完成
        else:                                                                   # 若不是注释表头格式
            df = pd.read_csv(                                                    # 让 pandas 自动推断分隔符并读取
                file_path,                                                       # 输入文件路径
                sep=None,                                                        # 自动推断分隔符
                engine="python",                                                 # 使用 python 引擎支持推断
            )                                                                   # 读取完成

    elif suffix in {".xlsx", ".xls"}:                                           # 如果是 Excel 文件
        df = pd.read_excel(file_path, header=0)                                 # 默认按第一行列名读取 Excel
        df = maybe_drop_units_row(df)                                            # 如有单位行则尝试删除

    else:                                                                       # 如果文件类型不受支持
        raise ValueError(f"不支持的文件类型：{file_path}")                       # 抛出异常

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")                 # 删除全空行和全空列
    df.columns = clean_column_names(df.columns)                                  # 清洗列名
    df = df.apply(pd.to_numeric, errors="coerce")                               # 所有列强制尝试转成数值，失败则记为 NaN

    valid_numeric_columns = [column for column in df.columns if df[column].notna().any()]  # 保留至少含一个数值的列
    df = df.loc[:, valid_numeric_columns].copy()                                # 只保留有效数值列

    if df.empty:                                                                # 如果清洗后没有任何可用列
        raise ValueError(f"文件中没有可用于相关分析的数值列：{file_path}")       # 抛出异常

    if df.isna().any().any():                                                   # 如果仍然存在 NaN，说明有非数值脏数据混入
        bad_columns = df.columns[df.isna().any()].tolist()                      # 找出包含 NaN 的列名
        raise ValueError(                                                        # 抛出异常并说明具体问题列
            f"文件中存在无法转换为数值的内容，请检查这些列：{bad_columns}；文件：{file_path}"  # 错误信息
        )                                                                       # 异常结束

    return df                                                                   # 返回清洗好的纯数值表
# ========================= 3. 文件发现与读取函数结束 =====================     # 文件读取区结束


# ========================= 4. 数据一致性校验函数 =========================     # 一致性校验区开始
def validate_and_align_dataframes(named_dfs: List[Tuple[str, pd.DataFrame]]) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:  # 校验所有文件的列是否一致
    if not named_dfs:                                                           # 如果输入列表为空
        raise ValueError("没有可用于校验的数据框。")                              # 抛出异常

    base_name, base_df = named_dfs[0]                                           # 以第一个文件作为基准
    base_columns = list(base_df.columns)                                        # 记录基准列顺序

    aligned: List[Tuple[str, pd.DataFrame]] = []                                # 初始化对齐后的结果列表
    aligned.append((base_name, base_df.loc[:, base_columns].copy()))            # 先加入基准文件

    for file_name, df in named_dfs[1:]:                                         # 遍历其余文件
        current_columns = list(df.columns)                                      # 当前文件的列顺序
        if set(current_columns) != set(base_columns):                           # 如果列集合与基准不一致
            missing = [col for col in base_columns if col not in current_columns]  # 找出缺失列
            extra = [col for col in current_columns if col not in base_columns]    # 找出额外列
            raise ValueError(                                                    # 抛出异常说明哪些列不匹配
                f"文件列不一致：{file_name}；缺少列={missing}；多出列={extra}"      # 错误信息
            )                                                                   # 异常结束
        aligned.append((file_name, df.loc[:, base_columns].copy()))             # 统一按基准列顺序重排后加入结果

    return aligned, base_columns                                                # 返回对齐后的数据及统一列顺序
# ========================= 4. 数据一致性校验函数结束 =====================     # 一致性校验区结束


# ========================= 5. Spearman 相关计算函数 ======================     # 相关计算区开始
def compute_spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:                  # 计算单个文件的 Spearman 秩相关矩阵
    return df.corr(method="spearman")                                           # 直接调用 pandas 的 Spearman 相关实现


def build_pairwise_summary(matrix_by_file: Dict[str, pd.DataFrame], columns: List[str], decimals: int) -> pd.DataFrame:  # 生成所有参数对的 mean ± SD 汇总表
    rows: List[Dict[str, object]] = []                                          # 初始化结果行列表
    file_names = list(matrix_by_file.keys())                                    # 提取所有文件名，保持固定顺序

    for i in range(len(columns)):                                               # 第一层循环遍历第一个参数
        for j in range(i + 1, len(columns)):                                    # 第二层循环只取上三角，避免重复参数对
            param_1 = columns[i]                                                # 当前参数1
            param_2 = columns[j]                                                # 当前参数2
            values = [float(matrix_by_file[name].loc[param_1, param_2]) for name in file_names]  # 收集这对参数在各文件中的实现相关系数
            arr = np.asarray(values, dtype=float)                               # 转成 numpy 数组方便统计
            mean_value = float(np.mean(arr))                                    # 计算均值
            sd_value = float(np.std(arr, ddof=1)) if len(arr) >= 2 else float("nan")  # 至少两个文件时才计算样本标准差

            row: Dict[str, object] = {                                          # 构造当前参数对的一行结果
                "Parameter_1": param_1,                                         # 参数1名称
                "Parameter_2": param_2,                                         # 参数2名称
                "File_Count": len(arr),                                         # 参与统计的文件数量
                "Achieved_Spearman_rho_Mean": mean_value,                       # 实现秩相关系数均值
                "Achieved_Spearman_rho_SD": sd_value,                           # 实现秩相关系数标准差
                "Achieved_Spearman_rho_Mean_SD": format_mean_sd(mean_value, sd_value, decimals),  # 格式化后的“均值 ± 标准差”
            }                                                                   # 当前行主字段结束

            for index, file_name in enumerate(file_names, start=1):             # 额外把每个文件对应的单次实现值也存下来
                stem = Path(file_name).stem                                     # 提取文件名主干，便于生成列名
                row[f"rho_file_{index}_{stem}"] = float(matrix_by_file[file_name].loc[param_1, param_2])  # 写入单文件相关系数值

            rows.append(row)                                                    # 把当前参数对加入总结果

    pairwise_df = pd.DataFrame(rows)                                            # 转成 DataFrame
    pairwise_df = pairwise_df.sort_values(by=["Parameter_1", "Parameter_2"]).reset_index(drop=True)  # 排序并重建索引
    return pairwise_df                                                          # 返回参数对汇总表


def build_mean_sd_matrices(matrix_by_file: Dict[str, pd.DataFrame], columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:  # 生成均值矩阵和标准差矩阵
    stack = np.stack(                                                           # 把所有文件的相关矩阵堆叠成三维数组
        [matrix_by_file[file_name].loc[columns, columns].to_numpy(dtype=float) for file_name in matrix_by_file],  # 每个文件一个二维矩阵
        axis=0,                                                                 # 沿新轴堆叠
    )                                                                           # 堆叠完成

    mean_matrix = np.mean(stack, axis=0)                                        # 对文件维度求均值得到均值矩阵
    if stack.shape[0] >= 2:                                                     # 如果文件数至少为 2
        sd_matrix = np.std(stack, axis=0, ddof=1)                               # 计算样本标准差矩阵
    else:                                                                       # 如果只有 1 个文件
        sd_matrix = np.full((len(columns), len(columns)), np.nan, dtype=float)  # 标准差矩阵全部记为 NaN

    mean_df = pd.DataFrame(mean_matrix, index=columns, columns=columns)         # 均值矩阵转成 DataFrame
    sd_df = pd.DataFrame(sd_matrix, index=columns, columns=columns)             # 标准差矩阵转成 DataFrame
    return mean_df, sd_df                                                       # 返回两个矩阵表
# ========================= 5. Spearman 相关计算函数结束 ==================     # 相关计算区结束


# ========================= 6. 输出函数 ===================================     # 输出区开始
def save_outputs(                                                               # 保存所有输出结果到磁盘
    output_dir: str,                                                            # 输出目录路径
    matrix_by_file: Dict[str, pd.DataFrame],                                    # 各文件对应的 Spearman 相关矩阵
    pairwise_df: pd.DataFrame,                                                  # 所有参数对的汇总表
    mean_df: pd.DataFrame,                                                      # 均值矩阵
    sd_df: pd.DataFrame,                                                        # 标准差矩阵
    decimals: int,                                                              # 小数位数
) -> None:                                                                      # 无返回值
    output_path = Path(output_dir).expanduser().resolve()                       # 标准化输出目录路径
    output_path.mkdir(parents=True, exist_ok=True)                              # 自动创建输出目录
    per_file_dir = output_path / "per_file_matrices"                            # 每个文件相关矩阵的子目录
    per_file_dir.mkdir(parents=True, exist_ok=True)                             # 自动创建子目录

    for file_name, corr_df in matrix_by_file.items():                           # 遍历每个文件的相关矩阵
        stem = Path(file_name).stem                                             # 取文件名主干
        corr_df.round(decimals).to_csv(per_file_dir / f"{stem}_spearman_matrix.csv", encoding="utf-8-sig")  # 保存单文件 Spearman 相关矩阵

    pairwise_df.round(decimals).to_csv(output_path / "pairwise_achieved_rank_correlation_mean_sd.csv", index=False, encoding="utf-8-sig")  # 保存参数对汇总表
    mean_df.round(decimals).to_csv(output_path / "spearman_mean_matrix.csv", encoding="utf-8-sig")  # 保存均值矩阵
    sd_df.round(decimals).to_csv(output_path / "spearman_sd_matrix.csv", encoding="utf-8-sig")      # 保存标准差矩阵

    report_lines = [                                                            # 生成简单的文本报告
        "Achieved sample rank-correlation summary",                             # 报告标题
        f"Sample file count: {len(matrix_by_file)}",                            # 文件数量
        f"Parameter count: {len(mean_df.columns)}",                             # 参数数量
        f"Output directory: {output_path}",                                     # 输出目录
        "",                                                                     # 空行
        "Notes:",                                                               # 说明标题
        "1. One file is treated as one complete sampling realization.",         # 说明1
        "2. Mean and SD are computed across files.",                            # 说明2
        "3. If there is only one file, SD is reported as NA.",                  # 说明3
    ]                                                                           # 文本报告内容结束
    (output_path / "summary_report.txt").write_text("\n".join(report_lines), encoding="utf-8")  # 保存文本报告
# ========================= 6. 输出函数结束 ===============================     # 输出区结束


# ========================= 7. 主流程函数 =================================     # 主流程区开始
def main() -> None:                                                             # 主函数入口
    sample_files = discover_sample_files(SAMPLE_DIR, RECURSIVE, OUTPUT_DIR)     # 自动发现目录下全部样本文件，并排除输出目录

    named_dfs: List[Tuple[str, pd.DataFrame]] = []                              # 初始化“文件名 + 数据框”列表
    for file_path in sample_files:                                              # 逐个读取样本文件
        df = load_single_sample_file(file_path)                                 # 读取并清洗单个文件
        named_dfs.append((str(file_path), df))                                  # 保存到列表中

    aligned_dfs, columns = validate_and_align_dataframes(named_dfs)             # 校验所有文件列是否一致并统一列顺序

    matrix_by_file: Dict[str, pd.DataFrame] = {}                                # 初始化“文件 -> 相关矩阵”字典
    for file_name, df in aligned_dfs:                                           # 遍历全部对齐后的样本表
        matrix_by_file[file_name] = compute_spearman_matrix(df)                 # 计算并保存单文件 Spearman 相关矩阵

    pairwise_df = build_pairwise_summary(matrix_by_file, columns, PAIR_TABLE_DECIMALS)  # 生成所有参数对的 mean ± SD 汇总表
    mean_df, sd_df = build_mean_sd_matrices(matrix_by_file, columns)            # 生成均值矩阵和标准差矩阵

    save_outputs(                                                               # 把结果写出到磁盘
        output_dir=OUTPUT_DIR,                                                  # 输出目录
        matrix_by_file=matrix_by_file,                                          # 各文件相关矩阵
        pairwise_df=pairwise_df,                                                # 参数对汇总表
        mean_df=mean_df,                                                        # 均值矩阵
        sd_df=sd_df,                                                            # 标准差矩阵
        decimals=PAIR_TABLE_DECIMALS,                                           # 小数位数
    )                                                                           # 输出完成

    print("计算完成。")                                                          # 在控制台提示完成
    print(f"识别到的样本文件数量：{len(sample_files)}")                           # 输出文件数量
    print(f"参数数量：{len(columns)}")                                           # 输出参数数量
    print(f"输出目录：{Path(OUTPUT_DIR).expanduser().resolve()}")                # 输出目录
    print("")                                                                   # 空行
    print("前 10 行参数对统计结果预览：")                                         # 输出预览标题
    print(pairwise_df.head(10).to_string(index=False))                          # 打印前 10 行结果预览

    if len(sample_files) == 1:                                                  # 如果只有一个样本文件
        print("")                                                               # 空行
        print("[CAUTION] 当前目录下只有 1 个样本文件，因此只能得到单次实现的相关矩阵。")  # 警告1
        print("[CAUTION] 这时均值可以计算，但标准差 SD 没有统计意义，会显示为 NA。")      # 警告2
        print("[CAUTION] 若你要严格回应审稿人的 “mean ± SD”，应至少提供 2 个及以上独立样本文件。")  # 警告3


if __name__ == "__main__":                                                      # 脚本直接运行时执行主函数
    main()                                                                      # 调用主函数