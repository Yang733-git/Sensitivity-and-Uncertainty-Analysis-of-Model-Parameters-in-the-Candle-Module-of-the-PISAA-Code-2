import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import matplotlib.lines as mlines
#根据散点数据绘制散点图


# 字体
#matplotlib.rcParams['font.family'] = 'serif'
#matplotlib.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = 'Times New Roman'  # 指定衬线字体为Times New Roman
#plt.rcParams['mathtext.fontset'] = 'stix'     # 数学公式字体（与Times兼容）

output_image_path = r"C:\Users\11201\Desktop\sensities\morris\results\散点图.png"
file_path = r'C:/Users/11201/Desktop/sensities/morris/散点图绘图数据.txt'
data_dict = {}

# 读取&解析
with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f]

current_material = None
i = 0
while i < len(lines):
    line = lines[i]
    if line.startswith('---') and '材料输出---' in line:
        material = line.strip('-').replace('材料输出', '')
        if material == '包壳':
            material = 'Cladding'
        elif material == '控制棒':
            material = 'Control rod'
        data_dict[material] = {}
        current_material = material
        i += 1
        continue

    if current_material and 'Morris分析结果' in line:
        output_var = line.split()[0]
        i += 1
        while i < len(lines) and 'mu_star' not in lines[i]:
            i += 1
        i += 1
        data_lines = []
        while (i < len(lines) and lines[i] and not lines[i].startswith('---')
               and 'Morris分析结果' not in lines[i]):
            data_lines.append(lines[i])
            i += 1

        df = pd.read_csv(io.StringIO('\n'.join(data_lines)),
                         sep=r'\s+', names=['parameter', 'mu_star', 'sigma'],
                         header=None)
        df.set_index('parameter', inplace=True)
        df = df.drop(index='parameter', errors='ignore')
        data_dict[current_material][output_var] = df
        continue
    i += 1

# 统一配色/图例
all_params = sorted({p for mat in data_dict.values() for df in mat.values() for p in df.index})
# 自动设置颜色
#cmap = plt.get_cmap('Accent', len(all_params))
#color_dict = {p: cmap(k) for k, p in enumerate(all_params)}

# 手动设置颜色（这里选择了几种鲜艳的颜色，可以根据需要调整）
color_list = [
    'red', 
    'orange', 
    'yellow', 
    'green', 
    'blue', 
    'purple', 
    'cyan', # K
    'magenta',  # L
    'pink', 
    'mediumturquoise', # MT
    'chartreuse', 
    'brown', # RHO
    'darkorchid', 
    'yellowgreen', 
    'gray', 
    'black'
]
# 确保 color_list 长度足够
if len(all_params) > len(color_list):
    print("警告：颜色列表中的颜色数量不足，考虑添加更多颜色")
# 手动为每个参数分配颜色
color_dict = {p: color_list[k % len(color_list)] for k, p in enumerate(all_params)}


fig, axes = plt.subplots(3, 3, figsize=(12, 7))  # 不用 constrained_layout

materials = list(data_dict.keys())
outputs = ['mLeav', 'mFrozen', 'mAccu']

row_title_map = {
    'UO2': r'燃料芯块',
    'Cladding': '燃料包壳',
    'Control rod': '控制棒',
}

col_title_map = {
    'mLeav': 'FOM1',
    'mFrozen': 'FOM2',
    'mAccu': 'FOM3',
}

for r, material in enumerate(materials):
    for c, output_var in enumerate(outputs):
        ax = axes[r, c]
        df = data_dict[material][output_var]

        for param, vals in df.iterrows():
            ax.scatter(vals['mu_star'], vals['sigma'],
                       label=param, color=color_dict[param], s=20, alpha=0.8, marker='x')

        # 标注前5个
        #for param in df['mu_star'].nlargest(5).index:
        #    x, y = df.loc[param, ['mu_star', 'sigma']]
        #    ax.annotate(param, xy=(x, y), xytext=(5, 5),
        #                textcoords='offset points', fontsize=8)

        ax.set_xlabel(r'$\mu^*$', fontsize=9)
        ax.set_ylabel(r'$\sigma$', fontsize=9)

        idx = r * 3 + c
        ax.text(0.05, 0.95, f'({chr(ord("a")+idx)})',
                transform=ax.transAxes, ha='left', va='top', fontsize=12)

# ——合并图例（右侧）——
handles = [mlines.Line2D([], [], linestyle='', marker='x', markersize=8, markeredgewidth=2,
                          markerfacecolor=color_dict[p], markeredgecolor=color_dict[p])
           for p in all_params]

# 预留右侧 20% 空间，并把子图间距压紧：
#  - rect：控制子图占用画布范围（右侧留白给图例；底部稍留给子标题）
#  - pad/w_pad/h_pad：全局、列间、行间的留白（可再调小/大）
plt.tight_layout(rect=[0.12, 0.02, 0.90, 0.90], pad=0.0, w_pad=0.2, h_pad=0.0)

for c, output_var in enumerate(outputs):
    pos = axes[0, c].get_position()
    x = pos.x0 + pos.width / 2
    y = min(pos.y1 + 0.02, 0.99)
    fig.text(x, y, col_title_map.get(output_var, output_var), ha='center', va='bottom', fontsize=16)

for r, material in enumerate(materials):
    pos = axes[r, 0].get_position()
    x = max(pos.x0 - 0.04, 0.01)
    y = pos.y0 + pos.height / 2
    fig.text(x, y, row_title_map.get(material, material), ha='right', va='center', fontsize=16)

fig.legend(handles, ['TH' if p == 'SIG' else p for p in all_params], title='输入参数',
           loc='center left', bbox_to_anchor=(0.915, 0.5),
           borderaxespad=0.0, fontsize=12)

# 保存图像
plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
print(f"结果已保存至: {output_image_path}")

plt.show()
