import matplotlib.pyplot as plt


def set_figure(fig_w=8., hw_ratio=1.0, font_size=10., tick_size=8., ms=7., lw=1.2):
    # print(plt.rcParams.keys())  # 很有用，查看所需属性
    # https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams
    # xlabel ylabel pad 无法在属性里设置 plt.xlabel("XX",labelpad=8.5)

    cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
    w = fig_w * cm_to_inc  # cm ==> inch
    h = w * hw_ratio
    plt.rcParams['figure.figsize'] = (w, h)  # 单位 inc
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = (14 * cm_to_inc, 6 * cm_to_inc)

    # 1. Times New Roman or Arial:
    plt.rc('font', family='Arial', weight='normal', size=float(font_size))
    # 2. Helvetica:
    # font = {'family': 'sans-serif', 'sans-serif': 'Helvetica',
    #         'weight': 'normal', 'size': float(font_size)}
    # plt.rc('font', **font)  # pass in the font dict as kwargs

    # pdf文字在illustrator中可编辑，实测可用
    # 参考：https://blog.csdn.net/weixin_44022515/article/details/120033378
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['pdf.fonttype'] = 42

    plt.rcParams['axes.linewidth'] = lw  # 图框宽度
    # plt.rcParams['axes.labelpad'] = 0.1  # 标签与坐标轴距离，default, 4

    # plt.rcParams['lines.markeredgecolor'] = 'k'
    plt.rcParams['lines.markeredgewidth'] = lw
    plt.rcParams['lines.markersize'] = ms

    # 刻度在内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['xtick.major.size'] = 2.0  # 刻度长度

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['ytick.major.size'] = 2.0

    plt.rcParams["legend.frameon"] = True  # 图框
    plt.rcParams["legend.framealpha"] = 0.9  # 不透明度
    plt.rcParams["legend.fancybox"] = False  # 圆形边缘
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams["legend.columnspacing"] = 1  # /font unit 以字体大小为单位
    plt.rcParams['legend.labelspacing'] = 0.2
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.borderpad"] = 0.3

    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.labelsize'] = tick_size+1
