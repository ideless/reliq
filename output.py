from math import ceil
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from utils import to_cdf
import numpy as np
import matplotlib.patheffects as pe
from datetime import datetime
from artifact import artifact_tuple_distr


def _get_stamp():
    now = datetime.now()
    return now.strftime('%m%d%H%M')


preset = None
accept = None
mains = None
weight = None


def import_preset(preset_):
    global preset, accept, mains, weight
    preset = preset_
    def accept(slot, minors, minors_4): return preset['strategy'][slot](
        minors, minors_4)
    mains = preset_['mains']
    weight = preset_['weight']


def print_meta():
    print(preset['name'])
    print('主词条:', preset['mains_description'])
    print('计分权重:', preset['weight_description'])
    print('强化策略:', [preset['strategy_description']])
    print('有效词条:', preset['format_name'])
    print()


gold_per_run = 1.065    # 20体期望金数
purple_per_run = 2.485  # 20体期望紫数
blue_per_run = 3.55     # 20体期望蓝数
exp_gold = 3780         # 0级金折算经验
exp_purple = 2520       # 0级紫折算经验
exp_blue = 1260         # 0级蓝折算经验
exp_gold_20 = 270475    # 强化到20级所需经验
p_x2 = 0.09             # 强化经验x2概率
p_x5 = 0.01             # 强化经验x5概率

# 可以完美循环的强化比例
exp_per_run = gold_per_run * exp_gold + \
    purple_per_run * exp_purple + blue_per_run * exp_blue
exp_per_gold = exp_per_run / gold_per_run
exp_gold_20_avg = exp_gold_20 / (1 + p_x2 + 4 * p_x5)
p_level_up_recom = exp_per_gold / (exp_per_gold+exp_gold_20_avg)


def print_p_level_up(result):
    def format_percentage(p):
        return str(round(p*100, 1))+'%'
    print('强化概率:', format_percentage(
        result['p_level_up']), '推荐值', format_percentage(p_level_up_recom))
    print()


def debug(count=270, sim=False, sim_rounds=10000):
    result = artifact_tuple_distr(
        mains, weight, accept, count, sim, sim_rounds=sim_rounds)
    # 检查浮点数错误
    errs = []
    for i in range(len(result['pdf'])):
        if result['pdf'][i] == 0:
            continue
        s = 0
        for m in weight:
            if len(result['minor_avg'][m]) > i:
                s += weight[m] * result['minor_avg'][m][i]
        errs.append(s - i)

    errs_abs = np.abs(np.array(errs))
    print('最大错误', np.max(errs_abs))
    print('平均错误', np.mean(errs_abs))
    print()

    fig, axs = plt.subplots(2)
    axs[0].plot(result['pdf'])
    for m in weight:
        axs[1].plot(result['minor_avg'][m], label=m)
    axs[1].legend()
    fig.savefig('debug.png', dpi=300)

    print_p_level_up(result)


days = [7, 14, 21, 28, 35, 42, 49, 56, 90, 180, 360, 720, 1800, 3600]
day_labels = ['1周', '2周', '3周', '4周', '5周', '6周',
              '7周', '8周', '3月', '6月', '1年', '2年', '5年', '10年']


def print_evolve_table():
    format = preset['format']
    # 表头
    print(f'天数\t{preset["format_name"]}', end='\t')
    for m in format:
        print(format[m]['name'], end='\t')
    print()
    # 逐行打印
    for day, day_label in zip(days, day_labels):
        count = round(day*9*gold_per_run)
        # , sim=True, sim_rounds=round(500000/count))
        result = artifact_tuple_distr(mains, weight, accept, count)
        # print(result['p_level_up'])
        print(day_label, end='\t')
        all, others = 0, []
        for m in format:
            s = 0
            for i in range(len(result['minor_avg'][m])):
                s += result['pdf'][i] * result['minor_avg'][m][i]
            # 约化成标准词条数
            s /= 8.5
            others.append(s)
            all += format[m]['weight'] * s
        print(round(all, 1), end='\t')
        for x in others:
            print(round(x, 1), end='\t')
        print()
    # 最后留一个空白行
    print()


text_font = FontProperties(fname=r"./fonts/NotoSansSC-Regular.otf", size=10)
title_font = FontProperties(fname=r"./fonts/NotoSansSC-Bold.otf", size=10)
suptitle_font = FontProperties(fname=r"./fonts/NotoSansSC-Bold.otf", size=15)
mark_font = FontProperties(fname=r"./fonts/NotoSansSC-Bold.otf", size=10)

font_stroke_width = 1.5


def plot_30days_minor_avgs():
    result = artifact_tuple_distr(
        mains, weight, accept, round(30*9*gold_per_run))
    title = preset['name']+'30天副词条分布'
    description = '主词条：'+preset['mains_description']+'\n' + \
        '计分权重：'+preset['weight_description'] + \
        '\n'+preset['strategy_description']
    format = preset['format']
    y_max = np.max([np.max(result['minor_avg'][m])/8.5 for m in format])
    figsize = (5.2, ceil(y_max/2))

    fig = plt.figure(dpi=100, figsize=figsize)
    ax = fig.gca()

    x_tick = 0.1
    y_tick = 1

    ax.set_xticks(np.arange(0, 1.01, x_tick))
    ax.set_yticks(np.arange(0, y_max + y_tick/2, y_tick))

    plt.xlim(-0.05, 1.05)
    plt.ylim(-y_tick/2, y_max+y_tick/2)

    plt.grid(visible=True, which='major',
             color='lightgray', linestyle='-', linewidth=1)
    plt.grid(visible=True, which='minor', color='lightgray',
             linestyle='-', linewidth=0.5)
    plt.minorticks_on()

    attention_pos = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    attention_lineheight = [0, 0, 0, 0, 0, 0]

    cdf = to_cdf(result['pdf'])
    for m in format:
        linestyle = '-'  # if m in target_minors else '--'
        y = np.zeros(len(cdf))
        y_ = result['minor_avg'][m]/8.5
        y[:len(y_)] = y_
        plt.plot(cdf, y,
                 linestyle, linewidth=2, c=format[m]['linecolor'])
        # 绘制标注点
        pos_i = 0
        for p, a in zip(cdf, result['minor_avg'][m]):
            pos = attention_pos[pos_i]
            if p > pos:
                attention_lineheight[pos_i] = max(
                    attention_lineheight[pos_i], a/8.5)
                plt.text(pos-0.07, a/8.5, str(round(a/8.5, 1)), fontproperties=text_font,
                         path_effects=[pe.withStroke(linewidth=font_stroke_width, foreground="white")])
                if pos == 0.5:
                    plt.text(pos+0.01, a/8.5, format[m]['name'], c=format[m]['linecolor'], fontproperties=mark_font,
                             path_effects=[pe.withStroke(linewidth=font_stroke_width, foreground="white")])
                pos_i += 1
            if pos_i >= len(attention_pos):
                break

    # 绘制标注线
    for pos, h in zip(attention_pos, attention_lineheight):
        plt.plot([pos, pos], [-y_tick/2, h+y_tick*2],
                 ':', c='gray', linewidth=1.5)
        plt.text(pos-0.08, h+y_tick*1.7, str(int(pos*100))+"%",
                 c='gray', fontproperties=mark_font, path_effects=[pe.withStroke(linewidth=font_stroke_width, foreground="white")])

    plt.title(title, fontproperties=title_font)
    plt.xlabel('概率', fontproperties=text_font)
    plt.ylabel('词条数', fontproperties=text_font)
    plt.text(0.3, y_max, description, c='#B0B0B0',
             fontproperties=mark_font,
             path_effects=[pe.withStroke(
                 linewidth=font_stroke_width, foreground="white")],
             horizontalalignment='center',
             verticalalignment='top')
    # plt.show()
    plt.savefig(f'output/dumps/{_get_stamp()}.png')


def print_30days_allpresets(presets):
    head_minors = ['cr', 'cd', 'em', 'er', 'atkp', 'atk'
                   'defp', 'def', 'hpp', 'hp']
    labels = {
        'hp': '小生命',
        'atk': '小攻击',
        'def': '小防御',
        'hpp': '大生命',
        'atkp': '大攻击',
        'defp': '大防御',
        'em': '精通',
        'er': '充能',
        'cr': '暴击',
        'cd': '爆伤',
    }
    # 表头
    print('配装\t有效词条数', end='\t')
    for m in head_minors:
        print(labels[m], end='\t')
    print()
    # 逐行打印
    count = round(30*9*gold_per_run)
    for pkey in presets:
        preset = presets[pkey]
        format = preset['format']
        weight = preset['weight']
        mains = preset['mains']

        def accept(slot, minors, minors_4): return preset['strategy'][slot](
            minors, minors_4)
        result = artifact_tuple_distr(mains, weight, accept, count)
        all, others = 0, []
        for m in head_minors:
            s = 0
            for i in range(len(result['minor_avg'][m])):
                s += result['pdf'][i] * result['minor_avg'][m][i]
            # 约化成标准词条数
            s /= 8.5
            others.append(s)
            if m in format:
                all += format[m]['weight'] * s
        print(preset['name'], end='\t')
        print(round(all, 1), end='\t')
        for x in others:
            print(round(x, 1), end='\t')
        print()
    print()
