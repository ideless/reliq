from artifact import arifact_tuple_distr
import numpy as np
import matplotlib.pyplot as plt

mains = {
    'flower': 'hp',
    'plume': 'atk',
    'sands': 'atkp',
    'goblet': 'pyroDB',
    'circlet': 'cr',
}

weight = {
    'hp': 0,
    'atk': 0,
    'def': 0,
    'hpp': 0,
    'atkp': 0.5,
    'defp': 0,
    'em': 0.25,
    'er': 0.25,
    'cr': 1,
    'cd': 1,
}

good_minors = ['cr', 'cd', 'atkp', 'em', 'er']


def accept(slot, minors, minors_4):
    return True


result = arifact_tuple_distr(mains, weight, accept, 270)

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


fig, axs = plt.subplots(2)
axs[0].plot(result['pdf'])
for m in weight:
    axs[1].plot(result['minor_avg'][m], label=m)
axs[1].legend()
fig.savefig('out.png', dpi=300)


# 导出结论
gold_per_run = 1.05

# 副词条成长表
days = [7, 14, 21, 28, 35, 42, 49, 56, 90, 180, 360]
# 表头
print('day', end='\t')
for m in good_minors:
    print(m, end='\t')
print('')
# 逐行打印
for day in days:
    print(day, end='\t')
    result = arifact_tuple_distr(
        mains, weight, accept, round(day*9*gold_per_run))
    for m in good_minors:
        s = 0
        for i in range(len(result['minor_avg'][m])):
            s += result['pdf'][i] * result['minor_avg'][m][i]
        # 约化成标准词条数
        s = round(s / 8.5, 1)
        print(s, end='\t')
    print('')
