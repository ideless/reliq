from artifact import single_artifact_distr
import numpy as np
import matplotlib.pyplot as plt

main = 'hp'
weight = {
    'hp': 0,
    'atk': 0,
    'def': 0,
    'hpp': 0,
    'atkp': 1,
    'defp': 0,
    'em': 0,
    'er': 0,
    'cr': 1,
    'cd': 1,
}


def accept(minors, minors_4):
    return True


result = single_artifact_distr(main, weight, accept)

# 检查错误
errs = []
for i in range(len(result['pdf'])):
    s = 0
    for m in weight:
        if len(result['minor_avg'][m]) > i:
            s += weight[m] * result['minor_avg'][m][i]
    errs.append(s - i)

print(result)

plt.plot(result['pdf'], label='pdf')
for m in weight:
    plt.plot(result['minor_avg'][m], label=m)
plt.legend()
plt.savefig('out.png', dpi=300)

errs_abs = np.abs(np.array(errs))
print(np.max(errs_abs))
print(np.mean(errs_abs))
