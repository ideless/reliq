from output import *
from presets import presets
from artifact import artifact_tuple_distr

# print_30days_allpresets(presets)
# exit()

preset = presets['test']

import_preset(preset)
print_meta()
debug()
print_evolve_table()
plot_30days_minor_avgs()
exit()

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


def accept(slot, minors, minors_4):
    return True


result = artifact_tuple_distr(mains, weight, accept, 500)

print(result)
