def s_all(minors, minors_4):
    '''所有胚子都强化'''
    return True


def s_1c2a(minors, minors_4):
    '''只强化初始含双暴，或1暴2攻精充'''
    score = 0
    for m in minors:
        if m in ('cr', 'cd'):
            score += 1
        elif m in ('atkp', 'em', 'er'):
            score += 0.5
    return score >= 2


def s_1c1a(minors, minors_4):
    '''只强化初始含双暴，或1暴1攻精充'''
    score = 0
    for m in minors:
        if m in ('cr', 'cd'):
            score += 1
        elif m in ('atkp', 'em', 'er'):
            score += 0.25
    return score >= 1.25


def s_2e(minors, minors_4):
    '''只强化初始含精充的'''
    return 'em' in minors and 'er' in minors


def s_er(minors, minors_4):
    '''只强化初始含充能的'''
    return 'er' in minors


def s_3a(minors, minors_4):
    '''只强化初始3攻精充'''
    score = 0
    for m in minors:
        if m in ('atkp', 'em', 'er'):
            score += 1
    return score >= 3


def s_1a(minors, minors_4):
    '''只强化初始1攻精充'''
    score = 0
    for m in minors:
        if m in ('atkp', 'em', 'er'):
            score += 1
    return score >= 1


def s_2h1a(minors, minors_4):
    '''只强化初始含2生充1攻精'''
    score = 0
    for m in minors:
        if m in ('hpp', 'er'):
            score += 1
        elif m in ('atkp', 'em'):
            score += 0.5
    return score >= 2.5


def s_2d(minors, minors_4):
    '''只强化初始含防充的'''
    return 'defp' in minors and 'er' in minors


def s_1e1c(minors, minors_4):
    '''只强化初始含1充1暴'''
    score = 0
    for m in minors:
        if m == 'er':
            score += 1
        elif m in ('cr', 'cd'):
            score += 0.25
    return score >= 1.25


def s_2a(minors, minors_4):
    '''只强化初始含攻充的'''
    return 'atkp' in minors and 'er' in minors
