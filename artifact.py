from data import P_MAIN, P_MINOR
from utils import convolve, convolve_avg, mix, to_cdf, from_cdf
from itertools import permutations
import numpy as np
from rich.progress import track


class Artifact:
    def __init__(self, slot=None, main=None, init_3=None, minors_4=None, minor_values=None, lv=0):
        '''如果不传参就随机初始化，如果传参不会检查参数'''
        # 部位
        self.slot = slot
        if self.slot is None:
            self.slot = np.random.choice(
                ['flower', 'plume', 'sands', 'goblet', 'circlet'])
        # 主词条
        self.main = main
        if self.main is None:
            mains = P_MAIN[self.slot]
            self.main = np.random.choice([m for m in mains], p=[
                                         mains[m] for m in mains])
        # 是否初始3词条
        self.init_3 = init_3
        if self.init_3 is None:
            self.init_3 = np.random.random() < 0.8
        # 副词条列表 (无论初始3词条还是4词条都是4个)
        self.minors_4 = minors_4
        if self.minors_4 is None:
            all_minors = [m for m in P_MINOR if m != self.main]
            p_all = 1 - P_MINOR[self.main] if self.main in P_MINOR else 1
            self.minors_4 = np.random.choice(
                all_minors, 4, replace=False, p=[P_MINOR[m]/p_all for m in all_minors])
        # 副词条词条数
        self.minor_values = minor_values
        if self.minor_values is None:
            self.minor_values = [0, 0, 0, 0]
            init_len = 3 if self.init_3 else 4
            for i in range(init_len):
                self.minor_values[i] = 7+np.random.choice(4)
        # 强化次数
        self.lv = 0
        for i in range(self.lv):
            self.level_up()

    def level_up(self):
        if self.init_3 and self.lv == 0:
            self.minor_values[-1] = 7+np.random.choice(4)
        else:
            i = np.random.choice(4)
            j = 7 + np.random.choice(4)
            self.minor_values[i] += j
        self.lv += 1

    def level_up_to_full(self):
        for i in range(5 - self.lv):
            self.level_up()

    def get_score(self, weight):
        '''返回的得分是一个浮点数'''
        s = 0
        for i in range(4):
            s += weight[self.minors_4[i]] * self.minor_values[i]
        return s

    def __str__(self):
        return \
            f'''Artifact <{hex(id(self))}>:
    Slot: {self.slot}
    Main: {self.main}
    Init 3: {self.init_3}
    LV: {self.lv}
    Minors:
        {self.minors_4[0]}: {self.minor_values[0]}
        {self.minors_4[1]}: {self.minor_values[1]}
        {self.minors_4[2]}: {self.minor_values[2]}
        {self.minors_4[3]}: {self.minor_values[3]}
'''


cache = {
    'weight': None,
    'd4': {},
    'd2': {},
    'd1': {},
    'mains': None,
    'accept': None,
    'single_results': {}
}


def single_artifact_distr(main, weight, accept, sim=False, **sim_options):
    result = {
        'pdf': [0],
        'minor_avg': {
            'hp': [],
            'atk': [],
            'def': [],
            'hpp': [],
            'atkp': [],
            'defp': [],
            'em': [],
            'er': [],
            'cr': [],
            'cd': [],
        },
        'p_level_up': 0
    }
    if sim:
        # 蒙特卡洛模拟
        sim_rounds = sim_options['sim_rounds'] if 'sim_rounds' in sim_options else 100000
        for i in track(range(sim_rounds), description='模拟中...'):
            artifact = Artifact(main=main)
            # 应用强化策略
            init_len = 3 if artifact.init_3 else 4
            if accept(artifact.minors_4[:init_len], artifact.minors_4):
                result['p_level_up'] += 1
                artifact.level_up_to_full()
                score = round(artifact.get_score(weight))
                # 更新pdf
                while score >= len(result['pdf']):
                    result['pdf'].append(0)
                result['pdf'][score] += 1
                # 更新副词条
                for m, v in zip(artifact.minors_4, artifact.minor_values):
                    while score >= len(result['minor_avg'][m]):
                        result['minor_avg'][m].append(0)
                    result['minor_avg'][m][score] += v
            else:
                # 不强化就相当于贡献0分
                result['pdf'][0] += 1
        # 归一化
        for i in range(len(result['pdf'])):
            if result['pdf'][i] > 0:
                for m in result['minor_avg']:
                    if i < len(result['minor_avg'][m]):
                        result['minor_avg'][m][i] /= result['pdf'][i]
                result['pdf'][i] /= sim_rounds
        result['p_level_up'] /= sim_rounds
        return result
    # 枚举所有初始4词条（共5040种）
    all_minors = [m for m in P_MINOR if m != main]
    p_all = 1 - P_MINOR[main] if main in P_MINOR else 1
    p_perm_3, p_perm_4 = {}, {}
    for perm in permutations(all_minors, 4):
        # 计算该排列的出现概率
        p, s = 1, p_all
        for m in perm:
            p *= P_MINOR[m] / s
            s -= P_MINOR[m]
        # 排序得到键
        perm_key = tuple(sorted(perm))
        # 应用强化策略
        if accept(perm[:3], perm):
            result['p_level_up'] += 0.8*p
            if perm_key in p_perm_3:
                p_perm_3[perm_key] += 0.8*p
            else:
                p_perm_3[perm_key] = 0.8*p
        else:
            result['pdf'][0] += 0.8*p
        if accept(perm, perm):
            result['p_level_up'] += 0.2*p
            if perm_key in p_perm_4:
                p_perm_4[perm_key] += 0.2*p
            else:
                p_perm_4[perm_key] = 0.2*p
        else:
            result['pdf'][0] += 0.2*p
    # 权重改变时必须清空cache
    if weight != cache['weight']:
        cache['weight'] = weight
        cache['d4'] = {}
        cache['d2'] = {}
        cache['d1'] = {}

    def get_d4(perm):
        '''这个函数计算初始4词条得分分布，及每个词条的条件期望'''
        if perm in cache['d4']:
            return cache['d4'][perm]
        result = {
            'pdf': np.zeros(41),
            'minor_avg': {}
        }
        for m in perm:
            result['minor_avg'][m] = np.zeros(41)
        # 枚举4词条的词条数，共4^4=256种
        for i in range(7, 11):
            s1 = weight[perm[0]] * i
            for j in range(7, 11):
                s2 = s1 + weight[perm[1]] * j
                for k in range(7, 11):
                    s3 = s2 + weight[perm[2]] * k
                    for l in range(7, 11):
                        s4 = round(s3 + weight[perm[3]] * l)
                        result['pdf'][s4] += 1
                        result['minor_avg'][perm[0]][s4] += i
                        result['minor_avg'][perm[1]][s4] += j
                        result['minor_avg'][perm[2]][s4] += k
                        result['minor_avg'][perm[3]][s4] += l
        # 归一化
        for i in range(41):
            if result['pdf'][i]:
                for m in perm:
                    result['minor_avg'][m][i] /= result['pdf'][i]
                result['pdf'][i] /= 256
        # 移除末尾的0，节省一点后续计算
        result['pdf'] = np.trim_zeros(result['pdf'], 'b')
        for m in perm:
            result['minor_avg'][m] = np.trim_zeros(result['minor_avg'][m], 'b')
        # 更新cache
        cache['d4'][perm] = result
        return result

    def get_d2(perm):
        '''这个函数计算升2级的得分分布，及每个词条的条件期望'''
        if perm in cache['d2']:
            return cache['d2'][perm]
        result = {
            'pdf': np.zeros(21),
            'minor_avg': {}
        }
        for m in perm:
            result['minor_avg'][m] = np.zeros(21)
        # 枚举2个被升级的词条以及对应的升级量，共4^4=256种
        for i in range(4):
            for j in range(4):
                for k in range(7, 11):
                    s1 = weight[perm[i]] * k
                    for l in range(7, 11):
                        s2 = round(s1 + weight[perm[j]] * l)
                        result['pdf'][s2] += 1
                        result['minor_avg'][perm[i]][s2] += k
                        result['minor_avg'][perm[j]][s2] += l
        # 归一化
        for i in range(21):
            if result['pdf'][i]:
                for m in perm:
                    result['minor_avg'][m][i] /= result['pdf'][i]
                result['pdf'][i] /= 256
        # 移除末尾的0，节省一点后续计算
        result['pdf'] = np.trim_zeros(result['pdf'], 'b')
        for m in perm:
            result['minor_avg'][m] = np.trim_zeros(result['minor_avg'][m], 'b')
        # 更新cache
        cache['d2'][perm] = result
        return result

    def get_d1(perm):
        '''这个函数计算升1级的得分分布，及每个词条的条件期望'''
        if perm in cache['d1']:
            return cache['d1'][perm]
        result = {
            'pdf': np.zeros(11),
            'minor_avg': {}
        }
        for m in perm:
            result['minor_avg'][m] = np.zeros(11)
        # 枚举被升级的词条以及对应的升级量，共4^2=16种
        for i in range(4):
            for j in range(7, 11):
                s1 = round(weight[perm[i]] * j)
                result['pdf'][s1] += 1
                result['minor_avg'][perm[i]][s1] += j
        # 归一化
        for i in range(11):
            if result['pdf'][i]:
                for m in perm:
                    result['minor_avg'][m][i] /= result['pdf'][i]
                result['pdf'][i] /= 16
        # 移除末尾的0，节省一点后续计算
        result['pdf'] = np.trim_zeros(result['pdf'], 'b')
        for m in perm:
            result['minor_avg'][m] = np.trim_zeros(result['minor_avg'][m], 'b')
        # 更新cache
        cache['d1'][perm] = result
        return result

    # 分别计算初始3、4词条时的得分分布与各副词条分布
    for perm in p_perm_3:
        p = p_perm_3[perm]
        d4 = get_d4(perm)
        d2 = get_d2(perm)
        score_pdf_of_perm = convolve(d4['pdf'], d2['pdf'], d2['pdf'])
        result['pdf'] = mix(result['pdf'], p * np.array(score_pdf_of_perm))
        for m in perm:
            # avg_of_m := E[value of m | perm, score] * Pr[score | perm]
            avg_of_m = convolve_avg([d4['pdf'], d2['pdf'], d2['pdf']], [
                                    d4['minor_avg'][m], d2['minor_avg'][m], d2['minor_avg'][m]])
            result['minor_avg'][m] = mix(
                result['minor_avg'][m], p * np.array(avg_of_m))

    for perm in p_perm_4:
        p = p_perm_4[perm]
        d4 = get_d4(perm)
        d2 = get_d2(perm)
        d1 = get_d1(perm)
        score_pdf_of_perm = convolve(
            d4['pdf'], d2['pdf'], d2['pdf'], d1['pdf'])
        result['pdf'] = mix(result['pdf'], p * np.array(score_pdf_of_perm))
        for m in perm:
            # avg_of_m := E[value of m | perm, score] * Pr[score | perm]
            avg_of_m = convolve_avg([d4['pdf'], d2['pdf'], d2['pdf'], d1['pdf']], [
                                    d4['minor_avg'][m], d2['minor_avg'][m], d2['minor_avg'][m], d1['minor_avg'][m]])
            result['minor_avg'][m] = mix(
                result['minor_avg'][m], p * np.array(avg_of_m))

    # 归一化，因为 E[value of m | score] = \sum_{perm} E[value of m | perm, score] * Pr[score | perm] / Pr[score]
    for i in range(len(result['pdf'])):
        if result['pdf'][i] > 0:
            for m in P_MINOR:
                if len(result['minor_avg'][m]) > i:
                    result['minor_avg'][m][i] /= result['pdf'][i]

    return result


def artifact_tuple_distr(mains, weight, accept, count, sim=False, **sim_options):
    result = {
        'pdf': [],
        'minor_avg': {
            'hp': [],
            'atk': [],
            'def': [],
            'hpp': [],
            'atkp': [],
            'defp': [],
            'em': [],
            'er': [],
            'cr': [],
            'cd': [],
        },
        'p_level_up': 0,
    }
    if sim:
        # 蒙特卡洛模拟
        sim_rounds = sim_options['sim_rounds'] if 'sim_rounds' in sim_options else 100000
        for i in track(range(sim_rounds), description='模拟中...'):
            # 初始化背包
            bag = {}
            for slot in mains:
                bag[slot] = {
                    'score': 0,
                    'artifact': None
                }
            n_level_up = 0
            # 逐个更新背包
            for j in range(count):
                artifact = Artifact()
                # 如果主词条不对直接舍弃
                if artifact.main != mains[artifact.slot]:
                    continue
                # 应用强化策略
                init_len = 3 if artifact.init_3 else 4
                if not accept(artifact.slot, artifact.minors_4[:init_len], artifact.minors_4):
                    continue
                # 强化到满级
                n_level_up += 1
                artifact.level_up_to_full()
                score = artifact.get_score(weight)
                # 如果得分够高就装备到背包
                if score > bag[artifact.slot]['score']:
                    bag[artifact.slot]['score'] = score
                    bag[artifact.slot]['artifact'] = artifact
            # 更新result pdf
            tot_score = 0
            for slot in bag:
                tot_score += bag[slot]['score']
            tot_score = round(tot_score)
            while tot_score >= len(result['pdf']):
                result['pdf'].append(0)
            result['pdf'][tot_score] += 1
            # 更新副词条
            tot_minors = {
                'hp': 0,
                'atk': 0,
                'def': 0,
                'hpp': 0,
                'atkp': 0,
                'defp': 0,
                'em': 0,
                'er': 0,
                'cr': 0,
                'cd': 0,
            }
            for slot in bag:
                a = bag[slot]['artifact']
                if a is None:
                    continue
                for m, v in zip(a.minors_4, a.minor_values):
                    tot_minors[m] += v
            for m in tot_minors:
                while tot_score >= len(result['minor_avg'][m]):
                    result['minor_avg'][m].append(0)
                result['minor_avg'][m][tot_score] += tot_minors[m]
            # 更新强化概率
            result['p_level_up'] += n_level_up / count
        # 归一化
        for i in range(len(result['pdf'])):
            if result['pdf'][i] > 0:
                for m in result['minor_avg']:
                    if i < len(result['minor_avg'][m]):
                        result['minor_avg'][m][i] /= result['pdf'][i]
                result['pdf'][i] /= sim_rounds
        result['p_level_up'] /= sim_rounds
        return result
    # 必要时清空cache
    if weight != cache['weight'] or mains != cache['mains'] or accept != cache['accept']:
        cache['weight'] = weight
        cache['mains'] = mains
        cache['accept'] = accept
        cache['d4'] = {}
        cache['d2'] = {}
        cache['d1'] = {}
        cache['single_results'] = {}
    # 计算5个部位的圣遗物分布数据
    single_results = {}
    for slot in mains:
        if slot in cache['single_results']:
            d = cache['single_results'][slot]
        else:
            d = single_artifact_distr(
                mains[slot], weight, lambda minors, minors_4: accept(slot, minors, minors_4))
            cache['single_results'][slot] = d
        # 更新刷了count件后5个部位圣遗物得分分布
        p = P_MAIN[slot][mains[slot]]/5
        cdf = to_cdf(d['pdf'])
        cdf = list(map(lambda x: (p*x+1-p)**count, cdf))
        single_results[slot] = {
            'pdf': from_cdf(cdf),
            'minor_avg': d['minor_avg']
        }
        # 更新强化概率
        result['p_level_up'] += d['p_level_up'] * p
    # 以下假设5个部位独立，这一假设会带来少量误差
    result['pdf'] = convolve(*[single_results[slot]['pdf']
                             for slot in single_results])
    for m in P_MINOR:
        avg_of_m = convolve_avg([single_results[slot]['pdf']
                                 for slot in single_results],
                                [single_results[slot]['minor_avg'][m]
                                 for slot in single_results])
        # 归一化
        for i in range(len(avg_of_m)):
            if result['pdf'][i] > 0:
                avg_of_m[i] /= result['pdf'][i]
        # 存储计算结果
        result['minor_avg'][m] = avg_of_m
    return result
