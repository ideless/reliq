from audioop import avg
from data import P_MAIN, P_MINOR
from utils import convolve, convolve_avg, mix, to_cdf, from_cdf
from itertools import permutations
import numpy as np

cache = {
    'weight': None,
    'd4': {},
    'd2': {},
    'd1': {},
}


def single_artifact_distr(main, weight, accept):
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
        }
    }
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
            if perm_key in p_perm_3:
                p_perm_3[perm_key] += p
            else:
                p_perm_3[perm_key] = p
        else:
            result['pdf'][0] += p
        if accept(perm, perm):
            if perm_key in p_perm_4:
                p_perm_4[perm_key] += p
            else:
                p_perm_4[perm_key] = p
        else:
            result['pdf'][0] += p
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
        p = p_perm_3[perm] * 0.8
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
        p = p_perm_4[perm] * 0.2
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


def arifact_tuple_distr(mains, weight, accept, count):
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
        }
    }
    # 计算5个部位的单圣遗物分布数据
    single_results = {}
    for slot in mains:
        single_results[slot] = single_artifact_distr(
            mains[slot], weight, lambda minors, minors_4: accept(slot, minors, minors_4))
        # 更新刷了count件后5个部位圣遗物得分分布
        p = P_MAIN[slot][mains[slot]]
        cdf = to_cdf(single_results[slot]['pdf'])
        cdf = list(map(lambda x: (p*x+1-p)**count, cdf))
        single_results[slot]['pdf'] = from_cdf(cdf)
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
