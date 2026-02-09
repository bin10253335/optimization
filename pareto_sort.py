import numpy as np
from coding import decoding
from problem import Job, visualization

#from GA import *
#from GA import Solution
import random
import copy
class Solution:
    def __init__(self, s) -> None:
        self.s = s
        self.fit = 0
        self.energy = 0
        self.Due = 0
        self.Ear = 0
        self.DE = 0
class Pareto(object):
    def __init__(self, pareto_rank):
        self.pareto_rank = pareto_rank
        self.ss = []

class PP(object):
    def __init__(self, id):
        self.id = id
        self.n = 0
        self.s = []

def non_dominate(p: list, f_num: int):
    fun_obj = np.zeros((len(p), f_num+5))
    for i in range(len(p)):
        for j in range(f_num):
            if j == 0:
                fun_obj[i][j] = p[i].fit
            elif j == 1:
                fun_obj[i][j] = p[i].energy
            else:
                fun_obj[i][j] = p[i].DE

    pareto_rank = 0
    pareto_ranks = []
    all_p = []
    F = Pareto(pareto_rank)
    for i in range(len(p)):
        pp = PP(i)
        for j in range(len(p)):
            less = 0
            equal = 0
            greater = 0
            for k in range(f_num):
                if fun_obj[i][k] < fun_obj[j][k]:
                    less += 1
                elif fun_obj[i][k] > fun_obj[j][k]:
                    greater += 1
                else:
                    equal += 1
            if less == 0 and equal != f_num:
                pp.n += 1
            elif greater == 0 and equal != f_num:
                pp.s.append(j)
        all_p.append(pp)
        if all_p[i].n == 0:
            F.ss.append(i)
            fun_obj[i][3] = 0
    pareto_ranks.append(F)
    while len(pareto_ranks[pareto_rank].ss) > 0:
        temp = []
        for j in range(len(pareto_ranks[pareto_rank].ss)):
            if len(all_p[pareto_ranks[pareto_rank].ss[j]].s) != 0:
                for k in range(len(all_p[pareto_ranks[pareto_rank].ss[j]].s)):
                    all_p[all_p[pareto_ranks[pareto_rank].ss[j]].s[k]].n -= 1
                    if all_p[all_p[pareto_ranks[pareto_rank].ss[j]].s[k]].n == 0:
                        fun_obj[all_p[pareto_ranks[pareto_rank].ss[j]].s[k]][3] = pareto_rank + 1
                        temp.append(all_p[pareto_ranks[pareto_rank].ss[j]].s[k])
        if len(temp) == 0:
            break
        else:
            pareto_rank += 1
            F = Pareto(pareto_rank)
            F.ss = copy.copy(temp)
            pareto_ranks.append(F)
    return pareto_ranks, fun_obj


def pareto_cross(population: list, fun_obj: np.array, Jobs: list, D: list, E: list, machine_num: int):
    offspring = []
    off_obj = np.zeros((1, fun_obj.shape[1]))
    pareto_set = []
    pareto_zero = []

    if len(population) == 2:
        for i in range(len(population)):
            chishu = 0
            gongxu_i = population[i][0].s[1]
            while chishu <= 10:
                a = np.random.randint(0, len(population[i][0].s[1]))  # 随机生成0到gx1_index之间的整数
                b = np.random.randint(0, len(population[i][0].s[1]))

                while b == a:
                    b = np.random.randint(0, len(fun_obj[i]))  # 重复生成随机数字，直到a不等于b
                if a < b:
                    jiaohuan = gongxu_i[a:b]
                    jiaohuan = jiaohuan[::-1]
                    gongxu_i[a:b] = jiaohuan

                    new_individual = Solution([population[i][0].s[0], gongxu_i])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:

                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                        chishu += 1
                    else:
                        chishu += 1
                else:
                    jiaohuan = gongxu_i[b:a]
                    jiaohuan = jiaohuan[::-1]
                    gongxu_i[b:a] = jiaohuan
                    new_individual = Solution([population[i][0].s[0], gongxu_i])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                        chishu += 1
                    else:
                        chishu += 1
        if len(offspring) != 0:
            off_obj = np.delete(off_obj, 0, 0)
        return population, offspring, off_obj
    else:
        for i in range(len(population)-1):
            length = np.zeros((1, 3))
            for j in range(i+1, len(population)):
                temp = [a - b for a, b in zip(population[i][0].s[1], population[j][0].s[1])]
                temp = np.array(temp)
                indexs = np.where(temp == 0)[0].tolist()
                if len(indexs) < 3:
                    continue
                else:
                    record_matrix = np.zeros((1,2))
                    t = 0
                    while t < len(indexs)-1:
                        record = 1
                        for l in np.arange(t+1, len(indexs)):
                            if indexs[l] - indexs[l-1] == 1 and (t+1) != len(indexs):
                                record += 1
                                t = l
                            else:
                                record_matrix = np.append(record_matrix, [[t, record]], axis=0)
                                t = l  # 跳过中间的无效循环，提高效率
                                break
                    record_matrix = np.delete(record_matrix, 0, axis = 0)
                    if len(record_matrix) != 0:
                        max_t = record_matrix[np.argmax(record_matrix, axis=0)[1], 0]
                        length = np.append(length, [[j, max_t, np.max(record_matrix, axis=0)[1]]], axis=0)
                    else:
                        max_t = np.random.randint(len(population[j][0].s[1])-5, len(population[j][0].s[1])-3)
                        length = np.append(length, [[j, max_t, 2]], axis=0)

            if len(length) > 1:
                length = np.delete(length, 0, axis=0)
            if length.ndim == 1:
                length = np.expand_dims(length, axis=0)
            if np.argmax(length, axis=0)[2] != 0:
                index = np.argmax(length, axis=0)[2]
            else:
                index = np.random.randint(0, len(length))
            Temp = length[index]
            gongxu1 = population[i][0].s[1]
            gongxu1_copy = gongxu1
            gongxu2 = population[int(Temp[0])][0].s[1]
            gongxu2_copy = gongxu2  # 记录副本
            gx1_index = int(Temp[1])
            print("gx1_index:", gx1_index)
            if gx1_index > 1:
                print("gx1_index > 1")
                if gx1_index == 2:
                    f = gongxu1[0]
                    gongxu1[0] = gongxu1[1]
                    gongxu1[1] = f
                    new_individual = Solution([population[i][0].s[0], gongxu1])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                    else:
                        continue
                elif gx1_index > 2:
                    chishu = 0
                    off = population[i]
                    obj = np.array([population[i][0].fit, population[i][0].energy, population[i][0].DE, fun_obj[i][3]])
                    flag = 0
                    while chishu <= 10:
                        a = np.random.randint(0, gx1_index-1)
                        b = np.random.randint(0, gx1_index-1)
                        while b == a:
                            b = np.random.randint(0, gx1_index - 1)
                        if a < b:
                            jiaohuan = gongxu1[a:b]
                            jiaohuan = jiaohuan[::-1]
                            gongxu1[a:b] = jiaohuan
                            new_individual = Solution([population[i][0].s[0], gongxu1])
                            _,new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due+0.45 * new_individual.Ear
                            if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                flag = 1
                                chishu += 1
                            else:
                                chishu += 1
                        else:
                            jiaohuan = gongxu1[b:a]
                            jiaohuan = jiaohuan[::-1]
                            gongxu1[b:a] = jiaohuan
                            new_individual = Solution([population[i][0].s[0], gongxu1])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                chishu += 1
                                flag = 1
                            else:
                                chishu += 1
                    if flag == 1:
                        offspring.append([off, obj])
                        off_obj = np.vstack((off_obj, obj))
            gongxu11 = population[i][0].s[1]
            gongxu22 = population[int(Temp[0])][0].s[1]
            if (gx1_index+int(Temp[2])-1) <= (len(population[i][0].s[1])-1-2):
                if (gx1_index+int(Temp[2])-1) == (len(population[i][0].s[1])-1-2):
                    f = gongxu11[-2]
                    gongxu11[-2] = gongxu11[-1]
                    gongxu11[-1] = f
                    new_individual = Solution([population[i][0].s[0], gongxu11])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                    else:
                        continue
                elif (gx1_index+int(Temp[2])-1) < (len(population[i][0].s[1])-1-2):
                    count = 0
                    flag = 0#设置标志位
                    off = population[i]
                    obj = np.array([population[i][0].fit, population[i][0].energy, population[i][0].DE, fun_obj[i][3]])
                    while count <= 10:
                        num = range(gx1_index+int(Temp[2]), len(population[i][0].s[1]))
                        num_2 = random.sample(num, 2)
                        a = num_2[0]
                        b = num_2[1]
                        if a < b:
                            gongxu11[a:b] = gongxu11[a:b][::-1]
                            new_individual = Solution([population[i][0].s[0], gongxu11])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                count += 1
                                flag = 1
                            else:
                                count += 1
                        else:

                            gongxu11[b:a] = gongxu11[b:a][::-1]
                            new_individual = Solution([population[i][0].s[0], gongxu11])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[i][0].fit and new_individual.energy <= population[i][0].energy and new_individual.DE <= population[i][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                count += 1
                                flag = 1
                            else:
                                count += 1
                    if flag == 1:
                        offspring.append([off, obj])
                        off_obj = np.vstack((off_obj, obj))
            if gx1_index > 1:
                if gx1_index == 2:
                    f = gongxu2[0]
                    gongxu2[0] = gongxu2[1]
                    gongxu2[1] = f
                    new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu2])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                elif gx1_index > 2:
                    chishu = 0
                    flag = 0
                    off = population[int(Temp[0])]
                    obj = np.array([population[int(Temp[0])][0].fit, population[int(Temp[0])][0].energy, population[int(Temp[0])][0].DE, fun_obj[int(Temp[0])][3]])
                    while chishu <= 10:
                        a = np.random.randint(0, 3)
                        b = np.random.randint(0, 3)
                        while b == a:
                            b = np.random.randint(0, 3)
                        if a < b:
                            gongxu2[a:b] = gongxu2[a:b][::-1]
                            new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu2])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                chishu += 1
                                flag = 1
                            else:
                                chishu += 1
                        else:
                            gongxu2[b:a] = gongxu2[b:a][::-1]
                            new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu2])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                chishu += 1
                                flag = 1
                            else:
                                chishu += 1
                    if flag == 1:
                        offspring.append([off, obj])
                        off_obj = np.vstack((off_obj, obj))
            if (gx1_index + int(Temp[2]) - 1) <= (len(population[int(Temp[0])][0].s[1]) - 1 - 2):
                if (gx1_index + int(Temp[2]) - 1) == (len(population[int(Temp[0])][0].s[1]) - 1 - 2):
                    f = gongxu22[-2]
                    gongxu22[-2] = gongxu22[-1]
                    gongxu22[-1] = f
                    new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu22])
                    _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                    new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                    if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                        obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                        offspring.append([new_individual, np.array(obj_temp)])
                        off_obj = np.vstack((off_obj, obj_temp))
                elif (gx1_index + int(Temp[2]) - 1) < (len(population[int(Temp[0])][0].s[1]) - 1 - 2):
                    count = 0
                    flag = 0
                    off = population[int(Temp[0])]
                    obj = np.array([population[int(Temp[0])][0].fit, population[int(Temp[0])][0].energy, population[int(Temp[0])][0].DE, fun_obj[int(Temp[0])][3]])
                    while count <= 15:
                        num = range(gx1_index + int(Temp[2]), len(population[int(Temp[0])][0].s[1]))
                        num_2 = random.sample(num, 2)
                        a = num_2[0]
                        b = num_2[1]
                        if a < b:
                            gongxu22[a:b] = gongxu22[a:b][::-1]

                            new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu22])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                count += 1
                                flag = 1
                            else:
                                count += 1
                        else:
                            gongxu22[b:a] = gongxu22[b:a][::-1]
                            new_individual = Solution([population[int(Temp[0])][0].s[0], gongxu22])
                            _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                            new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                            if new_individual.fit <= population[int(Temp[0])][0].fit and new_individual.energy <= population[int(Temp[0])][0].energy and new_individual.DE <= population[int(Temp[0])][0].DE:
                                off = new_individual
                                obj_temp = [new_individual.fit, new_individual.energy, new_individual.DE] + fun_obj[i][3:].tolist()
                                obj = np.array(obj_temp)
                                flag = 1
                                count += 1
                            else:
                                count += 1
                    if flag == 1:
                        offspring.append([off, obj])
                        off_obj = np.vstack((off_obj, obj))

    off_obj = np.delete(off_obj, 0, 0)
    return population, offspring, off_obj

def elitism(num_pop: int, Tpop: list, pop_obj: np.array, rank_pop: list):
    l = sum([len(rank_pop[i].ss) for i in range(len(rank_pop))])
    pop_num = []
    pop_num_obj = np.zeros((1, pop_obj.shape[1]))
    pop_rank = []
    sort_index = np.argsort(pop_obj[:, 3])
    pop_rank_obj = pop_obj[sort_index]
    for i in range(len(sort_index)):
        pop_rank.append(Tpop[sort_index[i]])
    max_rank = np.argmax(pop_obj[:, 3])
    if max_rank == 0:
        max_rank = 1
    pre_index = 0
    current_index = 0
    for i in range(int(max_rank)):
        current_index += len(rank_pop[i].ss)
        if current_index > num_pop:
            remain_pop = num_pop - pre_index
            temp_rank_pop = pop_rank[pre_index:current_index]
            temp_pop = []
            temp_obj = pop_rank_obj[pre_index:current_index]
            temp_index = np.argsort(-temp_obj[:, temp_obj.shape[1]-1])
            temp_obj = temp_obj[temp_index]
            for j in range(len(temp_index)):
                temp_pop.append(temp_rank_pop[temp_index[j]])
            for j in range(remain_pop):
                pop_num.append(temp_pop[j])
            pop_num_obj = np.vstack((pop_num_obj, temp_obj[0:remain_pop]))#需要修改
            pop_num_obj = np.delete(pop_num_obj, 0, 0)
            return pop_num, pop_num_obj
        elif current_index < num_pop:
            temp_obj = pop_rank_obj[pre_index:current_index]
            pop_num += pop_rank[pre_index:current_index]
            pop_num_obj = np.vstack((pop_num_obj, temp_obj))
        elif current_index == num_pop:
            temp_obj = pop_rank_obj[pre_index:current_index]
            pop_num += pop_rank[pre_index:current_index]
            pop_num_obj = np.vstack((pop_num_obj, temp_obj))
            pop_num_obj = np.delete(pop_num_obj, 0, 0)
            return pop_num, pop_num_obj
        pre_index = current_index
