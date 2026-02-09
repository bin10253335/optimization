from GA import *
import random
import copy
# class Solution:
#     def __init__(self, s) -> None:
#         self.s = s
#         self.fit = None
#         self.energy = 0
#         self.totalD = 0
#         self.totalE = 0
#         self.DE = 0
class Pareto(object):
    def __init__(self, pareto_rank):
        self.pareto_rank = pareto_rank
        self.ss = []

class PP(object):
    def __init__(self, id):
        self.id = id
        self.n = 0
        self.s = []
def pareto_cross(population: list, fun_obj: np.array, Jobs: list, D: list, E: list, machine_num: int):
    offspring = []
    off_obj = np.zeros((1, fun_obj.shape[1]))
    pareto_set = []
    pareto_zero = []
    for i in range(fun_obj.shape[0]):
        if fun_obj[i][3] == 0:
            pareto_zero.append((population[i], fun_obj[i]))
        else:
            pareto_set.append((population[i], fun_obj[i]))
    # temp = fun_obj[:, 3].tolist()
    # temp.sort()
    # set_temp = list(set(temp))
    # pareto_set = [[] for i in range(len(set_temp))]
    # for i in range(len(set_temp)):
    #     print("I:", i)
    #     for j in range(fun_obj.shape[0]):
    #         if fun_obj[j][3] == set_temp[i]:
    #             pareto_set[i].append(population[j])
    # for i in range(fun_obj.shape[0]):
    #     if fun_obj[i][3] == 0:
    #         pareto_set.append(population[i])
    #for i in range(len(pareto_set)):#
    for i in range(len(pareto_zero)):
        number = [0 for k in range(len(pareto_set))]
        length = np.zeros((1, 3))
        for j in range(len(pareto_set)):
            temp = [a - b for a, b in zip(pareto_zero[i][0].s[1], pareto_set[j][0].s[1])]
            indexs = np.where(temp == 0)[0].tolist()
            if len(indexs) <=3:
                continue#
            else:
                record_matrix = np.zeros((1,2))
                t = 0
                while t < len(indexs)-1:
                    record = 0
                    for l in np.arange(t+1, len(indexs)):
                        if indexs[l] - indexs[l-1] == 1 and (t+1) != len(indexs):
                            record += 1
                        else:
                            record_matrix = np.append(record_matrix, [[t, record]], axis=0)
                            t = l
                            break

                record_matrix = np.delete(record_matrix, 0, axis = 0)
                if len(record_matrix) != 0:
                    max_t = record_matrix[np.argmax(record_matrix, axis=0)[1], 0]
                    length = np.append(length, [[j, max_t, np.max(record_matrix, axis=0)[1]]], 0)
                else:
                    max_t = np.random.randint(len(pareto_set[i][0].s[1])-2, len(pareto_set[i][0].s[1]))
                    length = np.append(length, [[j, max_t, len(pareto_zero[j][0].s[1])-max_t-1]], 0)

        length = np.delete(length, 0, axis=0)
        index = np.argmax(length, axis=0)[2]
        Temp = length[index]
        gongxu1 = pareto_zero[i][0].s[1]
        gongxu1_copy = gongxu1
        gongxu2 = pareto_set[int(Temp[0])][0].s[1]
        gongxu2_copy = gongxu2

        gx1_index = Temp[1]
        if gx1_index > 1:
            if gx1_index == 2:
                f = gongxu2[0]
                gongxu2[0] = gongxu2[1]
                gongxu2[1] = f
                new_individual = Solution([pareto_set[int(Temp[0])][0].s[0], gongxu2])
                _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                if new_individual.fit <= pareto_set[int(Temp[0])][0].fit and new_individual.energy <= pareto_set[int(Temp[0])][0].energy and new_individual.DE <= pareto_set[int(Temp[0])][0].DE:
                    offspring.append(new_individual)
                    off_obj = np.vstack((off_obj, [new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]]))
                else:
                    continue
            elif gx1_index > 2:
                chishu = 0
                off = pareto_set[int(Temp[0])][0]
                obj = pareto_set[int(Temp[0])][1]
                while chishu <= 10:
                    a = np.random.randint(0, gx1_index-1)
                    b = np.random.randint(0, gx1_index-1)
                    while b == a:
                        b = np.random.randint(0, gx1_index - 1)
                    if a < b:
                        gongxu2[a:b] = gongxu2[a:b][::-1]
                        new_individual = Solution([pareto_set[int(Temp[0])][0].s[0], gongxu2])
                        _,new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due+0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_set[int(Temp[0])][0].fit and new_individual.energy <= pareto_set[int(Temp[0])][0].energy and new_individual.DE <= pareto_set[int(Temp[0])][0].DE:
                            off = new_individual
                            obj = np.array([new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]])
                            chishu += 1
                        else:
                            chishu += 1
                    else:
                        gongxu2[b:a] = gongxu2[b:a][::-1]
                        new_individual = Solution([pareto_set[int(Temp[0])][0].s[0], gongxu2])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_set[int(Temp[0])][0].fit and new_individual.energy <= pareto_set[int(Temp[0])][0].energy and new_individual.DE <= pareto_set[int(Temp[0])][0].DE:
                            off = new_individual
                            obj = np.array([new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]])
                            chishu += 1
                        else:
                            chishu += 1
                offspring.append(off)
                off_obj = np.vstack((off_obj, obj))

        gongxu11 = pareto_zero[i][0].s[1]
        gongxu22 = pareto_set[int(Temp[0])][0].s[1]
        if (gx1_index+Temp[2]-1) <= (len(pareto_set[int(Temp[0])][0].s[1])-1-2):
            if (gx1_index+Temp[2]-1) == (len(pareto_set[int(Temp[0])][0].s[1])-1-2):
                f = gongxu22[-2]
                gongxu22[-2] = gongxu22[-1]
                gongxu22[-1] = f
            elif (gx1_index+Temp[2]-1) < (len(pareto_set[int(Temp[0])][0].s[1])-1-2):
                count = 0
                off = pareto_set[int(Temp[0])][0]
                obj = pareto_set[int(Temp[0])][1]
                while count <= 10:
                    num = range(gx1_index+Temp[2], len(pareto_set[int(Temp[0])][0].s[1]))
                    num_2 = random.sample(num, 2)
                    a = num_2[0]
                    b = num_2[1]
                    if a < b:
                        gongxu22[a:b] = gongxu22[a:b][::-1]
                        new_individual = Solution([pareto_set[int(Temp[0])][0].s[0], gongxu22])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_set[int(Temp[0])][0].fit and new_individual.energy <= pareto_set[int(Temp[0])][0].energy and new_individual.DE <= pareto_set[int(Temp[0])][0].DE:
                            off = new_individual
                            obj = np.array([new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]])

                            count += 1
                        else:
                            count += 1
                    else:
                        gongxu22[b:a] = gongxu22[b:a][::-1]
                        new_individual = Solution([pareto_set[int(Temp[0])][1].s[0], gongxu22])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_set[int(Temp[0])][1].fit and new_individual.energy <= pareto_set[int(Temp[0])][1].energy and new_individual.DE <= pareto_set[int(Temp[0])][1].DE:
                            off = new_individual
                            obj = np.array([new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]])
                            count += 1
                        else:
                            count += 1
                offspring.append(off)
                off_obj = np.vstack((off_obj, obj))
        if gx1_index > 1:
            if gx1_index == 2:
                f = gongxu1[0]
                gongxu1[0] = gongxu1[1]
                gongxu1[1] = f
                new_individual = Solution([pareto_zero[i][0].s[0], gongxu1])
                _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear

                if new_individual.fit <= pareto_zero[i][0].fit and new_individual.energy <= pareto_zero[i][0].energy and new_individual.DE <= pareto_zero[i][0].DE:
                    offspring.append(new_individual)
                    off_obj = np.vstack((off_obj, [new_individual.fit, new_individual.energy, new_individual.DE, fun_obj[Temp[0]]]))
                else:
                    continue
            elif gx1_index > 2:
                chishu = 0
                off = pareto_zero[i][0]
                obj = pareto_zero[i][1]
                while chishu <= 10:
                    a = np.random.randint(0, 3)
                    b = np.random.randint(0, 3)
                    while b == a:
                        b = np.random.randint(0, 3)
                    if a < b:
                        gongxu1[a:b] = gongxu1[a:b][::-1]
                        new_individual = Solution([pareto_zero[i][0].s[0], gongxu1])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_zero[i][0].fit and new_individual.energy <= pareto_zero[i][0].energy and new_individual.DE <= pareto_zero[i][0].DE:
                            off = new_individual
                            obj = np.array([[new_individual.fit, new_individual.energy, new_individual.DE]])
                            chishu += 1
                        else:
                            chishu += 1
                    else:
                        gongxu1[b:a] = gongxu1[b:a][::-1]
                        new_individual = Solution([pareto_zero[i][0].s[0], gongxu1])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_zero[i][0].fit and new_individual.energy <= pareto_zero[i][0].energy and new_individual.DE <= pareto_zero[i][0].DE:
                            off = new_individual
                            obj = np.array([[new_individual.fit, new_individual.energy, new_individual.DE]])
                            chishu += 1
                        else:
                            chishu += 1
                offspring.append(off)
                off_obj = np.vstack((off_obj, obj))
        if (gx1_index + Temp[2] - 1) <= (len(pareto_zero[i][0].s[1]) - 1 - 2):
            if (gx1_index + Temp[2] - 1) == (len(pareto_zero[i][0].s[1]) - 1 - 2):
                f = gongxu11[-2]
                gongxu11[-2] = gongxu11[-1]
                gongxu11[-1] = f
            elif (gx1_index + Temp[2] - 1) < (len(pareto_zero[i][0].s[1]) - 1 - 2):
                count = 0
                off = pareto_zero[i][0]
                obj = pareto_zero[i][1]
                while count <= 10:
                    num = range(gx1_index + Temp[2], len(pareto_zero[i][0].s[1]))
                    num_2 = random.sample(num, 2)
                    a = num_2[0]
                    b = num_2[1]
                    if a < b:

                        gongxu11[a:b] = gongxu11[a:b][::-1]
                        new_individual = Solution([pareto_zero[i][0].s[0], gongxu11])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_zero[i][0].fit and new_individual.energy <= pareto_zero[i][0].energy and new_individual.DE <= pareto_zero[i][0].DE:
                            off = new_individual
                            obj = np.array([[new_individual.fit, new_individual.energy, new_individual.DE]])
                            count += 1
                        else:
                            count += 1
                    else:
                        gongxu11[b:a] = gongxu11[b:a][::-1]
                        new_individual = Solution([pareto_zero[i][0].s[0], gongxu11])
                        _, new_individual.fit, new_individual.energy, new_individual.Ear, new_individual.Due = decoding(new_individual.s, Jobs, D, E, machine_num)
                        new_individual.DE = 0.55 * new_individual.Due + 0.45 * new_individual.Ear
                        if new_individual.fit <= pareto_zero[i][0].fit and new_individual.energy <= pareto_zero[i][0].energy and new_individual.DE <= pareto_zero[i][0].DE:
                            off = new_individual
                            obj = np.array([[new_individual.fit, new_individual.energy, new_individual.DE]])
                            count += 1
                        else:
                            count += 1
                offspring.append(off)
                off_obj = np.vstack((off_obj, obj))
    off_obj = np.delete(off_obj, 0, 0)
    return pareto_zero+pareto_set, offspring, off_obj