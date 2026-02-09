import numpy as np
from coding import decoding, encoding
from GA import Solution
import copy
import random

def machine_cross(population: list, fun_obj: np.array, Jobs: list, D: list, E: list, machine_num: int):
    offspring = []
    off_obj = np.zeros((1, fun_obj.shape[1]))
    for i in range(len(population)):
        p = copy.copy(population[i])
        temp_offspring = population[i]
        temp_off_obj = fun_obj[i]
        count = 0
        while count < 10:
            appear_count = 0
            l = np.random.randint(0, len(p[0].s[1]))
            for k in range(0, l):
                if p[0].s[1][k] == p[0].s[1][l]:
                    appear_count += 1
            length_s1 = len(Jobs[p[0].s[1][l]].minfo[appear_count])
            if p[0].s[1][l] > 0:
                temp_number = list(np.cumsum([Jobs[w].osNum for w in range(p[0].s[1][l])]))[-1]
                temp_number += appear_count
            else:
                temp_number = appear_count
            initial_sol = p[0].s[0][temp_number]
            for j in range(length_s1):
                temp_psl = p[0].s[0][temp_number]
                if j != initial_sol:
                    p[0].s[0][temp_number] = j
                _, p[0].fit, p[0].energy, p[0].E, p[0].D = decoding(p[0].s, Jobs, D, E, machine_num)
                p[0].DE = 0.45 * p[0].D + 0.50 * p[0].E
                DE_differ1, compT_differ1, energy_differ1 = population[i][0].DE - p[0].DE, population[i][0].fit - p[0].fit, population[i][0].energy - p[0].energy
                if all(j >= 0 for j in [DE_differ1, compT_differ1, energy_differ1]) and any(j > 0 for j in [DE_differ1, compT_differ1, energy_differ1]):

                    temp_obj = fun_obj[i]
                    temp_obj[0:3] = [p[0].fit, p[0].energy, p[0].DE]
                    temp_offspring = [p[0], temp_obj]
                    temp_off_obj = temp_obj
                else:
                    p[0].s[0][temp_number] = temp_psl
            count = count + 1
        offspring.append(temp_offspring)
        off_obj = np.vstack((off_obj, temp_off_obj))
    off_obj = np.delete(off_obj, 0, 0)
    return offspring, off_obj
