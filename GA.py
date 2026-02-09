from pareto_sort import pareto_cross
import numpy as np
import random
from coding import encoding, decoding
import copy
from sklearn import preprocessing
class Solution:
    def __init__(self, s) -> None:
        self.s = s
        
        self.fit = 0
        self.energy = 0
        self.Due = 0
        self.Ear = 0
        self.DE = 0

class Q(object):
    def __init__(self, s, a):
        self.state = 0
        self.epsilon = 0.8
        self.current_state = random.randint(0, s-1)
        self.table = np.random.uniform(0, 0.1, size=(s,a))

def initializePopulation(Jobs, os, machines_num, popSize):
    population = [Solution(encoding(Jobs, os, machines_num, stype='GS')),
                  Solution(encoding(Jobs, os, machines_num, stype='LS'))
                  ]
    for i in range(2, popSize):
        population.append(Solution(encoding(Jobs, os, machines_num, stype='RS')))
    return population


def Qlearning(population: list, fun_obj: np.array, lsr: float, q: Q, Differ, rs, jobSequence, Jobs, D, E, machines_num, allOT):

    lm = 0
    temp = fun_obj[:, 3].tolist()
    temp.sort()
    set_temp = list(set(temp))
    colunm_number = fun_obj.shape[1]
    pareto = [[] for i in range(len(set_temp))]
    cur_state = q.state
    lm_value = 0
    for i in range(len(set_temp)):
        for j in range(fun_obj.shape[0]):
            if fun_obj[j][3] == set_temp[i]:
                pareto[i].append([population[j], fun_obj[j]])
    all_Dgen = []
    ac_gen = []
    for i in range(len(pareto)):
        fun_value = np.zeros((len(pareto[i]), 3))
        for j in range(len(pareto[i])):
            fun_value[j] = [pareto[i][j][0].fit, pareto[i][j][0].energy, pareto[i][j][0].DE]
        if len(pareto[i]) >= 2:
            max_obj = np.max(fun_value[:, 0:3], axis=0)
            min_obj = np.min(fun_value[:, 0:3], axis=0)
            if (max_obj - min_obj).all() > 0:
                fun_value = (fun_value - min_obj) / (max_obj - min_obj)
                
                max_obj = (max_obj - min_obj) / (max_obj - min_obj)
                min_obj = (min_obj - min_obj) / (max_obj - min_obj)
            else:
                scaler = preprocessing.Normalizer()
                fun_value = scaler.fit_transform(fun_value)
                max_obj = scaler.fit_transform(np.expand_dims(max_obj, axis=0))
                min_obj = scaler.fit_transform(np.expand_dims(min_obj, axis=0))
                max_obj = np.reshape(max_obj, (3,))
                min_obj = np.reshape(min_obj, (3,))
            avr_dis = np.sum(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])) / fun_value.shape[0]
            if avr_dis == 0:
                D_gen = 0
            else:
                D_gen = np.sqrt(np.sum(np.square(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2]) - avr_dis)) / fun_value.shape[0]) / avr_dis
            if i == 0:
                lm_dis = np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])  
                lm_min = np.min(lm_dis)
                if lm_min < lm_value:
                    lm = 1
        else:
            scaler = preprocessing.Normalizer()
            fun_value = scaler.fit_transform(fun_value)
            min_obj = np.array([0,0,0])
            avr_dis = np.sum(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2]))
            D_gen = np.sqrt(np.sum(np.square(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2]) - avr_dis)) / fun_value.shape[0]) / avr_dis
        all_Dgen.append(D_gen)
    F_Dgen = sum(all_Dgen) / len(all_Dgen)
    if np.random.rand() < q.epsilon:
        action = int(np.random.rand() * len(q.table[1]))
    else:
        action = np.argmax(q.table[cur_state])
    if action == 0:
        ac1_objfun = np.zeros((1, fun_obj.shape[1]))
        ac1_gen = []
        for i in range(len(pareto)):

            ac1_gen.append([])
            for j in range(len(pareto[i])):
                temp_contr = []
                k = np.random.randint(0, len(pareto[i]))
                temp1 = copy.copy(pareto[i][j])
                temp2 = copy.copy(pareto[i][k])
                temp_contr.append(temp1)
                temp_contr.append(temp2)
                C1, C2 = cross(pareto[i][j][0].s, pareto[i][k][0].s, rs, jobSequence)
                pareto[i][j][0].s = C1
                pareto[i][k][0].s = C2
                temp_contr.append(pareto[i][j])
                temp_contr.append(pareto[i][k])
                for o in range(2, len(temp_contr)):
                    _, temp_contr[o][0].fit, temp_contr[o][0].energy, temp_contr[o][0].Ear, temp_contr[o][0].Due = decoding(temp_contr[o][0].s,Jobs, D, E, machines_num)
                    temp_contr[o][0].DE = 0.45 * temp_contr[o][0].Ear + 0.45 * temp_contr[o][0].Due
                obj_value = np.zeros((1, 3))
                for o in range(len(temp_contr)):
                    obj_value = np.append(obj_value, [[temp_contr[o][0].fit, temp_contr[o][0].energy, temp_contr[o][0].DE]], axis=0)
                obj_value = np.delete(obj_value, 0, 0)
                max_obj = np.max(obj_value[:, 0:3], axis=0)
                min_obj = np.min(obj_value[:, 0:3], axis=0)
                if (max_obj - min_obj).all() > 0:
                    obj_value = (obj_value - min_obj) / (max_obj - min_obj)
                    max_obj = (max_obj - min_obj) / (max_obj - min_obj)
                    min_obj = (min_obj - min_obj) / (max_obj - min_obj)
                else:
                    scaler = preprocessing.Normalizer()
                    obj_value = scaler.fit_transform(obj_value)
                    max_obj = scaler.fit_transform(np.expand_dims(max_obj, axis=0))
                    min_obj = scaler.fit_transform(np.expand_dims(min_obj, axis=0))
                    max_obj = np.reshape(max_obj, (3,))
                    min_obj = np.reshape(min_obj, (3,))
                dis = np.sqrt(np.cumsum(np.square((obj_value[:, 0:3] - min_obj)), 1)[:, 2])
                dis = dis.tolist()

                if max([dis[2], dis[3]]) < max([dis[0], dis[1]]) and min([dis[2], dis[3]]) < min([dis[0], dis[1]]):
                    pareto[i][j][1][0:3] = np.array([temp_contr[2][0].fit, temp_contr[2][0].energy, temp_contr[2][0].DE])
                    pareto[i][k][1][0:3] = np.array([temp_contr[3][0].fit, temp_contr[3][0].energy, temp_contr[3][0].DE])
                elif max([dis[2], dis[3]]) > max([dis[0], dis[1]]) and min([dis[2], dis[3]]) < min([dis[0], dis[1]]):
                    if np.argmax([dis[0], dis[1]]) == 0:
                        pareto[i][k][0].s = temp2[0].s
                    else:
                        pareto[i][k][1][0:3] = np.array([temp_contr[np.argmin([dis[2], dis[3]]) + 1][0].fit, temp_contr[2][0].energy, temp_contr[2][0].DE])

                        pareto[i][j][0].s = temp1[0].s

                else:
                    pareto[i][j][0].s = temp1[0].s
                    pareto[i][k][0].s = temp2[0].s
            P_cross = []
            P_fun = np.expand_dims(pareto[i][0][1], 0)
            for j in range(len(pareto[i])):
                P_cross.append(pareto[i][j])
                if j > 0:
                    P_fun = np.vstack((P_fun, pareto[i][j][1]))
            if len(P_cross) >= 2:
                parents, offspring, off_obj = pareto_cross(P_cross, P_fun, Jobs, D, E, machines_num)
            else:
                offspring = P_cross
                off_obj = np.expand_dims(pareto[i][0][1], 0)
            if len(offspring) != 0 and len(offspring) < len(P_cross):
                ac1_gen[i] = offspring
                ac1_objfun = np.vstack((ac1_objfun, off_obj))
                for j in range(len(P_cross)-len(offspring)):
                    ac1_gen[i].append(P_cross[j])
                    ac1_objfun = np.vstack((ac1_objfun, pareto[i][j][1]))
            elif len(offspring) != 0 and len(offspring) == len(P_cross):
                ac1_gen[i] = offspring
                ac1_objfun = np.vstack((ac1_objfun, off_obj))
            elif len(offspring) != 0 and len(offspring) > len(P_cross):
                ac1_gen[i] = offspring[0:len(P_cross)]
                ac1_objfun = np.vstack((ac1_objfun, off_obj[0:len(P_cross)]))
            else:
                ac1_gen[i] = P_cross
                ac1_objfun = np.vstack((ac1_objfun, P_fun))
        ac1_objfun = np.delete(ac1_objfun, 0, 0)
        ac1_result = ac1_gen[0]
        for i in range(1, len(ac1_gen)):
            ac1_result += ac1_gen[i]
    elif action == 1:
        ac2_gen = []
        ac2_objfun = np.zeros((1, fun_obj.shape[1]))
        for i in range(len(pareto)):
            ac2_gen.append([])
            orig_sol = []
            for j in range(len(pareto[i])):
                orig_sol.append(copy.copy(pareto[i][j]))
                pareto[i][j][0].s = mutation(pareto[i][j][0].s, rs, allOT)
                _, pareto[i][j][0].fit, pareto[i][j][0].energy, pareto[i][j][0].Ear, pareto[i][j][0].Due = decoding(pareto[i][j][0].s, Jobs, D, E, machines_num)
                pareto[i][j][0].DE = 0.45 * pareto[i][j][0].Ear + 0.45 * pareto[i][j][0].Due
            total_sol = orig_sol + pareto[i]
            fun_value = np.zeros((len(total_sol), 3))

            for j in range(len(total_sol)):

                fun_value[j] = [total_sol[j][0].fit, total_sol[j][0].energy, total_sol[j][0].DE]
            max_obj = np.max(fun_value[:, 0:3], axis=0)
            min_obj = np.min(fun_value[:, 0:3], axis=0)
            if (max_obj - min_obj).all() > 0:
                fun_value = (fun_value - min_obj) / (max_obj - min_obj)

                max_obj = (max_obj - min_obj) / (max_obj - min_obj)
                min_obj = (min_obj - min_obj) / (max_obj - min_obj)
            else:
                scaler = preprocessing.Normalizer()
                fun_value = scaler.fit_transform(fun_value)
                max_obj = scaler.fit_transform(np.expand_dims(max_obj, axis=0))
                min_obj = scaler.fit_transform(np.expand_dims(min_obj, axis=0))
                max_obj = np.reshape(max_obj, (3,))
                min_obj = np.reshape(min_obj, (3,))
            dis = np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])
            for j in range(len(orig_sol)):
                if dis[j] > dis[j+len(orig_sol)]:
                    orig_sol[j][0] = total_sol[j+len(orig_sol)][0]
                    orig_sol[j][1][0:3] = np.array([total_sol[j+len(orig_sol)][0].fit, total_sol[j+len(orig_sol)][0].energy, total_sol[j+len(orig_sol)][0].DE])
                else:
                    continue
            P_fun = np.expand_dims(orig_sol[0][1], 0)
            for j in range(len(orig_sol)):
                if j > 0:
                    P_fun = np.vstack((P_fun, orig_sol[j][1]))
                else:
                    continue
            if len(orig_sol) >= 2:
                parents, offspring, off_obj = pareto_cross([i for i in orig_sol], P_fun, Jobs, D, E, machines_num)
            else:
                offspring = orig_sol
                off_obj = np.expand_dims(orig_sol[0][1], 0)
            if len(offspring) != 0 and len(offspring) < len(orig_sol):
                ac2_gen[i] = offspring
                ac2_objfun = np.vstack((ac2_objfun, off_obj))
                for j in range(len(orig_sol)-len(offspring)):
                    ac2_gen[i].append(orig_sol[j])
                    ac2_objfun = np.vstack((ac2_objfun, orig_sol[j][1]))
            elif len(offspring) != 0 and len(offspring) == len(orig_sol):
                ac2_gen[i] = offspring
                ac2_objfun = np.vstack((ac2_objfun, off_obj))
            elif len(offspring) != 0 and len(offspring) > len(orig_sol):
                ac2_gen[i] = offspring[0:len(orig_sol)]
                ac2_objfun = np.vstack((ac2_objfun, off_obj[0:len(orig_sol)]))
            else:
                ac2_gen[i] = [k for k in orig_sol]
                ac2_objfun = np.vstack((ac2_objfun, P_fun))
        ac2_objfun = np.delete(ac2_objfun, 0, 0)
        ac2_result = ac2_gen[0]
        for i in range(1, len(ac2_gen)):
            ac2_result += ac2_gen[i]
    elif action == 2:
        ac3_gen = [] 
        ac3_objfun = np.zeros((1, fun_obj.shape[1]))
        for i in range(len(pareto)):
            ac3_gen.append([]) 
            for j in range(len(pareto[i])):
                temp_contr = [] 
                k = np.random.randint(0, len(pareto[i]))
                temp1 = copy.copy(pareto[i][j]) 
                temp2 = copy.copy(pareto[i][k])
                temp_contr.append(temp1)
                temp_contr.append(temp2)
                C1, C2 = cross(pareto[i][j][0].s, pareto[i][k][0].s, rs, jobSequence)
                pareto[i][j][0].s = C1
                pareto[i][k][0].s = C2
                temp_contr.append(pareto[i][j])
                temp_contr.append(pareto[i][k])
                for o in range(2, len(temp_contr)):  
                    
                    _, temp_contr[o][0].fit, temp_contr[o][0].energy, temp_contr[o][0].Ear, temp_contr[o][0].Due = decoding(temp_contr[o][0].s, Jobs, D, E, machines_num)
                    temp_contr[o][0].DE = 0.45 * temp_contr[o][0].Ear + 0.45 * temp_contr[o][0].Due
                obj_value = np.zeros((1, 3))
                for o in range(len(temp_contr)):
                    obj_value = np.append(obj_value,[[temp_contr[o][0].fit, temp_contr[o][0].energy, temp_contr[o][0].DE]], axis=0) 
                obj_value = np.delete(obj_value, 0, 0) 
                max_obj = np.max(obj_value[:, 0:3], axis=0) 
                min_obj = np.min(obj_value[:, 0:3], axis=0)
                
                if (max_obj - min_obj).all() > 0:
                    obj_value = (obj_value - min_obj) / (max_obj - min_obj)
                    
                    max_obj = (max_obj - min_obj) / (max_obj - min_obj)
                    min_obj = (min_obj - min_obj) / (max_obj - min_obj)
                else:
                    scaler = preprocessing.Normalizer()
                    obj_value = scaler.fit_transform(obj_value)
                    max_obj = scaler.fit_transform(np.expand_dims(max_obj, axis=0))
                    min_obj = scaler.fit_transform(np.expand_dims(min_obj, axis=0))
                    max_obj = np.reshape(max_obj, (3,))
                    min_obj = np.reshape(min_obj, (3,))

                dis = np.sqrt(np.cumsum(np.square((obj_value[:, 0:3] - min_obj)), 1)[:, 2])
                dis = dis.tolist() 

                if max([dis[2], dis[3]]) < max([dis[0], dis[1]]) and min([dis[2], dis[3]]) < min([dis[0], dis[1]]):
                    pareto[i][j][1][0:3] = np.array([temp_contr[2][0].fit, temp_contr[2][0].energy, temp_contr[2][0].DE])
                    
                    pareto[i][k][1][0:3] = np.array([temp_contr[3][0].fit, temp_contr[3][0].energy, temp_contr[3][0].DE])
                elif max([dis[2], dis[3]]) > max([dis[0], dis[1]]) and min([dis[2], dis[3]]) < min([dis[0], dis[1]]):
                    if np.argmax([dis[0], dis[1]]) == 0:
                        
                        
                        pareto[i][j][1][0:3] = np.array([temp_contr[np.argmin([dis[2], dis[3]]) + 1][0].fit, temp_contr[2][0].energy,temp_contr[2][0].DE])
                        
                        
                        pareto[i][k][0].s = temp2[0].s
                    else:
                        pareto[i][k][1][0:3] = np.array([temp_contr[np.argmin([dis[2], dis[3]]) + 1][0].fit, temp_contr[2][0].energy,temp_contr[2][0].DE])
                        
                        pareto[i][j][0].s = temp1[0].s

                else:
                    
                    rand = np.random.rand()
                    if rand < 0.1:
                        
                        pareto[i][j][1][0:3] = np.array([temp_contr[2][0].fit, temp_contr[2][0].energy, temp_contr[2][0].DE])
                        
                        pareto[i][k][1][0:3] = np.array([temp_contr[3][0].fit, temp_contr[3][0].energy, temp_contr[3][0].DE])
                    else:
                        pareto[i][j][0].s = temp1[0].s
                        pareto[i][k][0].s = temp2[0].s
            
            
            P_cross = []  
            P_fun = np.expand_dims(pareto[i][0][1], 0)
            for j in range(len(pareto[i])):
                P_cross.append(pareto[i][j])
                
                if j > 0:
                    P_fun = np.vstack((P_fun, pareto[i][j][1]))
            if len(P_cross) >= 2:
                parents, offspring, off_obj = pareto_cross(P_cross, P_fun, Jobs, D, E, machines_num)
            else:
                offspring = P_cross
                off_obj = P_fun
            
            if len(offspring) != 0 and len(offspring) < len(P_cross):
                ac3_gen[i] = offspring
                ac3_objfun = np.vstack((ac3_objfun, off_obj))
                for j in range(len(P_cross) - len(offspring)):
                    ac3_gen[i].append(P_cross[j])
                    
                    ac3_objfun = np.vstack((ac3_objfun, pareto[i][j][1]))
            elif len(offspring) != 0 and len(offspring) == len(P_cross):
                ac3_gen[i] = offspring
                ac3_objfun = np.vstack((ac3_objfun, off_obj))
            elif len(offspring) != 0 and len(offspring) > len(P_cross):
                ac3_gen[i] = offspring[0:len(P_cross)]
                ac3_objfun = np.vstack((ac3_objfun, off_obj[0:len(P_cross)]))
            else:
                ac3_gen[i] = P_cross
                ac3_objfun = np.vstack((ac3_objfun, P_fun))
        ac3_objfun = np.delete(ac3_objfun, 0, 0)
        ac3_result = ac3_gen[0]
        for i in range(1, len(ac3_gen)):
            ac3_result += ac3_gen[i]
        
    elif action == 3:
        ac4_gen = []
        ac4_objfun = np.zeros((1, fun_obj.shape[1]))  
        
        for i in range(len(pareto)):
            ac4_gen.append([])
            orig_sol = []  
            for j in range(len(pareto[i])):
                orig_sol.append(copy.copy(pareto[i][j]))
                off_s = mutation(pareto[i][j][0].s, rs, allOT)
                
                
                _, pareto[i][j][0].fit, pareto[i][j][0].energy, pareto[i][j][0].Ear, pareto[i][j][0].Due = decoding(off_s, Jobs, D, E, machines_num)
                pareto[i][j][0].DE = 0.45 * pareto[i][j][0].Ear + 0.45 * pareto[i][j][0].Due
            
            total_sol = orig_sol + pareto[i]  
            fun_value = np.zeros((len(total_sol), 3))  
            for j in range(len(total_sol)):
                
                fun_value[j] = [total_sol[j][0].fit, total_sol[j][0].energy, total_sol[j][0].DE]  
            
            max_obj = np.max(fun_value[:, 0:3], axis=0)  
            min_obj = np.min(fun_value[:, 0:3], axis=0)  
            
            
            if (max_obj - min_obj).all() > 0:
                fun_value = (fun_value - min_obj) / (max_obj - min_obj)
                
                max_obj = (max_obj - min_obj) / (max_obj - min_obj)
                min_obj = (min_obj - min_obj) / (max_obj - min_obj)
            else:
                scaler = preprocessing.Normalizer()
                fun_value = scaler.fit_transform(fun_value)
                max_obj = scaler.fit_transform(np.expand_dims(max_obj, axis=0))
                min_obj = scaler.fit_transform(np.expand_dims(min_obj, axis=0))
                max_obj = np.reshape(max_obj, (3,))
                min_obj = np.reshape(min_obj, (3,))
            dis = np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])  
            for j in range(len(orig_sol)):
                if dis[j] > dis[j + len(orig_sol)]:
                    
                    orig_sol[j][0] = total_sol[j + len(orig_sol)][0]  
                    orig_sol[j][1][0:3] = np.array([total_sol[j + len(orig_sol)][0].fit, total_sol[j + len(orig_sol)][0].energy, total_sol[j + len(orig_sol)][0].DE])
                else:
                    rand = np.random.rand()
                    if rand < 0.1:
                        orig_sol[j][0] = total_sol[j + len(orig_sol)][0]  
                        orig_sol[j][1][0:3] = np.array([total_sol[j + len(orig_sol)][0].fit, total_sol[j + len(orig_sol)][0].energy, total_sol[j + len(orig_sol)][0].DE])

                    else:
                        continue
            
            P_fun = np.expand_dims(orig_sol[0][1], 0)
            for j in range(len(orig_sol)):
                if j > 0:
                    P_fun = np.vstack((P_fun, orig_sol[j][1]))
            if len(orig_sol) >= 2:
                parents, offspring, off_obj = pareto_cross(orig_sol, P_fun, Jobs, D, E, machines_num)
            else:
                offspring = []
                offspring.append(orig_sol[0])
                off_obj = np.expand_dims(orig_sol[0][1], 0)
            if len(offspring) != 0 and len(offspring) < len(orig_sol):
                ac4_gen[i] = offspring
                ac4_objfun = np.vstack((ac4_objfun, off_obj))
                for j in range(len(orig_sol) - len(offspring)):
                    ac4_gen[i].append(orig_sol[j])
                    ac4_objfun = np.vstack((ac4_objfun, orig_sol[j][1]))
            elif len(offspring) != 0 and len(offspring) == len(orig_sol):
                ac4_gen[i] = offspring
                ac4_objfun = np.vstack((ac4_objfun, off_obj))
            elif len(offspring) != 0 and len(offspring) > len(orig_sol):
                ac4_gen[i] = offspring[0:len(orig_sol)]
                ac4_objfun = np.vstack((ac4_objfun, off_obj[0:len(orig_sol)]))
            else:
                fin_sol = [first for first in orig_sol]
                
                ac4_gen[i] = fin_sol
                ac4_objfun = np.vstack((ac4_objfun, P_fun))
        ac4_objfun = np.delete(ac4_objfun, 0, 0)
        ac4_result = ac4_gen[0]
        for i in range(1, len(ac4_gen)):
            ac4_result += ac4_gen[i]
    if lm == 1:
        if F_Dgen > Differ:
            q.state = 1
        elif F_Dgen == Differ:
            q.state = 2
        else:
            q.state = 3
    else:
        if F_Dgen > Differ:
            q.state = 4
        elif F_Dgen == Differ:
            q.state = 5
        else:
            q.state = 6
    
    if action == 0:
        
        
        
        
        return ac1_result, ac1_objfun, action, lm, F_Dgen
    elif action == 1:
        
        
        
        return ac2_result, ac2_objfun, action, lm, F_Dgen
    elif action == 2:
        
        
        
        return ac3_result, ac3_objfun, action, lm, F_Dgen
    elif action == 3:
        
        
        
        return ac4_result, ac4_objfun, action, lm, F_Dgen
    
    
    



def get_reward(population: list, fun_obj: np.array, lm, Differ, q: Q):
    
    
    
    
    cur_state = q.current_state  
    lm_value = 0  
    
    
    
    
    
    
    temp = fun_obj[:, 3].tolist()  
    temp.sort()
    set_temp = list(set(temp))  
    colunm_number = fun_obj.shape[1]  
    
    pareto = [[] for i in range(len(set_temp))]  
    
    for i in range(len(set_temp)):
        
        
        for j in range(fun_obj.shape[0]):
            if fun_obj[j][3] == set_temp[i]:
                pareto[i].append(population[j])
                
        
    
    
    all_Dgen = []
    
    for i in range(len(pareto)):
        fun_value = np.zeros((len(pareto[i]), 3))  
        for j in range(len(pareto[i])):
            fun_value = np.append(fun_value, [[pareto[i][j][0].fit, pareto[i][j][0].energy, pareto[i][j][0].DE]], axis=0)  

        
        max_obj = np.max(fun_value[:, 0:3], axis=0)  
        min_obj = np.min(fun_value[:, 0:3], axis=0)  
        
        fun_value = (fun_value - min_obj) / (max_obj - min_obj)
        
        avr_dis = np.sum(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])) / fun_value.shape[0]
        
        
        D_gen = np.sqrt(np.sum(np.square(np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2]) - avr_dis)) / fun_value.shape[0]) / avr_dis
        D_gen = D_gen.tolist()
        
        all_Dgen.append(D_gen)  
        if i == 0:
            lm_dis = np.sqrt(np.cumsum(np.square((fun_value[:, 0:3] - min_obj)), 1)[:, 2])  
            lm_max = np.max(lm_dis)  
            if lm_max > lm_value:
                lm = 1  
    
    
    F_Dgen = sum(all_Dgen) / len(all_Dgen)
    if lm == 1 and F_Dgen > Differ:
        state = 0
    elif lm == 1 and F_Dgen == Differ:
        state = 1
    elif lm == 1 and F_Dgen < Differ:
        state = 2
    elif lm == 0 and F_Dgen > Differ:
        state = 3
    elif lm == 0 and F_Dgen == Differ:
        state = 4
    else:
        state = 5
    reward = cur_state - state
    return reward, cur_state, state

    


def update_qtable(q: Q, state: int, state_t: int, action: int, reward):
    q.state = state_t
    q.table[state][action] = q.table[state][action] + 0.1*(reward+0.9*np.max(q.table[action])-q.table[state][action])
    return True

def selection(population: list, fitnessratio):
    '''轮盘赌选择
    population:种群[individual]
    fitness:种群个体对应的适应度
    return:
        best:当前锦标赛中最好的个体
    '''
    r = random.random()
    target_idx = np.where(fitnessratio >= r)[0][0]
    while len(np.where(fitnessratio >= r)[0]) == 0:
        target_idx = np.where(fitnessratio >= r)[0][0]
    return population[target_idx]


def getbest(population):

    fitness = np.array([individual.fit for individual in population])
    total_energy = np.array([individual.energy for individual in population])
    total_DE = np.array([individual.DE for individual in population])
    final = np.vstack((fitness, total_energy, total_DE))
    
    num_max = np.max(final, 0)
    num_min = np.min(final, 0)
    process = (final - num_min) / (num_max - num_min)
    cum = np.cumsum(process, 0)[2]
    bestidx = np.argmin(cum)
    
    bestS = population[bestidx]
    return bestS



def cross(P1: list, P2: list, rs: list, jobSequence: list):
    '''
    P1,P2 [ms,os]
    rs:机器选择位点rs=list(range(工序总数))
    jobSequence:工件当前在Jobs列表中的的相对顺序=list(range(工件总数))
    '''

    def mscross(ms1: np.array, ms2: np.array, rs: list):
        '''均匀交叉
        ms1,ms2:父代机器选择部分
        rs:机器选择位点rs=list(range(工序总数))
        '''
        r = random.randint(1, len(rs))
        
        crossIndex = random.sample(rs, r)  
        c1, c2 = ms1.copy(), ms2.copy()
        c1[crossIndex], c2[crossIndex] = c2[crossIndex], c1[crossIndex]

        return c1, c2

    def oscross(os1: np.array, os2: np.array, jobSequence: list):

        def generate_offspring(os1, os2, c):
            i, j = 0, 0
            while i < len(os2):
                if os2[i] not in crossIds:
                    while j < len(os1):
                        if os1[j] not in crossIds:
                            c[j] = os2[i]
                            j += 1
                            break
                        j += 1
                i += 1
            return c

        r = random.randint(1, len(jobSequence))  
        crossIds = sorted(random.sample(jobSequence, r))  
        c1, c2 = os1.copy(), os2.copy()
        c1 = generate_offspring(os1, os2, c1)
        c2 = generate_offspring(os2, os1, c2)

        return c1, c2

    msc1, msc2 = mscross(P1[0], P2[0], rs)
    osc1, osc2 = oscross(P1[1], P2[1], jobSequence)
    C1, C2 = [msc1, osc1], [msc2, osc2]
    return C1, C2



def mutation(s, rs, allOT):
    def msmutation(ms, rs, allOT):
        '''
        Step1:在变异染色体中随机选择r个位置
        Step2:依次选择每一个位置,对每一个位置的机器设置为当前工序可选机器集合加工时间最短的机器
        ms:机器选择部分编码
        rs:list(range(工序总数))
        allOT:当前排序下所有工序的时间矩阵
        '''
        r = random.randint(1, len(rs))
        mutaionIndex = random.sample(rs, r)  
        for i in mutaionIndex:
            ms[i] = np.argmin(allOT[i])

        return ms

    def osmutation(os):
        '''
        随机选择两个位点进行转置
        '''
        osIndex = list(range(len(os)))
        a, b = sorted(random.sample(osIndex, 2))
        if b == len(osIndex):
            b += 1
        temp = list(os[a:b])
        temp.reverse()
        os[a:b] = temp
        return os

    s[0] = msmutation(s[0], rs, allOT)
    s[1] = osmutation(s[1])
    return s

def tournament(pop: list, fun_obj: np.array):
    parent_pop = []
    pop_size = fun_obj.shape[0]  
    suoyin = fun_obj.shape[1]  
    tour = 2
    a = pop_size // 2
    chrom_candidate = np.zeros((tour, 1))  
    chrom_rank = np.zeros((tour, 1))
    chrom_distance = np.zeros((tour, 1))
    chrom_obj = np.zeros((a, suoyin))  
    rank = suoyin - 5  
    distance = suoyin - 1  
    for i in range(a):
        for j in range(tour):
            if j == 0:
                chrom_candidate[j][0] = int(round(pop_size * random.random()))
                if chrom_candidate[j][0] == pop_size:
                    chrom_candidate[j][0] = pop_size - 1
            if j == 1:
                while True:
                    chrom_candidate[j][0] = int(round(pop_size * random.random()))
                    if chrom_candidate[j][0] != chrom_candidate[j - 1][0] and chrom_candidate[j][0] < pop_size:
                        break
                    
                    
                
                
                
                

        for j in range(tour):
            chrom_rank[j] = fun_obj[int(chrom_candidate[j][0])][rank]
            chrom_distance[j] = fun_obj[int(chrom_candidate[j][0])][distance]
        
        
        minchrom_candidate = np.where(chrom_rank == np.min(chrom_rank))[0]  
        
        if len(minchrom_candidate) != 1:
            maxchrom_candidate = \
            np.where(chrom_distance[minchrom_candidate] == np.max(chrom_distance[minchrom_candidate]))[0]
            chrom_obj[i, :] = fun_obj[int(chrom_candidate[minchrom_candidate[maxchrom_candidate[0]]][0]), :]
            parent_pop.append(pop[int(chrom_candidate[minchrom_candidate[maxchrom_candidate[0]]][0])])
        else:
            chrom_obj[i, :] = fun_obj[int(chrom_candidate[minchrom_candidate[0]][0]), :]
            parent_pop.append(pop[int(chrom_candidate[minchrom_candidate[0]][0])])

    return parent_pop, chrom_obj

def crowding_sort(pareto_ranks, pop, fun_obj, f_num, flag):
    fun_obj_crowd = np.zeros((fun_obj.shape[0], fun_obj.shape[1]))  
    pop_crowd = []  
    
    
    

    sort_index = np.argsort(fun_obj[:, 3])  
    fun_obj = fun_obj[sort_index]
    
    
    temp = []  
    for i in range(len(sort_index)):
        if flag == 0:
            temp.append(pop[sort_index[i]])
        elif flag == 1:
            temp.append(pop[sort_index[i]][0])
    
    current_index = 0
    for pareto_rank in range(len(pareto_ranks)):
        
        nd = np.zeros(len(pareto_ranks[pareto_rank].ss))
        y = np.zeros((len(pareto_ranks[pareto_rank].ss), fun_obj.shape[1]))  
        previous_index = current_index
        flag = sum([len(pareto_ranks[m].ss) for m in range(pareto_rank)])
        for i in range(len(pareto_ranks[pareto_rank].ss)):
            if pareto_rank == 0:
                y[i, :] = fun_obj[i, :]  
            else:
                y[i, :] = fun_obj[i + flag, :]

        current_index += len(pareto_ranks[pareto_rank].ss)
        chrom = temp[previous_index:current_index]  
        
        for j in range(f_num):
            index = np.argsort(y[:, j])
            obj_sort = y[index]  
            
            objsort_chrom = []
            for k in range(len(index)):
                objsort_chrom.append(chrom[k])  
            
            fmin = obj_sort[0][j]
            fmax = obj_sort[len(index) - 1][j]
            
            y[index[0]][f_num + 1 + j] = np.inf
            y[index[len(index) - 1]][f_num + 1 + j] = np.inf
            
            for k in range(1, len(index) - 1):
                pre_f = obj_sort[k - 1][j]
                next_f = obj_sort[k + 1][j]
                if fmax == fmin:
                    y[index[k]][f_num + 1 + j] = np.inf
                else:
                    y[index[k]][f_num + 1 + j] = (next_f - pre_f) / (fmax - fmin)

        
        for j in range(f_num):
            nd += y[:, 4 + j]
        y[:, fun_obj.shape[1] - 1] = nd  
        fun_obj_crowd[previous_index:current_index] = y
        pop_crowd += chrom

    return pop_crowd, fun_obj_crowd




