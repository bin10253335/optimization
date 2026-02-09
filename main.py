import random
import copy
from problem import Job, visualization
from coding import encoding, decoding
from mpl_toolkits.mplot3d import Axes3D
from Cross_machine import machine_cross
from GA import *
from pareto_sort import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
######## generate instance ######
random.seed()
machines_num = 10
jobs_num = 15
machine_list = list(range(1, machines_num + 1))
products = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']  # 设置每一种产品对应一种颜色
colors = ['r', 'slateblue', 'orange', 'c', 'm', 'y', 'burlywood', 'AliceBlue', 'tan', 'aliceblue', 'darkgreen', 'brown', 'royalblue', 'olive', 'olivedrab', 'chocolate', 'saddlebrown', 'peru', 'darkslateblue','Beige']
ptocolor = dict(zip(products, colors))
Jobs = []
Joblists = []
Ts = []
Es = []
T_ijave = []
########################################
####### setting benchmark instances#######
#parameters = parse(filename)
#processing information setting
for i in range(jobs_num):
    Tad = []
    for j in range(Jobs[i].osNum):
        T_ijk = [k for k in Ts[i][j] if k != -1]
        Tad.append(sum(T_ijk) / len(T_ijk))
    T_ijave.append(sum(Tad))
D = [int(T_ijave[i] * 1.5) for i in range(jobs_num)]
E = [D[i] for i in range(len(D))]
###################################################
for i in range(jobs_num):
    onum = random.randint(3, 7)  #operations
    candMs = []
    candTs = []
    candEs = []
    Tad = []
    for j in range(onum):
        mnum = random.randint(2, machines_num)
        candm = random.sample(machine_list, mnum)
        candMs.append(candm)
        #operations time
        candt = [random.randint(6, 36) for _ in candm]
        candTs.append(candt)
        #operations energy
        cande = [random.randint(8, 10) for _ in candm]
        candEs.append(cande)
    Joblists.append(candMs)
    Ts.append(candTs)
    Es.append(candEs)
    for j in range(onum):
        T_ijk = [k for k in Ts[i][j] if k != -1]
        Tad.append(sum(T_ijk) / len(T_ijk))
    T_ijave.append(sum(Tad))
    D = int(T_ijave[i] * 1.5)
    E = T_ijave[i]
    productType = products[random.randint(0, len(products) - 1)]
    Jobs.append(Job(i + 1, candMs, candTs, candEs, D, E, productType))
###################################################
jobSequence = list(range(jobs_num))
allOT = []
for job in Jobs:
    allOT.extend(job.tinfo)
os = []
for i, job in enumerate(Jobs):
    for j in range(job.osNum):
        os.append(i)
os = np.array(os)
rs = list(range(len(os)))

# parameters
popSize = 100  # population_size
maxGen = 200  # iteration_num
pc = 0.75  # 交crossover
pm = 0.2  # mutation

# initialization population
population = initializePopulation(Jobs, os, machines_num, popSize)
for individual in population:
    if individual.fit == 0:
    #if individual.fit is None:
        _, individual.fit, individual.energy, individual.Ear, individual.Due = decoding(individual.s, Jobs, D, E, machines_num)
        individual.DE = 0.45 * individual.Ear + 0.50 * individual.Due
bestS = copy.deepcopy(getbest(population))
bestfits = [bestS.Due]
gen = 0
#sorting
f_num = 3
pareto_ranks, fun_obj = non_dominate(population, f_num)
crowd_pop, obj_crowd = crowding_sort(pareto_ranks, population, fun_obj, f_num, 0)
########################################
q = Q()#initialization Q table
Differ = 0
while gen < maxGen:
    print("gen:", gen)
    Differ = 0
    fitness = [(1/individual.fit+1/individual.energy+1/individual.DE)/3 for individual in population]
    cumSum = np.cumsum(fitness)
    fitnessratio = cumSum / cumSum[-1]
    parents = []
    for i in range(popSize):
        parents.append(selection(population, fitnessratio))
    population = parents.copy()
    crowd_pop, obj_crowd = crowding_sort(pareto_ranks, population, fun_obj, f_num)
    parent1, parent_obj1 = tournament(crowd_pop, obj_crowd)
    parent2, parent_obj2 = tournament(crowd_pop, obj_crowd)
    population = parent1 + parent2
    fun_obj = np.vstack((parent_obj1, parent_obj2))
    parent_cross, offspring_cross, off_obj = pareto_cross(population, fun_obj, Jobs, D, E, machines_num)
    population = parent_cross + offspring_cross
    if gen == 0:
        Differ = 0
    else:
        Differ = Differ
    population, population_obj, action, lm, F_Dgen = Qlearning(population, fun_obj, 0.7, q, Differ, rs, jobSequence, Jobs, D, E, machines_num, allOT)
    Differ = F_Dgen
    #update Q
    reward, state, state_t = get_reward(population, population_obj, lm, Differ, q)
    update_qtable(q, state, state_t, action, reward)
    population, population_obj = machine_cross(population, population_obj, Jobs, D, E, machines_num)

    pareto_ranks, fun_obj = non_dominate([p[0] for p in population], f_num)

    population, fun_obj = crowding_sort(pareto_ranks, population, fun_obj, f_num, 1)
    crowd_pop, obj_crowd = crowding_sort(pareto_ranks, population, fun_obj, f_num)
    crowd_pop, obj_crowd = elitism(popSize, crowd_pop, obj_crowd, pareto_ranks)

    curbestS = getbest(population)
    if curbestS.Due <= bestS.Due:
        bestS = copy.deepcopy(curbestS)
    else:
        population[len(population)-1] = copy.deepcopy(bestS)
    bestfits.append(bestS.Due)
    machines, completionT, energy, total_E, total_D = decoding(bestS.s, Jobs, D, E, machines_num)
    visualization(machines, completionT, energy, total_E, total_D, ptocolor, machines_num, gen)
    gen += 1
machines, completionT, energy, total_E, total_D = decoding(bestS.s, Jobs, D, E, machines_num)
plt.plot(bestfits)
visualization(machines, completionT, energy, total_E, total_D, ptocolor, machines_num, gen)