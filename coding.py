import numpy as np
import random
from problem import Machine

def encoding(Jobs: list, os: np.array, machines_num: int, stype='GS'):
    ms = np.zeros(len(os), dtype=int)
    mt = np.zeros(machines_num, dtype=float)
    osRecord = 0
    for job in Jobs:
        if stype == 'RS':
            for i in range(job.osNum):
                ms[osRecord] = random.randint(0, len(job.minfo[i]) - 1)
                osRecord += 1
        else:
            for i in range(job.osNum):
                candM = job.minfo[i]
                tmpm = mt[candM-1] + job.tinfo[i]
                min_val = np.min(tmpm)
                min_index = np.argmin(tmpm)
                mt[candM[min_index]] = min_val
                ms[osRecord] = min_index
                osRecord += 1
            if stype == 'LS':
                mt = np.zeros(machines_num, dtype=float)
    indexList = list(range(len(os)))
    random.shuffle(indexList)
    return [ms, os[indexList]]

def decoding(s: list, Jobs: list, D: list, E: list, machines_num):
    ms, os = s[0], s[1]
    machines = [Machine(i) for i in range(machines_num)]
    osCumSum = list(np.cumsum([job.osNum for job in Jobs]))
    osCumSum.insert(0, 0)
    energy = 0
    total_D = 0
    total_E = 0
    for num in os:
        job = Jobs[num]
        if job.osCount == 0:
            jobms = ms[osCumSum[num]:osCumSum[num + 1]].copy()
            job.decode(jobms)

        mId = job.Jm[job.osCount]
        mot = job.T[job.osCount]
        moe = job.E[job.osCount]
        energy += (moe*mot)
        machines[mId-1].process(job, mot, moe)

    completionT = 0
    for machine in machines:
        if machine.processInfo and machine.processInfo[-1].endT > completionT:
            completionT = machine.processInfo[-1].endT

    for i in range(len(Jobs)):
        if Jobs[i].endTs[-1] > (Jobs[i].Due + Jobs[i].startTs[0]):
            total_D = total_D + (Jobs[i].endTs[-1] - Jobs[i].Due - Jobs[i].startTs[0])
        elif Jobs[i].endTs[-1] < Jobs[i].Ear:
            total_E += Jobs[i].Ear - Jobs[i].endTs[-1]
        else:
            continue
    for job in Jobs:
        job.reset()

    return machines, completionT, energy, total_E, total_D