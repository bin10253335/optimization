import numpy as np
import matplotlib.pyplot as plt


class Job:
    def __init__(self, jobId: int, minfo: list, tinfo: list, done_info: list, D, E, productType: str) -> None:
        self.id = jobId  
        self.osNum = len(minfo)  
        self.minfo = self.list_to_array(minfo)  
        self.tinfo = self.list_to_array(tinfo, infotype='t')  
        self.Due = D
        self.Ear = E
        
        
        self.energy_info = self.list_to_array(done_info) 
        self.productType = productType  
        self.osCount = 0  
        self.startTs = [0]  
        self.endTs = [0]  

    def list_to_array(self, listinfo: list, infotype='m'):
        info = listinfo.copy()
        if infotype == 'm':
            for i in range(self.osNum):
                info[i] = np.array(info[i]) - 1  
        else:
            for i in range(self.osNum):
                info[i] = np.array(info[i])
        return info

    def decode(self, ms: np.array):
        '''解码机器顺序矩阵和时间矩阵顺序'''
        Jm, T, E = [], [], []
        for i in range(self.osNum):
            Jm.append(self.minfo[i][ms[i]])
            T.append(self.tinfo[i][ms[i]])
            E.append(self.energy_info[i][ms[i]])
        self.Jm, self.T, self.E  = Jm, T, E
        return Jm, T, E

    def updateT(self, startT, endT):

        self.startTs.append(startT)
        self.endTs.append(endT)
        self.osCount += 1

    def reset(self):
        '''重置时间'''
        self.startTs = [0]
        self.endTs = [0]
        self.osCount = 0


class WorkInfo:
    def __init__(self, startT=None, duration=None, endT=None, jobid=None, joboreder=None, jobtype=None):
        self.startT = startT  
        self.duration = duration  
        self.endT = endT  
        self.jobid = jobid  
        self.joborder = joboreder  
        self.jobtype = jobtype  

    def __str__(self):
        return f"开始加工时间:{self.startT},加工持续时间:{self.duration},加工结束时间:{self.endT},工件编号:{self.jobid},工件次序:{self.joborder},工件类属产品:{self.jobtype}"


class Machine:
    def __init__(self, id):
        self.id = id  
        self.processInfo = []  

    def process(self, job, mot, moe):
        '''工件工序的加工操作
        jobid:工件编号
        mot:机器加工该工件当前工序的耗时
        moe:机器加工该工件当前工序的能耗
        '''
        if self.processInfo:
            startT = max(job.endTs[-1], self.processInfo[-1].endT)
            if startT == self.processInfo[-1].endT and job.productType == self.processInfo[-1].jobtype:  
                mot *= 0.75
                moe *= 0.80
        else:
            startT = job.endTs[-1]
        endT = startT + mot
        job.updateT(startT, endT)
        workinfo = WorkInfo(startT, mot, endT, jobid=job.id, joboreder=job.osCount, jobtype=job.productType)
        self.processInfo.append(workinfo)

    def reset(self):
        '''重置'''
        self.processInfo = []



def visualization(machines, completionT, EC, TE, TD, ptocolor, machines_num, gen):
    '''绘制甘特图
    '''
    plt.figure(figsize=(len(machines) * 10, len(machines)))
    for i in range(machines_num):
        for osProcessInfo in machines[i].processInfo:
            plt.barh(i, width=osProcessInfo.duration, height=1, left=osProcessInfo.startT, align='center',
                     color=ptocolor[osProcessInfo.jobtype], edgecolor='k')
            plt.text(x=osProcessInfo.startT + osProcessInfo.duration / 2, y=i,
                     s=f"{osProcessInfo.jobid}",
                     horizontalalignment='center', fontsize=7)

    plt.xticks(fontsize=15)
    plt.yticks(list(range(machines_num)), ['machine%d' % i for i in range(machines_num)], fontsize=15)
    plt.tick_params(direction='in')
    plt.title(f'Makespan: {completionT:.2f}, EC: {EC:.2f}, TE: {TE:.2f}, TD: {TD:.2f}', fontsize=15)
    plt.show()