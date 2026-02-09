#!/usr/bin/env python
from keras.src.layers import average


def parse(path):
    file = open(path, 'r')
    #file = open('E:/360Download/flexible-job-shop-master/test_data/Brandimarte_Data/Text/Mk02.fjs', 'r')
    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))
    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []
    pro_etime = []#record all jobs estimated processing time
    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))
        #print(len(currentLineValues))
        operations = []
        operation_time = []###################################### record average processing time
        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j+1

            operation = []
            op_time = []########################################record each operation's processing time
            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})
                op_time.append(processingTime)
            average_time = sum(op_time) / len(op_time)###########################计算每道工序所需要的平均加工时间
            operations.append(operation)
            operation_time.append(average_time)##################################
        pro_etime.append(sum(operation_time))#################################
        jobs.append(operations)
    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs}, pro_etime