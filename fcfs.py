import json 
import random
import numpy as np
import time
import pandas as pd

def initialize_order(data):
    #transfer the data to an array
    temp = []
    for n in data["P ID"]:
        temp.append(n) #possibly convert to integer
    random.shuffle(temp)
    return temp

def fcfs(data,order,M):
    J = int(max(data["P ID"].values())) + 1

    totalTime = [0]*M
    remainingTime = [0]*M
    convertedOrder = [[] for j in range(M)]


    #initialize all the machine with first patient
    for i in range(M):
        totalTime[i] += data["Time"][order[i]]
        remainingTime[i] += data["Time"][order[i]]
        convertedOrder[i].append(int(order[i]))

    for order_i in order[M:]:
        #subract the time that is passed to make the machine free
        maskedRT = np.ma.masked_equal(remainingTime,0,copy=False)
        minVal = maskedRT.min()
        for i in range(M):
            remainingTime[i] -= minVal
        
        """
        #testing
        print(minVal)
        print(remainingTime)
        print(totalTime)        
        """

        #if machine is free add the time 
        for i in range(M):
            if remainingTime[i] == 0:
                totalTime[i] += data["Time"][order_i]
                remainingTime[i] += data["Time"][order_i]
                #add the stuff to the machine here
                convertedOrder[i].append(int(order_i))
                break

    #add the extra numbers to fit the gene description
    j = J
    #print(J)
    for i in range(M):
        while len(convertedOrder[i]) < J:
            convertedOrder[i].append(j)
            j+=1
    finalOrder = []
    for i in range(M):
        finalOrder.extend(convertedOrder[i])

    OptimalTime = 1000
    #print(finalOrder)
    return OptimalTime / max(totalTime),finalOrder

def sWPT(data,order):
    flowTime = [0]*len(order)
    for i in range(len(order)):
        flowTime[i] = (order[i],float(data["Weight"][order[i]]) / data["Time"][order[i]])
    #print(flowTime)
    flowTime.sort(key= lambda x: x[1])
    #print(flowTime)
    #return list(zip(*flowTime))[0]
    return list(a for a in list(zip(*flowTime))[0])

if __name__ == "__main__":
    #load json
    f = open('test2.json')
    data = json.load(f)
    
    array = []
    df = {}

    for iteration in range(100):
        prevTime = time.time()
        output=[0]*10
        for i in range(10):
            order = initialize_order(data)
            output[i] = fcfs(data,order,3)
        curTime = time.time()
        array.append({"Time" : curTime - prevTime,"Score" : min(output)[0]})
    df = pd.DataFrame.from_dict(array)
    #df.to_json(path_or_buf="D:\Projects\MAJOR PROJECT\\output\\fcfs_40.json")

    array2 = []
    df2 = {}
    prevTime = time.time()
    order = initialize_order(data)
    #print(sWPT(data,order))
    print(fcfs(data,sWPT(data,order),3))
    curTime = time.time()
    array2.append({"Time" : curTime - prevTime,"Score" : min(output)[0]})
    df2 = pd.DataFrame.from_dict(array2)
    #df2.to_json(path_or_buf="D:\Projects\MAJOR PROJECT\\output\\h1_40.json")




    """    
    order = initialize_order(data)
    print(order)
    #print(sWPT(data,order))
    print("total time : ", fcfs(data,order,3))

    checkSimilarity = [0]*100
    for j in range(100):
        outputTime=[0]*100
        order = [0]*100
        for i in range(100):
            order[i] = initialize_order(data)
            outputTime[i] = fcfs(data,order[i])
        
        arrayTemp = np.array(outputTime)
        #print("totat Time : ",min(outputTime), order[arrayTemp.argmin()])
        str1 = ""
        for ele in order[arrayTemp.argmin()]:
            str1 += ele
        checkSimilarity[j] = str1

    if (len(checkSimilarity) == len(set(checkSimilarity))):
        print("unique")
    else:
        print("not unique")
    """