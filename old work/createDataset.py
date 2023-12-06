import time
from faker import Faker
import random
import pandas as pd

fake = Faker()
#random.seed(time.time())
"""
Patient ID
Name 
Treatment Time : 20,30,45
MADRS : 0-60
DASS-D : 0-42
DASS-A : 0-42
DASS-S : 0-42
Patient Priority : 1-4
Weights : 10,5,2,1
"""

Db = {}
data = []
treatmentTime = [20,30,45]

#create database
for pID in range(4):
    tempName = fake.name()
    tempTime = random.choice(treatmentTime) 
    tempMADRS = random.randint(0,60)
    tempDASSD = random.randint(0,42)
    tempDASSA = random.randint(0,42)
    tempDASSS = random.randint(0,42)

    normalizedScore = ((tempMADRS/60) + (tempDASSD/42) + (tempDASSA/42) + (tempDASSS/42))/4

    data.append({"P ID" : pID,"Name" : tempName, "Time": tempTime, "MADRS": tempMADRS, "DASS-D" : tempDASSD, "DASS-A" : tempDASSA, "DASS-S" : tempDASSS, "Priority" : -1, "Weight" : -1,"Score" : normalizedScore})

df = pd.DataFrame.from_dict(data)
df["Priority"] = pd.qcut(df["Score"], q = 4, labels=False)

for i in range(4):
    df.loc[df["Priority"] == i,'Weight'] = 10 if i == 0 else 5 if i == 1 else 2 if i == 2 else 1
print(df)

df.to_json(path_or_buf="D:\\projects\\nic genetic algo project\\test.json")