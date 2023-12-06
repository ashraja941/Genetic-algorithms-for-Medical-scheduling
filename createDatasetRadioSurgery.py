import time
from faker import Faker
import random
import pandas as pd

fake = Faker()
#random.seed(time.time())
"""
Patient ID : ordered
Name : random
Treatment Time : random(30,120)
size : 1-4 smaller -> bigger
shape : 1-4 simple -> complex
Patient Priority : 1-4
Weights : 10,5,2,1
"""

Db = {}
data = []
treatmentTime = [60,90,120] #possible time slots

#create database
for pID in range(40):
    tempName = fake.name()
    # tempTime = random.choice(treatmentTime) 
    tempTime = random.randint(30,120)
    tempShape = random.randint(0,4)
    tempSize = random.randint(0,4)

    normalizedScore = ((tempShape/4) + (tempSize/4))/2

    data.append({"P ID" : pID,"Name" : tempName, "Time": tempTime, "Shape": tempShape, "Size" : tempSize, "Priority" : -1, "Weight" : -1,"Score" : normalizedScore})

df = pd.DataFrame.from_dict(data)
df["Priority"] = pd.qcut(df["Score"], q = 4, labels=False)

for i in range(4):
    df.loc[df["Priority"] == i,'Weight'] = 10 if i == 0 else 5 if i == 1 else 2 if i == 2 else 1
print(df)

df.to_json(path_or_buf="D:\Projects\MAJOR PROJECT\\test2.json")