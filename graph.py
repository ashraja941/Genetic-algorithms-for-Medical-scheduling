import pandas as pd
import matplotlib.pyplot as plt
import json

with open('test2.json',"r") as openfile:
    test = json.load(openfile)

df = pd.DataFrame.from_dict(test, orient="index")
df = df.T
#print(df)
#df = df.sort_values(by=['Time'])
#df.plot.pie(y='Time', figsize=(5, 5), autopct='%1.1f%%', startangle=90)
# plt.show()
print(df)

ax = plt.gca()
df.plot(kind='line',
        # x='P ID',
        y='Time',
        color='green', ax=ax)

plt.title('LinePlots')
plt.show()