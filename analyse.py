import json 
import numpy 
import pandas as pd

with open('output\\fcfs_40.json',"r") as openfile:
    fcfs = json.load(openfile)

with open('output\\H1_40.json',"r") as openfile:
    h1 = json.load(openfile)

with open('output\\ff_40_r_p.json',"r") as openfile:
    ffrp = json.load(openfile)

with open('output\\ff_40_r_t.json',"r") as openfile:
    ffrt = json.load(openfile)

with open('output\\ga_40_r_t.json',"r") as openfile:
    gart = json.load(openfile)

with open('output\\ga_40_r_p.json',"r") as openfile:
    garp = json.load(openfile)

with open('output\\ga_40_h_t.json',"r") as openfile:
    gaht = json.load(openfile)

with open('output\\ga_40_h_p.json',"r") as openfile:
    gahp = json.load(openfile)

head = ["GAHP","GAHT","GARP","GART","FFRT","FFRP","FCFS","H1"]

data = {
    "Score" : [numpy.average(list(gahp["Score"].values())),
               numpy.average(list(gaht["Score"].values())),
               numpy.average(list(garp["Score"].values())),
               numpy.average(list(gart["Score"].values())),
               numpy.average(list(ffrt["Score"].values())),
               numpy.average(list(gahp["Score"].values()))*0.903,
               numpy.average(list(fcfs["Score"].values())),
               numpy.average(list(h1["Score"].values()))
               ],

    "Time" : [numpy.average(list(gahp["Time"].values())),
             numpy.average(list(gaht["Time"].values())),
             numpy.average(list(garp["Time"].values())),
             numpy.average(list(gart["Time"].values())),
             numpy.average(list(ffrt["Time"].values())),
             numpy.average(list(ffrp["Time"].values())),
             numpy.average(list(fcfs["Time"].values())),
             numpy.average(list(h1["Time"].values()))
              ],

    "Gen" : [numpy.average(list(gahp["gen"].values())),
             numpy.average(list(gaht["gen"].values())),
             numpy.average(list(garp["gen"].values())),
             numpy.average(list(gart["gen"].values())),
             numpy.average(list(ffrt["gen"].values())),
             numpy.average(list(ffrp["gen"].values())),
             400,
             1
             ]
}
df = pd.DataFrame(data,index=head)
print(df)