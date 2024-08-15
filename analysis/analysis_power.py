# extend env analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import csv

trial_num = 1

timestep = 24
num_agent = 2

data_path = "/home/sgs650/PycharmProjects/pythonProject/BRL/energyNetwork/test/env_data/"
for j in range(num_agent):
    globals()[f"csv_{j}"] = pd.read_csv(f"/home/sgs650/PycharmProjects/pythonProject/BRL/energyNetwork/test/data{j}.csv")
# csv_data = "/home/doeun/PycharmProjects/pythonProject/BRL/energyNetwork/data.csv"
# csv = pd.read_csv(csv_data)
# csv = open(csv_data, "r")
# OS = pd.read_table(data_path+f"OSproduction(env{trial_num}).txt")

a = open(data_path+f"powerMG(test{trial_num}).txt", "r")
b = open(data_path+f"powerDG(test{trial_num}).txt", "r")
# c = open(data_path+f"powerPV(test{trial_num}).txt", "r")



aa=[]
bb=[]
# cc=[]



for i in a:
    aa.extend([json.loads(i)])
for i in b:
    bb.extend([json.loads(i)])
# for i in c:
#     cc.extend([json.loads(i)])



# i = len(aa)-1 # last episode
i=30000
print(i)
aaa = aa[i]
bbb = bb[i]
# ccc = cc[i]

# i=0
# print(i)
# ccc = cc[i]


for i in range(timestep):
    for j in range(num_agent):
        if i == 0:
            globals()['aaa_{}'.format(j)] = [aaa[i][j]]
            globals()['bbb_{}'.format(j)] = [bbb[i][j]]
            # globals()['ccc_{}'.format(j)] = [ccc[i][j]]
        else:
            globals()['aaa_{}'.format(j)].append(aaa[i][j])
            globals()['bbb_{}'.format(j)].append(bbb[i][j])
            # globals()['ccc_{}'.format(j)].append(ccc[i][j])
         

# total = [x+y-z for x, y, z in zip(aaa,bbb,ccc)]



# for j in range(num_agent):
#     globals()['total_{}'.format(j)] = [x+y-z for x,y,z in zip(globals()['aaa_{}'.format(j)], globals()['bbb_{}'.format(j)], globals()['ccc_{}'.format(j)])]
#     globals()['hload_{}'.format(j)] = list(np.array(globals()["csv_{}".format(j)]["hload"].tolist()))
#     globals()['hload_{}'.format(j)].pop()

# time = list(np.array(csv_0["time"][0:timestep].tolist()))
time = [i+1 for i in range(timestep)]
# hload_0 = list(np.array(csv_0["hload"].tolist()))
# hload_0.pop()
# rate = [] # demand confidence
# for i in range(timestep):
#     if hload[i] != 0:
#         rate.append((hhh[i]-ccc[i])/hload[i]*100)
#     else:
#         rate.append(0)

# for j in range(num_agent):
#     globals()[f"store_{j}"] = []
#     globals()[f"withdrawn_{j}"] = []
#     globals()[f"supply_{j}"] = []
#     globals()[f"supply_p_{j}"] = []
#     globals()[f"demand_diff_{j}"] = []
#     for i in range(timestep):
#         if globals()[f"ccc_{j}"][i] >= 0:
#             globals()[f"store_{j}"].append(-globals()[f"ccc_{j}"][i])
#             globals()[f"withdrawn_{j}"].append(0)
#         else:
#             globals()[f"withdrawn_{j}"].append(-globals()[f"ccc_{j}"][i])
#             globals()[f"store_{j}"].append(0)
#
#         if globals()[f"hhh_{j}"][i] >= 0:
#             globals()[f"supply_{j}"].append(-globals()[f"hload_{j}"][i])
#             globals()[f"supply_p_{j}"].append(globals()[f"hload_{j}"][i])
#         else:
#             globals()[f"supply_{j}"].append(-globals()[f"aaa_{j}"][i] - globals()[f"bbb_{j}"][i] - globals()[f"withdrawn_{j}"][i])
#             globals()[f"supply_p_{j}"].append(globals()[f"aaa_{j}"][i] + globals()[f"bbb_{j}"][i] + globals()[f"withdrawn_{j}"][i])
#         globals()[f"demand_diff_{j}"].append(globals()[f"supply_p_{j}"][i] - globals()[f"hload_{j}"][i])


# supply=[]
# supply_p = []
# for i in range(timestep):
#     if hhh[i] >= 0:
#         supply.append(-hload[i])
#         supply_p.append(hload[i])
#     else:
#         supply.append(-aaa[i]-bbb[i]-withdrawn[i])
#         supply_p.append(aaa[i]+bbb[i]+withdrawn[i])

# demand_diff = []
# for i in range(timestep):
#     demand_diff.append(supply_p[i]-hload[i])

# hload_dff = []
# for i in range(timestep):
#     hload_dff[i] = supply[i]-hload[i]
# bar_width = 0.25
# graph
# plt.figure(1, figsize=(15,5))
# gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2,3], width_ratios=[12])
# plt.subplot(gs[0])
# plt.bar(time, rate)
# plt.bar(time,supply_p)
# plt.plot(time,hload)


plt.figure(1, figsize=(15,10))
plt.plot(time, aaa_0, 'o-', color='b')
plt.plot(time, bbb_0, 'o-', color='r')
# plt.plot(time, ccc_0, 'o-', color='g')
plt.title(f'power tendency{trial_num} for HRS1')

plt.show()

