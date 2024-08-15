# extend_CO2 env51 analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import csv

trial_num = 53

timestep = 24
num_agent = 2

data_path = "/home/sgs650/PycharmProjects/pythonProject/BRL/energyNetwork/test/env_data/"
for j in range(num_agent):
    globals()[f"csv_{j}"] = pd.read_csv(f"/home/sgs650/PycharmProjects/pythonProject/BRL/energyNetwork/test/data{j}.csv")
# csv_data = "/home/sgs650/PycharmProjects/pythonProject/BRL/energyNetwork/data.csv"
# csv = pd.read_csv(csv_data)
# csv = open(csv_data, "r")
# OS = pd.read_table(data_path+f"OSproduction(env{trial_num}).txt")

a = open(data_path+f"OSproduction(test{trial_num}).txt", "r")
b = open(data_path+f"purchaseG(test{trial_num}).txt", "r")
c = open(data_path+f"purchaseB(test{trial_num}).txt", "r")
d = open(data_path+f"purchaseX(test{trial_num}).txt", "r")
e = open(data_path+f"diffSOH(test{trial_num}).txt", "r")
f = open(data_path+f"SOH(test{trial_num}).txt", "r")
g = open(data_path+f"action_HDS(test{trial_num}).txt", "r")
h = open(data_path+f"profit(test{trial_num}).txt", "r")
k = open(data_path+f"cost(test{trial_num}).txt", "r")
l = open(data_path+f"hydroRemain(test{trial_num}).txt", "r")

w = open(data_path+f"G_CO2_dist_total(test{trial_num}).txt", "r")
x = open(data_path+f"G_CO2_central_total(test{trial_num}).txt", "r")
y = open(data_path+f"B_CO2_total(test{trial_num}).txt", "r")
z = open(data_path+f"X_CO2_total(test{trial_num}).txt", "r")




aa=[]
bb=[]
cc=[]
dd=[]
ee=[]
ff=[]
gg=[]
hh=[]
kk=[]
ll=[]
ww=[]
xx=[]
yy=[]
zz=[]




for i in a:
    aa.extend([json.loads(i)])
for i in b:
    bb.extend([json.loads(i)])
for i in c:
    cc.extend([json.loads(i)])
for i in d:
    dd.extend([json.loads(i)])
for i in e:
    ee.extend([json.loads(i)])
    # ee.extend([json.loads(i, strict=False)])
for i in f:
    ff.extend([json.loads(i)])
for i in g:
    gg.extend([json.loads(i)])
for i in h:
    hh.extend([json.loads(i)])
for i in k:
    kk.extend([json.loads(i)])
for i in l:
    ll.extend([json.loads(i)])
for i in w:
    ww.extend([json.loads(i)])
for i in x:
    xx.extend([json.loads(i)])
for i in y:
    yy.extend([json.loads(i)])
for i in z:
    zz.extend([json.loads(i)])


# i = len(aa)-1 # last episode
i=49760
print(i)
aaa = aa[i]
bbb = bb[i]
ccc = cc[i]
ddd = dd[i]
eee = ee[i]
fff = ff[i]
ggg = gg[i]
hhh = hh[i]
kkk = kk[i]
lll = ll[i]
www = ww[i]
xxx = xx[i]
yyy = yy[i]
zzz = zz[i]


for i in range(timestep):
    for j in range(num_agent):
        if i == 0:
            globals()['aaa_{}'.format(j)] = [aaa[i][j]]
            globals()['bbb_{}'.format(j)] = [bbb[i][j]]
            globals()['ccc_{}'.format(j)] = [ccc[i][j]]
            globals()['ddd_{}'.format(j)] = [ddd[i][j]]
            globals()['eee_{}'.format(j)] = [eee[i][j]]
            globals()['fff_{}'.format(j)] = [fff[i][j]]
            # globals()['ggg_{}'.format(j)] = [ggg[i][j]]
            # globals()['hhh_{}'.format(j)] = [hhh[i][j]]
            globals()['kkk_{}'.format(j)] = [kkk[i][j]]
            globals()['lll_{}'.format(j)] = [lll[i][j]]
            # globals()['www_{}'.format(j)] = [www[i][j]]
            # globals()['xxx_{}'.format(j)] = [xxx[i][j]]
            # globals()['yyy_{}'.format(j)] = [yyy[i][j]]
            # globals()['zzz_{}'.format(j)] = [zzz[i][j]]
        else:
            globals()['aaa_{}'.format(j)].append(aaa[i][j])
            globals()['bbb_{}'.format(j)].append(bbb[i][j])
            globals()['ccc_{}'.format(j)].append(ccc[i][j])
            globals()['ddd_{}'.format(j)].append(ddd[i][j])
            globals()['eee_{}'.format(j)].append(eee[i][j])
            globals()['fff_{}'.format(j)].append(fff[i][j])
            # globals()['ggg_{}'.format(j)].append(ggg[i][j])
            # globals()['hhh_{}'.format(j)].append(hhh[i][j])
            globals()['kkk_{}'.format(j)].append(kkk[i][j])
            globals()['lll_{}'.format(j)].append(lll[i][j])
            # globals()['www_{}'.format(j)].append(www[i][j])
            # globals()['xxx_{}'.format(j)].append(xxx[i][j])
            # globals()['yyy_{}'.format(j)].append(yyy[i][j])
            # globals()['zzz_{}'.format(j)].append(zzz[i][j])



for j in range(num_agent):
    globals()['total_{}'.format(j)] = [a+b+c+d-e for a,b,c,d,e in zip(globals()['aaa_{}'.format(j)], globals()['bbb_{}'.format(j)],
                                                                      globals()['ccc_{}'.format(j)], globals()['ddd_{}'.format(j)],
                                                                      globals()['eee_{}'.format(j)])]
    globals()['hload_{}'.format(j)] = list(np.array(globals()["csv_{}".format(j)]["hload"].tolist()))
    globals()['hload_{}'.format(j)].pop()



time = [i+1 for i in range(timestep)]


for j in range(num_agent):
    globals()[f"store_{j}"] = []
    globals()[f"withdrawn_{j}"] = []
    globals()[f"supply_{j}"] = []
    globals()[f"supply_p_{j}"] = []
    globals()[f"demand_diff_{j}"] = []
    for i in range(timestep):
        if globals()[f"eee_{j}"][i] >= 0:
            globals()[f"store_{j}"].append(-globals()[f"eee_{j}"][i])
            globals()[f"withdrawn_{j}"].append(0)
        else:
            globals()[f"withdrawn_{j}"].append(-globals()[f"eee_{j}"][i])
            globals()[f"store_{j}"].append(0)

        if globals()[f"lll_{j}"][i] >= 0:
            globals()[f"supply_{j}"].append(-globals()[f"hload_{j}"][i])
            globals()[f"supply_p_{j}"].append(globals()[f"hload_{j}"][i])
        else:
            globals()[f"supply_{j}"].append(- globals()[f"aaa_{j}"][i] - globals()[f"bbb_{j}"][i] - globals()[f"ccc_{j}"][i] - globals()[f"ddd_{j}"][i] - globals()[f"withdrawn_{j}"][i])
            globals()[f"supply_p_{j}"].append(globals()[f"aaa_{j}"][i] + globals()[f"bbb_{j}"][i] + globals()[f"ccc_{j}"][i] + globals()[f"ddd_{j}"][i] + globals()[f"withdrawn_{j}"][i])
        globals()[f"demand_diff_{j}"].append(globals()[f"supply_p_{j}"][i] - globals()[f"hload_{j}"][i])

# demand satisfaction graph_HRS0
# bar_width = 0.25
# plt.figure(1, figsize=(15,5))
# plt.bar(time, demand_diff_0)
# plt.title(f'demand satisfaction{trial_num} for HRS0')

# bottom = np.add(aaa_0,bbb_0, ccc_0)
# 
# Hydrogen balance_HRS0
plt.figure(2, figsize=(15,5))
plt.bar(time, aaa_0, label='On-site_Production')
plt.bar(time, bbb_0, bottom=aaa_0, color='g', label='Purchase_Green')
plt.bar(time, ccc_0, bottom = bbb_0, color='r', label='Purchase_Blue')
plt.bar(time, ddd_0, bottom=ccc_0, color='b', label='Purchase_Gray')
plt.bar(time, withdrawn_0, color='y', label='Withdrawn')
plt.bar(time, supply_0, label='demand')
plt.bar(time, store_0, bottom=supply_0, label='store')
plt.title(f'hydrogen balance{trial_num} for HRS0')
plt.legend()

plt.show()

# SOH level_HRS0
plt.figure(3, figsize=(15,10))
plt.plot(time, eee_0, 'o-')
plt.title(f'SOH in Tank{trial_num} for HRS0')
#
# # Purchasing H2 tendency graph_HRS0
# plt.figure(4, figsize=(15,10))
# plt.plot(time, aaa_0, 'o-', color='p')
# plt.plot(time, bbb_0, 'o-', color='b')
# plt.plot(time, ccc_0, 'o-', color='r')
# plt.title(f'Purchasing hydrogen tendency{trial_num} for HRS0')
#
#
# demand satisfaction graph_HRS1
# plt.figure(5, figsize=(15,5))
# plt.bar(time, demand_diff_1)
# plt.title(f'demand satisfaction{trial_num} for HRS1')
# # bottom = np.add(aaa_1, bbb_1, ccc_1)
# plt.show()
#
# Hydrogen balance_HRS1
plt.figure(6, figsize=(15,5))
plt.bar(time, aaa_1, label='On-site_Production')
plt.bar(time, bbb_1, color='g', label='Purchase_Green')
plt.bar(time, ccc_1, color='r', label='Purchase_Blue')
plt.bar(time, ddd_1, color='b', label='Purchase_Gray')
plt.bar(time, withdrawn_1, color='y', label='Withdrawn')
plt.bar(time, supply_1, label='demand')
plt.bar(time, store_1, bottom=supply_1, label='store')
plt.title(f'hydrogen balance{trial_num} for HRS1')
plt.legend()
plt.show()


# # SOH level_HRS1
# plt.figure(7, figsize=(15,10))
# plt.plot(time, eee_1, 'o-')
# plt.title(f'SOH in Tank{trial_num} for HRS1')
# 
# # Purchasing H2 tendency graph_HRS1
# plt.figure(8, figsize=(15,10))
# plt.plot(time, aaa_1, 'o-', color='g')
# plt.plot(time, bbb_1, 'o-', color='b')
# plt.plot(time, ccc_1, 'o-', color='r')
# plt.title(f'Purchasing hydrogen tendency{trial_num} for HRS1')
# 
# 
# plt.show()

# result
# G_pie = ggg[3]
# B_pie = ggg[4]
# X_pie = ggg[5]
# G_CO2_dist_total = www
# G_CO2_central_total = xxx
# B_CO2_total = yyy
# X_CO2_total = zzz
# # G_CO2_dist_total = aaa * 2.89
#
# print(f"Selling Green Hydrogen price to HRS from HDS is {G_pie}")
# print(f"Selling Blue Hydrogen price to HRS from HDS is {B_pie}")
# print(f"Selling Gray Hydrogen price to HRS from HDS is {X_pie}")
# print(f"HDS gets a profit of {hhh}")
# print(f"HRS0 cost is {sum(kkk_0)} and HRS1 cost is {sum(kkk_1)}")
# print(f"Green_CO2_dist: {www}")
# print(f"Green_CO2_central: {xxx}")
# print(f"Blue_CO2: {yyy}")
# print(f"Gray_CO2: {zzz}")
# print(f"HRS0 Green is {sum(bbb_0)} and HRS1 Green is {sum(bbb_1)}")
# print(f"HRS0 Blue is {sum(ccc_0)} and HRS1 Blue is {sum(ccc_1)}")
# print(f"HRS0 Gray is {sum(ddd_0)} and HRS1 Gray is {sum(ddd_1)}")
# print(f"HRS0 OS is {sum(aaa_0)} and HRS1 OS is {sum(aaa_1)}")
print(f"HRS0 supply rate is {1+sum(demand_diff_0)/95} and HRS1 supply rate is {1+sum(demand_diff_1)/71}")
