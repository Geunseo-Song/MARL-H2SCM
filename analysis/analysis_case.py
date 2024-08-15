# extend_CO2 env51 analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import json
import csv

trial_num = 50

timestep = 24
num_agent = 2

data_path = "/home/sgs650/MARLlib/CaseStudy/env_data/"
for j in range(num_agent):
    globals()[f"csv_{j}"] = pd.read_csv(f"/home/sgs650/MARLlib/CaseStudy/data{j}.csv")
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






n=42560
# hydrogen price
ggg = gg[n]
G_pie = ggg[3]
B_pie = ggg[4]
X_pie = ggg[5]
print(f"Selling Green Hydrogen price to HRS from HDS is {G_pie}")
print(f"Selling Blue Hydrogen price to HRS from HDS is {B_pie}")
print(f"Selling Gray Hydrogen price to HRS from HDS is {X_pie}")


# profit
profit = []
for i in range(n-5000,n):
    hhh = hh[i]
    profit += hhh
sum_profit = sum(profit)
# ** profit = list, sum_profit = value
max_profit = max(profit)
min_profit = min(profit)
print(f"----------------------------------------profit and operation cost------------------------------------------")
print(f"HDS gets a profit of {sum_profit/5000} max: {max_profit} min: {min_profit}")


# operation cost
cost_HRS0 = []
cost_HRS1 = []
for i in range(n-5000,n):
    kkk = kk[i]
    for i in range(timestep):
        for j in range(num_agent):
            if i == 0:
                globals()['kkk_{}'.format(j)] = [kkk[i][j]]
            else:
                globals()['kkk_{}'.format(j)].append(kkk[i][j])
        HRS0 = [sum(kkk_0)]
        HRS1 = [sum(kkk_1)]
    cost_HRS0 += HRS0
    cost_HRS1 += HRS1
sum_cost_HRS0 = sum(cost_HRS0)
max_cost_HRS0 = max(cost_HRS0)
min_cost_HRS0 = min(cost_HRS0)
sum_cost_HRS1 = sum(cost_HRS1)
max_cost_HRS1 = max(cost_HRS1)
min_cost_HRS1 = min(cost_HRS1)
print(f"HRS0 cost is {sum_cost_HRS0/5000} max: {max_cost_HRS0} min: {min_cost_HRS0}")
print(f"HRS1 cost is {sum_cost_HRS1/5000} max: {max_cost_HRS1} min: {min_cost_HRS1}")



# CO2
G_CO2_central_total = []
G_CO2_dist_total = []
B_CO2_total = []
X_CO2_total = []
for i in range(n-5000,n):
    www = ww[i]
    xxx = xx[i]
    yyy = yy[i]
    zzz = zz[i]
    www_w = sum(www, [])
    www_ww = [sum(www_w)]
    G_CO2_central_total += xxx[1]
    G_CO2_dist_total += www_ww
    B_CO2_total += yyy[1]
    X_CO2_total += zzz[1]

sum_G_CO2 = sum(G_CO2_central_total)
sum_G_CO2_dist = sum(G_CO2_dist_total)
sum_B_CO2 = sum(B_CO2_total)
sum_X_CO2 = sum(X_CO2_total)
max_G_CO2 = max(G_CO2_central_total)
max_G_CO2_dist = max(G_CO2_dist_total)
max_B_CO2 = max(B_CO2_total)
max_X_CO2 = max(X_CO2_total)
min_G_CO2 = min(G_CO2_central_total)
min_G_CO2_dist = min(G_CO2_dist_total)
min_B_CO2 = min(B_CO2_total)
min_X_CO2 = min(X_CO2_total)
print(f"----------------------------------------CO2 emission------------------------------------------")
print(f"Green CO2_central: {sum_G_CO2/10000} max: {max_G_CO2} min: {min_G_CO2}")
print(f"Green CO2_dist: {sum_G_CO2_dist/5000} max: {max_G_CO2_dist} min: {min_G_CO2_dist}")
print(f"Blue CO2: {sum_B_CO2/10000} max: {max_B_CO2} min: {min_B_CO2}")
print(f"Gray CO2: {sum_X_CO2/10000} max: {max_X_CO2} min: {min_X_CO2}")



# H2 purchase & on-site production
OS_H2_HRS0 = []
G_H2_HRS0 = []
G_H2_HRS1 = []
B_H2_HRS0 = []
B_H2_HRS1 = []
X_H2_HRS0 = []
X_H2_HRS1 = []
for i in range(n-5000,n):
    aaa = aa[i]
    bbb = bb[i]
    ccc = cc[i]
    ddd = dd[i]
    for i in range(timestep):
        for j in range(num_agent):
            if i == 0:
                globals()['aaa_{}'.format(j)] = [aaa[i][j]]
                globals()['bbb_{}'.format(j)] = [bbb[i][j]]
                globals()['ccc_{}'.format(j)] = [ccc[i][j]]
                globals()['ddd_{}'.format(j)] = [ddd[i][j]]
            else:
                globals()['aaa_{}'.format(j)].append(aaa[i][j])
                globals()['bbb_{}'.format(j)].append(bbb[i][j])
                globals()['ccc_{}'.format(j)].append(ccc[i][j])
                globals()['ddd_{}'.format(j)].append(ddd[i][j])
        OS_HRS0 = [sum(aaa_0)]
        G_HRS0 = [sum(bbb_0)]
        G_HRS1 = [sum(bbb_1)]
        B_HRS0 = [sum(ccc_0)]
        B_HRS1 = [sum(ccc_1)]
        X_HRS0 = [sum(ddd_0)]
        X_HRS1 = [sum(ddd_1)]
    OS_H2_HRS0 += OS_HRS0
    G_H2_HRS0 += G_HRS0
    G_H2_HRS1 += G_HRS1
    B_H2_HRS0 += B_HRS0
    B_H2_HRS1 += B_HRS1
    X_H2_HRS0 += X_HRS0
    X_H2_HRS1 += X_HRS1
sum_OS_HRS0 = sum(OS_H2_HRS0)
max_OS_HRS0 = max(OS_H2_HRS0)
min_OS_HRS0 = min(OS_H2_HRS0)
sum_G_HRS0 = sum(G_H2_HRS0)
max_G_HRS0 = max(G_H2_HRS0)
min_G_HRS0 = min(G_H2_HRS0)
sum_G_HRS1 = sum(G_H2_HRS1)
max_G_HRS1 = max(G_H2_HRS1)
min_G_HRS1 = min(G_H2_HRS1)
sum_B_HRS0 = sum(B_H2_HRS0)
max_B_HRS0 = max(B_H2_HRS0)
min_B_HRS0 = min(B_H2_HRS0)
sum_B_HRS1 = sum(B_H2_HRS1)
max_B_HRS1 = max(B_H2_HRS1)
min_B_HRS1 = min(B_H2_HRS1)
sum_X_HRS0 = sum(X_H2_HRS0)
max_X_HRS0 = max(X_H2_HRS0)
min_X_HRS0 = min(X_H2_HRS0)
sum_X_HRS1 = sum(X_H2_HRS1)
max_X_HRS1 = max(X_H2_HRS1)
min_X_HRS1 = min(X_H2_HRS1)
print(f"----------------------------------------H2 purchase------------------------------------------")
print(f"Green H2 purchase for HRS0 is {sum_G_HRS0/5000} max: {max_G_HRS0} min: {min_G_HRS0}")
print(f"Green H2 purchase for HRS1 is {sum_G_HRS1/5000} max: {max_G_HRS1} min: {min_G_HRS1}")
print(f"On-site H2 HRS0 is {sum_OS_HRS0/5000} max: {max_OS_HRS0} min: {min_OS_HRS0}")
print(f"Blue H2 purchase for HRS0 is {sum_B_HRS0/5000} max: {max_B_HRS0} min: {min_B_HRS0}")
print(f"Blue H2 purchase for HRS1 is {sum_B_HRS1/5000} max: {max_B_HRS1} min: {min_B_HRS1}")
print(f"Gray H2 purchase for HRS0 is {sum_X_HRS0/5000} max: {max_X_HRS0} min: {min_X_HRS0}")
print(f"Gray H2 purchase for HRS1 is {sum_X_HRS1/5000} max: {max_X_HRS1} min: {min_X_HRS1}")




