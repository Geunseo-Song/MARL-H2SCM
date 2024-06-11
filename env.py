# Hierarchical Structure of HDS and HRS
import os
from typing import List, Union, Any

import gym
import numpy as np
import pandas as pd
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from copy import copy
import logging

logger = logging.getLogger(__name__)

env_name = "env"

class HydroRefuelSys(MultiAgentEnv):


    def __init__(self, config=None):
        self.num_HRS = config["num_HRS"] 
        self.HDS_agent = ["HDS"]
        self.HRS_agent = ["HRS"+str(n) for n in range(self.num_HRS)]
        self.agents = self.HDS_agent + self.HRS_agent

        # HRS
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(8,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(5,))

        self._agent_ids = set(self.agents)
        self._spaces_in_preferred_format = True
        super().__init__()

        # data import
        self.hload, self.PVrate, self.MGpie = [], [], []
        for i in range(self.num_HRS):
            a = '/home/username/MARLlib/CaseStudy/data{}.csv'.format(i)
            if os.path.isfile(a):
                b = pd.read_csv(a)
                self.hload.append([b["hload"]])
                self.PVrate.append([b["r"]])
                self.MGpie.append([b["Pie"]])
            else:  # In case the file does not exist.
                a = '/home/username/MARLlib/CaseStudy/data0.csv'
                b = pd.read_csv(a)
                self.hload.append([b["hload"]])
                self.PVrate.append([b["r"]])
                self.MGpie.append([b["Pie"]])
                
        # HDS variable initialization
        self.G_pie = []
        self.B_pie = []
        self.X_pie = []
        self.penalty_HDS = 0
        self.hydroG = 0
        self.hydroB = 0
        self.hydroX = 0
        self.hydroGAccum = [0 for i in range(self.num_HRS)]
        self.hydroBAccum = [0 for i in range(self.num_HRS)]
        self.hydroXAccum = [0 for i in range(self.num_HRS)]
        self.hydroAvail = 0
        self.hydroGAvail = 0
        self.hydroBAvail = 0
        self.hydroXAvail = 0
        self.G_CO2_central_total = 0
        self.G_CO2_dist_total = 0
        self.B_CO2_total = 0
        self.X_CO2_total = 0
        
        # HRS variable initialization
        self.penalty_HRS = 0
        self.penalty_ELE = 0
        self.SOH = [0 for i in range(self.num_HRS)]
        self.hydro_G = [0 for i in range(self.num_HRS)]
        self.hydro_B = [0 for i in range(self.num_HRS)]
        self.hydro_X = [0 for i in range(self.num_HRS)]
        self.hydroOS = [0 for i in range(self.num_HRS)]
        self.powerMG = [0 for i in range(self.num_HRS)]
        self.powerDG = [0 for i in range(self.num_HRS)]
        self.powerELE = [0 for i in range(self.num_HRS)]
        self.powerCOM = [0 for i in range(self.num_HRS)]
        self.diff_SOH = [0 for i in range(self.num_HRS)]

        # General parameter
        self.period = 24
        self.eta = 2.5 # compressor and chiller coefficient
        self.LHV = 39.72 # lower heating value of hydrogen(kWh/kg)



        # HDS parameter
        if config is not None and "G_CV" in config:
            self.G_CV = config["G_CV"]
        else:
            self.G_CV = 0.6
        if config is not None and "B_CV" in config:
            self.B_CV = config["B_CV"]
        else:
            self.B_CV = 0
        if config is not None and "X_CV" in config:
            self.X_CV = config["X_CV"]
        else:
            self.X_CV = 0
        if config is not None and "G_capacity" in config:
            self.G_capaciy = config["G_capacity"]
        else: 
            self.G_capacity = 1
        if config is not None and "B_capacity" in config:
            self.B_capaciy = config["B_capacity"]
        else: 
            self.B_capacity = 1
        if config is not None and "X_capacity" in config:
            self.X_capaciy = config["X_capacity"]
        else: 
            self.X_capacity = 1
        if config is not None and "G_CO2_central" in config:
            self.G_CO2_central = config["G_CO2_central"]
        else:
            self.G_CO2_central = 2.87
        if config is not None and "G_CO2_dist" in config:
            self.G_CO2_dist = config["G_CO2_dist"]
        else:
            self.G_CO2_dist = 2.89
        if config is not None and "B_CO2" in config:
            self.B_CO2 = config["B_CO2"]
        else:
            self.B_CO2 = 6.44
        if config is not None and "X_CO2" in config:
            self.X_CO2 = config["X_CO2"]
        else:
            self.X_CO2 = 9.91

        
        # Techno-economic analysis formula 
        if config is not None and "Gpie" in config:
            self.Gpie = config["Gpie"]
        else:
            self.Gpie = 3.08 * self.G_capacity - self.G_CV
        if config is not None and "Bpie" in config:
            self.Bpie = config["Bpie"]
        else:
            self.Bpie = 2.17 * self.B_capacity - self.B_CV
        if config is not None and "Xpie" in config:
            self.Xpie = config["Xpie"]
        else:
            self.Xpie = 1.74 * self.X_capacity - self.X_CV
      


        # HRS parameter
        if config is not None and "spv" in config:
            self.spv = config["spv"]
        else:
            self.spv = [600 for i in range(self.num_HRS)]
        if config is not None and "etas" in config:
            self.etas = config["etas"]
        else:
            self.etas = [0.186 for i in range(self.num_HRS)]
        if config is not None and "Pmax" in config:
            self.Pmax = config["Pmax"]
        else:
            self.Pmax = [500 for i in range(self.num_HRS)]
        if config is not None and "HGmax" in config:
            self.HGmax = config["HGmax"]
        else:
            self.HGmax = [5 for i in range(self.num_HRS)]
        if config is not None and "HBmax" in config:
            self.HBmax = config["HBmax"]
        else:
            self.HBmax = [5 for i in range(self.num_HRS)]
        if config is not None and "HXmax" in config:
            self.HXmax = config["HXmax"]
        else:
            self.HXmax = [5 for i in range(self.num_HRS)]
        if config is not None and "DGmax" in config:
            self.DGmax = config["DGmax"]
        else:
            self.DGmax = [50 for i in range(self.num_HRS)]
        if config is not None and "ELEmax" in config:
            self.ELEmax = config["ELEmax"]
        else:
            self.ELEmax = [250 for i in range(self.num_HRS)]
        if config is not None and "ELEeff" in config:
            self.ELEeff = config["ELEeff"]
        else:
            self.ELEeff = [0.68 for i in range(self.num_HRS)]
        if config is not None and "SOHmax" in config:
            self.SOHmax = config["SOHmax"]
        else:
            self.SOHmax = [30 for i in range(self.num_HRS)]
        if config is not None and "SOHmin" in config:
            self.SOHmin = config["SOHmin"]
        else:
            self.SOHmin = [0 for i in range(self.num_HRS)]
        if config is not None and "DGpie" in config:
            self.DGpie = config["DGpie"]
        else:
            self.DGpie = [0.05 for i in range(self.num_HRS)]
        if config is not None and "G_pie_max" in config:
            self.G_pie_max = config["G_pie_max"]
        else:
            self.G_pie_max = [self.Gpie/0.4, self.Gpie/0.4]
        if config is not None and "B_pie_max" in config:
            self.B_pie_max = config["B_pie_max"]
        else:
            self.B_pie_max = [self.Bpie/0.4, self.Bpie/0.4]
        if config is not None and "X_pie_max" in config:
            self.X_pie_max = config["X_pie_max"]
        else:
            self.X_pie_max = [self.Xpie/0.4, self.Xpie/0.4]
        self.powerPV = []
        for ag in range(self.num_HRS):
            power = [self.spv[ag] * self.etas[ag] * x for x in self.PVrate[ag]]
            self.powerPV.extend([power])
        if config is not None and "HRS_OS" in config:
            self.HRS_OS = config["HRS_OS"]
        else:
            self.HRS_OS = [1,0]



    def reset(self):
        # [[HRS0_X_pie, HRS1_X_pie, Xpie], [HRS0_B_pie, HRS1_B_pie, Bpie], [HRS0_G_pie, HRS1_G_pie, Gpie], [demand of HRS0, demand of HRS1, hydroM(=hydroB + hydroX + hydroG)]]
        self.HDS_state = {}
        self.HDS_state["HDS"] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        # self.HRS_state = {"HRS0": [0,0,0,0,0,0,0,0], "HRS2": [0,0,0,0,0,0,0,0]}
        self.HRS_state = {}
        for ag in self.HRS_agent:
            self.HRS_state[ag] = [0,0,0,0,0,0,0,0]
        
        self.state = dict(self.HDS_state, **self.HRS_state) # **Variadic parameter
        self.steps_remaining_at_level = None
        self.num_high_level_steps = 0
        self.HRS_agent_ids = []
        for ag in self.HRS_agent: # No need for 1 day simulation
            ids = f"{ag}_level_{self.num_high_level_steps}"
            self.HRS_agent_ids.append(ids)
        # self.HRS_agent_ids = ["HRS0_level_{}".format(self.num_high_level_steps), "HRS1_level_{}".format(self.num_high_level_steps)]
        # print(type(self.HRS_agent_ids))
        # print(self.HRS_agent_ids)
        # print("------------------------------------reset------------------------------------")
        
        # HDS variable initialization
        self.G_pie = []
        self.B_pie = []
        self.X_pie = []
        self.penalty_HDS = 0
        self.hydroG = 0
        self.hydroB = 0
        self.hydroX = 0
        self.hydroGAccum = [0 for i in range(self.num_HRS)]
        self.hydroBAccum = [0 for i in range(self.num_HRS)]
        self.hydroXAccum = [0 for i in range(self.num_HRS)]
        self.hydroAvail = 0
        self.hydroGAvail = 0
        self.hydroBAvail = 0
        self.hydroXAvail = 0
        self.H_pie = []
        self.G_CO2_central_total = 0
        self.G_CO2_dist_total = 0
        self.B_CO2_total = 0
        self.X_CO2_total = 0


        
        # HRS variable initialization
        self.penalty_HRS = 0
        self.penalty_ELE = 0
        self.SOH = [0 for i in range(self.num_HRS)]
        self.hydro_G = [0 for i in range(self.num_HRS)]
        self.hydro_B = [0 for i in range(self.num_HRS)]
        self.hydro_X = [0 for i in range(self.num_HRS)]
        self.hydroOS = [0 for i in range(self.num_HRS)]
        self.powerMG = [0 for i in range(self.num_HRS)]
        self.powerDG = [0 for i in range(self.num_HRS)]
        self.powerELE = [0 for i in range(self.num_HRS)]
        self.powerCOM = [0 for i in range(self.num_HRS)]
        self.diff_SOH = [0 for i in range(self.num_HRS)]

        # # save data
        # self.HDSaction_acc = []
        # self.HRSaction_acc = []
        # self.cost_acc = []
        # self.profit_acc = []
        # self.SOH_acc = []
        # self.Remain_acc = []
        # self.OSprod_acc = []
        # self.SOHdiff_acc = []
        # self.purchaseG_acc = []
        # self.purchaseB_acc = []
        # self.purchaseX_acc = []
        # self.H2buy_acc = []
        # self.GH2buy_acc = []
        # self.BH2buy_acc = []
        # self.XH2buy_acc = []
        # self.H2pro_acc = []
        # self.powerMG_acc = []
        # self.powerDG_acc = []
        # self.powerPV_acc = []
        # self.G_pie_acc = []
        # self.B_pie_acc = []
        # self.X_pie_acc = []
        # self.G_CO2_central_total_acc = []
        # self.G_CO2_dist_total_acc = []
        # self.B_CO2_total_acc = []
        # self.X_CO2_total_acc = []
        # self.SOH_rcd = []
        # self.Remain_rcd = []
        # self.OSprod_rcd = []
        # self.SOHdiff_rcd = []
        # self.purchase_rcd = []
        # self.H2buy_rcd = []
        # self.H2pro_rcd = []
        # self.a = open(f"./env_data/action_HDS({env_name}).txt", "a") # HDS action
        # self.b = open(f"./env_data/action_HRS({env_name}).txt", "a") # HRS action
        # self.c = open(f"./env_data/cost({env_name}).txt", "a") # cost HRS
        # self.d = open(f"./env_data/SOH({env_name}).txt", "a") # SOH level
        # self.e = open(f"./env_data/hydroRemain({env_name}).txt", "a") # hydroRemain
        # self.f = open(f"./env_data/OSproduction({env_name}).txt", "a") # hydroOS
        # self.g = open(f"./env_data/diffSOH({env_name}).txt", "a") # SOH difference
        # self.h = open(f"./env_data/profit({env_name}).txt", "a")  # profit HDS
        # self.i = open(f"./env_data/H2buy({env_name}).txt", "a")
        # self.j = open(f"./env_data/H2pro({env_name}).txt", "a")
        # self.k = open(f"./env_data/powerMG({env_name}).txt", "a")
        # self.l = open(f"./env_data/powerDG({env_name}).txt", "a")
        # self.m = open(f"./env_data/powerPV({env_name}).txt", "a")
        # self.n = open(f"./env_data/purchaseG({env_name}).txt", "a")
        # self.o = open(f"./env_data/purchaseB({env_name}).txt", "a")
        # self.p = open(f"./env_data/purchaseX({env_name}).txt", "a")
        # self.q = open(f"./env_data/GH2buy({env_name}).txt", "a")
        # self.r = open(f"./env_data/BH2buy({env_name}).txt", "a")
        # self.s = open(f"./env_data/XH2buy({env_name}).txt", "a")
        # self.t = open(f"./env_data/G_CO2_central_total({env_name}).txt", "a")
        # self.u = open(f"./env_data/G_CO2_dist_total({env_name}).txt", "a")
        # self.v = open(f"./env_data/B_CO2_total({env_name}).txt", "a")
        # self.w = open(f"./env_data/X_CO2_total({env_name}).txt", "a")

        return {"HDS": self.state["HDS"]}

    def step(self, action_dict):
        # assert len(action_dict) == 1, action_dict
        if "HDS" in action_dict:
            return self._HDS_step(action_dict["HDS"])
        else:
            return self._HRS_step(action_dict)

    def _HDS_step(self, action):
        # print("----------------------------------HDS step-------------------------------------")
        logger.debug("High level agent sets goal")
        G_pie = action[0] * self.G_pie_max[0]
        self.G_pie.append(G_pie)
        G_pie = action[1] * self.G_pie_max[1]
        self.G_pie.append(G_pie)
        B_pie = action[2] * self.B_pie_max[0]
        self.B_pie.append(B_pie)
        B_pie = action[3] * self.B_pie_max[1]
        self.B_pie.append(B_pie)
        X_pie = action[4] * self.X_pie_max[0]
        self.X_pie.append(X_pie)
        X_pie = action[5] * self.X_pie_max[1]
        self.X_pie.append(X_pie)

        self.hydroG = action[6] * sum(self.HGmax) * 1.1 * self.period
        self.hydroB = action[7] * sum(self.HBmax) * 1.1 * self.period
        self.hydroX = action[8] * sum(self.HXmax) * 1.1 * self.period
        self.hydroAvail = copy(self.hydroG + self.hydroB + self.hydroX)
        self.hydroGAvail = copy(self.hydroG)
        self.hydroBAvail = copy(self.hydroB)
        self.hydroXAvail = copy(self.hydroX)
        self.G_CO2_central_total = self.G_CO2_central * self.hydroG
        self.B_CO2_total = self.B_CO2 * self.hydroB
        self.X_CO2_total = self.X_CO2 * self.hydroX


        self.HDSaction_acc.extend([self.hydroG, self.hydroB, self.hydroX, self.G_pie, self.B_pie, self.X_pie])

        self.steps_remaining_at_level = 24
        self.num_high_level_steps += 1
        self.HRS_agent_ids = []
        for ag in self.HRS_agent:
            ids = f"{ag}_level_{self.num_high_level_steps}"
            self.HRS_agent_ids.append(ids)
        obs, rew, done = {}, {}, {}
        for ag in range(self.num_HRS):
            obs[f"HRS{ag}_level_{self.num_high_level_steps}"] = [self.steps_remaining_at_level, self.G_pie[ag], self.B_pie[ag], self.X_pie[ag], self.SOH[ag], self.hload[ag][0][24 - self.steps_remaining_at_level], self.MGpie[ag][0][24 - self.steps_remaining_at_level], self.powerPV[ag][0][24 - self.steps_remaining_at_level]]
            rew[f"HRS{ag}_level_{self.num_high_level_steps}"] = 0
            done[f"HRS{ag}_level_{self.num_high_level_steps}"] = False

        done["__all__"] = False
        return obs, rew, done, {}

    def _HRS_step(self, action):
        obs, rew, done = {}, {}, {}
        # print("----------------------------------HRS step-------------------------------------")
        logger.debug("Low level agent step {}".format(action))
        self.steps_remaining_at_level -= 1
        rew, obs, done = {}, {}, {}

        # act_rcd, H2buy_rcd, GH2buy_rcd, BH2buy_rcd, XH2buy_rcd, H2pro_rcd, cost_rcd, Remain_rcd, OSprod_rcd, SOH_rcd, SOHdiff_rcd, \
        # purchaseG_rcd, purchaseB_rcd, purchaseX_rcd, powerMG_rcd, powerDG_rcd, powerPV_rcd, \
        # G_CO2_central_total_rcd, G_CO2_dist_total_rcd, B_CO2_total_rcd, X_CO2_total_rcd = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []  # record for just HRS loop

        # print(f"----------------------------------step for HRS0-------------------------------------")
        act = list(action.values())[0]
        hydro_G = act[0] * self.HGmax[0]
        hydro_B = act[1] * self.HBmax[0]
        hydro_X = act[2] * self.HXmax[0]
        powerMG = act[3] * self.Pmax[0]
        powerDG = act[4] * self.DGmax[0]


        if self.hydroGAvail >= hydro_G:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroGAvail >= 0, "hydroGAvail can't be negative"
            self.penalty_HDS += (hydro_G - self.hydroGAvail) * self.G_pie[0]
            hydro_G = copy(self.hydroGAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroG in HDS--------------------------------------------------")
        if self.hydroBAvail >= hydro_B:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroBAvail >= 0, "hydroBAvail can't be negative"
            self.penalty_HDS += (hydro_B - self.hydroBAvail) * self.B_pie[0]
            hydro_B = copy(self.hydroBAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroB in HDS--------------------------------------------------")
        if self.hydroXAvail >= hydro_X:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroXAvail >= 0, "hydroXAvail can't be negative"
            self.penalty_HDS += (hydro_X - self.hydroXAvail) * self.X_pie[0]
            hydro_X = copy(self.hydroXAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroX in HDS--------------------------------------------------")

        self.hydroGAccum[0] += hydro_G
        self.hydroGAvail = self.hydroG - sum(self.hydroGAccum)
        self.hydroBAccum[0] += hydro_B
        self.hydroBAvail = self.hydroB - sum(self.hydroBAccum)
        self.hydroXAccum[0] += hydro_X
        self.hydroXAvail = self.hydroX - sum(self.hydroXAccum)

        Pele = (powerMG + powerDG + self.powerPV[0][0][23 - self.steps_remaining_at_level]) * (
                self.LHV / (self.LHV + self.ELEeff[0] * self.eta))
        if Pele >= self.ELEmax[0]:
            self.penalty_ELE = (Pele - self.ELEmax[0]) / 5
            Pele = self.ELEmax[0]
        else:
            self.penalty_ELE = 0
        hydroOS = (self.ELEeff[0] * self.HRS_OS[0] / self.LHV) * Pele
        Pcom = self.eta * hydroOS
        self.G_CO2_dist_total = self.G_CO2_dist * hydroOS

        assert Pcom + Pele <= powerMG + powerDG + self.powerPV[0][0][23 - self.steps_remaining_at_level], "Pcom Error"
        hydroRemain = (hydro_G + hydro_B + hydro_X) + hydroOS - self.hload[0][0][23 - self.steps_remaining_at_level]

        if hydroRemain > self.SOHmax[0] - self.SOH[0] and hydroRemain >= 0:
            if hydroRemain - (self.SOHmax[0] - self.SOH[0]) < 1:
                self.penalty_HRS = hydroRemain - (self.SOHmax[0] - self.SOH[0]) * (
                            1 + (self.G_pie[0] + self.B_pie[0] + self.X_pie[0]) / 3)
            else:
                self.penalty_HRS = (hydroRemain - (self.SOHmax[0] - self.SOH[0])) * 10 * (
                            1 + (self.G_pie[0] + self.B_pie[0] + self.X_pie[0]) / 3)
            self.diff_SOH[0] = self.SOHmax[0] - self.SOH[0]
            self.SOH[0] = copy(self.SOHmax[0])
        elif hydroRemain <= (self.SOHmax[0] - self.SOH[0]) and hydroRemain >= 0:
            self.diff_SOH[0] = hydroRemain
            self.SOH[0] += hydroRemain
            self.penalty_HRS = 0
        elif -hydroRemain <= self.SOH[0] and hydroRemain < 0:
            assert hydroRemain < 0, "hydroRemain should not be positive value(1)"
            self.diff_SOH[0] = hydroRemain
            self.SOH[0] += hydroRemain
            self.penalty_HRS = 0
        elif -hydroRemain > self.SOH[0] and hydroRemain < 0:
            assert hydroRemain < 0, "hydroRemain should not be positive value(2)"
            self.diff_SOH[0] = - self.SOH[0]
            if -(self.SOH[0] + hydroRemain) < 1:
                self.penalty_HRS = -(self.SOH[0] + hydroRemain) * (
                            1 + (self.G_pie[0] + self.B_pie[0] + self.X_pie[0]) / 3)
            else:
                self.penalty_HRS = -(self.SOH[0] + hydroRemain) * 10 * (
                            1 + (self.G_pie[0] + self.B_pie[0] + self.X_pie[0]) / 3)
            self.SOH[0] = 0
        else:
            raise ValueError

        H2buy = hydro_G * self.G_pie[0] + hydro_B * self.B_pie[0] + hydro_X * self.X_pie[0]
        GH2buy = hydro_G * self.G_pie[0]
        BH2buy = hydro_B * self.B_pie[0]
        XH2buy = hydro_X * self.X_pie[0]
        H2pro = powerMG * self.MGpie[0][0][23 - self.steps_remaining_at_level] + powerDG * self.DGpie[0]
        cost = H2buy + H2pro

        # act_rcd.append(list(act))
        # H2buy_rcd.append(H2buy)
        # GH2buy_rcd.append(GH2buy)
        # BH2buy_rcd.append(BH2buy)
        # XH2buy_rcd.append(XH2buy)
        # H2pro_rcd.append(H2pro)
        # cost_rcd.append(cost)
        # Remain_rcd.append(hydroRemain)
        # OSprod_rcd.append(hydroOS)
        # SOH_rcd.append(self.SOH[0])
        # SOHdiff_rcd.append(self.diff_SOH[0])
        # purchaseG_rcd.append(hydro_G)
        # purchaseB_rcd.append(hydro_B)
        # purchaseX_rcd.append(hydro_X)
        # powerMG_rcd.append(powerMG)
        # powerDG_rcd.append(powerDG)
        # powerPV_rcd.append(self.powerPV[0])
        # G_CO2_central_total_rcd.append(self.G_CO2_central_total)
        # G_CO2_dist_total_rcd.append(self.G_CO2_dist_total)
        # B_CO2_total_rcd.append(self.B_CO2_total)
        # X_CO2_total_rcd.append(self.X_CO2_total)

        reward = -cost - self.penalty_HRS - self.penalty_ELE + hydroOS * self.G_CV
        rew[f"HRS{0}_level_{self.num_high_level_steps}"] = reward
        obs[f"HRS{0}_level_{self.num_high_level_steps}"] = [self.steps_remaining_at_level, self.G_pie[0],
                                                             self.B_pie[0], self.X_pie[0],
                                                             self.SOH[0],
                                                             self.hload[0][0][24 - self.steps_remaining_at_level],
                                                             self.MGpie[0][0][24 - self.steps_remaining_at_level],
                                                             self.powerPV[0][0][
                                                                 24 - self.steps_remaining_at_level]]
        done[f"HRS{0}_level_{self.num_high_level_steps}"] = False

        # print(f"----------------------------------step for HRS{1}-------------------------------------")
        act = list(action.values())[1]
        hydro_G = act[0] * self.HGmax[1]
        hydro_B = act[1] * self.HBmax[1]
        hydro_X = act[2] * self.HXmax[1]



        if self.hydroGAvail >= hydro_G:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroGAvail >= 0, "hydroGAvail can't be negative"
            self.penalty_HDS += (hydro_G - self.hydroGAvail) * self.G_pie[1]
            hydro_G = copy(self.hydroGAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroG in HDS--------------------------------------------------")
        if self.hydroBAvail >= hydro_B:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroBAvail >= 0, "hydroBAvail can't be negative"
            self.penalty_HDS += (hydro_B - self.hydroBAvail) * self.B_pie[1]
            hydro_B = copy(self.hydroBAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroB in HDS--------------------------------------------------")
        if self.hydroXAvail >= hydro_X:
            self.penalty_HDS = 0
            pass
        else:
            assert self.hydroXAvail >= 0, "hydroXAvail can't be negative"
            self.penalty_HDS += (hydro_X - self.hydroXAvail) * self.X_pie[1]
            hydro_X = copy(self.hydroXAvail)
            # print(
            #     "-----------------------------------------Out of stock_hydroX in HDS--------------------------------------------------")

        self.hydroGAccum[1] += hydro_G
        self.hydroGAvail = self.hydroG - sum(self.hydroGAccum)
        self.hydroBAccum[1] += hydro_B
        self.hydroBAvail = self.hydroB - sum(self.hydroBAccum)
        self.hydroXAccum[1] += hydro_X
        self.hydroXAvail = self.hydroX - sum(self.hydroXAccum)

        Pele = (powerMG + powerDG + self.powerPV[1][0][23 - self.steps_remaining_at_level]) * (
                self.LHV / (self.LHV + self.ELEeff[1] * self.eta))
        if Pele >= self.ELEmax[1]:
            self.penalty_ELE = (Pele - self.ELEmax[1]) / 5
            Pele = self.ELEmax[1]
        else:
            self.penalty_ELE = 0
        hydroOS = (self.ELEeff[1] * self.HRS_OS[1] / self.LHV) * Pele
        Pcom = self.eta * hydroOS
        self.G_CO2_dist_total = self.G_CO2_dist * hydroOS

        assert Pcom <= powerMG + powerDG + self.powerPV[1][0][
            23 - self.steps_remaining_at_level], "Pcom Error"
        hydroRemain = (hydro_G + hydro_B + hydro_X) + hydroOS - self.hload[1][0][23 - self.steps_remaining_at_level]
        if hydroRemain > self.SOHmax[1] - self.SOH[1] and hydroRemain >= 0:
            if hydroRemain - (self.SOHmax[1] - self.SOH[1]) < 1:
                self.penalty_HRS = hydroRemain - (self.SOHmax[1] - self.SOH[1]) * (
                            1 + (self.G_pie[1] + self.B_pie[1] + self.X_pie[1]) / 3)
            else:
                self.penalty_HRS = (hydroRemain - (self.SOHmax[1] - self.SOH[1])) * 10 * (
                            1 + (self.G_pie[1] + self.B_pie[1] + self.X_pie[1]) / 3)
            self.diff_SOH[1] = self.SOHmax[1] - self.SOH[1]
            self.SOH[1] = copy(self.SOHmax[1])
        elif hydroRemain <= (self.SOHmax[1] - self.SOH[1]) and hydroRemain >= 0:
            self.diff_SOH[1] = hydroRemain
            self.SOH[1] += hydroRemain
            self.penalty_HRS = 0
        elif -hydroRemain <= self.SOH[1] and hydroRemain < 0:
            assert hydroRemain < 0, "hydroRemain should not be positive value(1)"
            self.diff_SOH[1] = hydroRemain
            self.SOH[1] += hydroRemain
            self.penalty_HRS = 0
        elif -hydroRemain > self.SOH[1] and hydroRemain < 0:
            assert hydroRemain < 0, "hydroRemain should not be positive value(2)"
            self.diff_SOH[1] = - self.SOH[1]
            if -(self.SOH[1] + hydroRemain) < 1:
                self.penalty_HRS = -(self.SOH[1] + hydroRemain) * (
                            1 + (self.G_pie[1] + self.B_pie[1] + self.X_pie[1]) / 3)
            else:
                self.penalty_HRS = -(self.SOH[1] + hydroRemain) * 10 * (
                            1 + (self.G_pie[1] + self.B_pie[1] + self.X_pie[1]) / 3)
            self.SOH[1] = 0
        else:
            raise ValueError

        H2buy = hydro_G * self.G_pie[1] + hydro_B * self.B_pie[1] + hydro_X * self.X_pie[1]
        GH2buy = hydro_G * self.G_pie[1]
        BH2buy = hydro_B * self.B_pie[1]
        XH2buy = hydro_X * self.X_pie[1]
        H2pro = powerMG * self.MGpie[1][0][23 - self.steps_remaining_at_level] + powerDG * self.DGpie[1]
        cost = H2buy

        # act_rcd.append(list(act))
        # H2buy_rcd.append(H2buy)
        # GH2buy_rcd.append(GH2buy)
        # BH2buy_rcd.append(BH2buy)
        # XH2buy_rcd.append(XH2buy)
        # H2pro_rcd.append(H2pro)
        # cost_rcd.append(cost)
        # Remain_rcd.append(hydroRemain)
        # OSprod_rcd.append(hydroOS)
        # SOH_rcd.append(self.SOH[1])
        # SOHdiff_rcd.append(self.diff_SOH[1])
        # purchaseG_rcd.append(hydro_G)
        # purchaseB_rcd.append(hydro_B)
        # purchaseX_rcd.append(hydro_X)
        # powerMG_rcd.append(powerMG)
        # powerDG_rcd.append(powerDG)
        # powerPV_rcd.append(self.powerPV[1])
        # G_CO2_central_total_rcd.append(self.G_CO2_central_total)
        # B_CO2_total_rcd.append(self.B_CO2_total)
        # X_CO2_total_rcd.append(self.X_CO2_total)

        reward = -cost - self.penalty_HRS - self.penalty_ELE
        rew[f"HRS{1}_level_{self.num_high_level_steps}"] = reward
        obs[f"HRS{1}_level_{self.num_high_level_steps}"] = [self.steps_remaining_at_level, self.G_pie[1],
                                                             self.B_pie[1], self.X_pie[1],
                                                             self.SOH[1],
                                                             self.hload[1][0][24 - self.steps_remaining_at_level],
                                                             self.MGpie[1][0][24 - self.steps_remaining_at_level],
                                                             self.powerPV[1][0][
                                                                 24 - self.steps_remaining_at_level]]
        done[f"HRS{1}_level_{self.num_high_level_steps}"] = False


        # self.HRSaction_acc.append(act_rcd)
        # self.H2buy_acc.append(H2buy_rcd)
        # self.GH2buy_acc.append(GH2buy_rcd)
        # self.BH2buy_acc.append(BH2buy_rcd)
        # self.XH2buy_acc.append(XH2buy_rcd)
        # self.H2pro_acc.append(H2pro_rcd)
        # self.cost_acc.append(cost_rcd)
        # self.SOH_acc.append(SOH_rcd)
        # self.Remain_acc.append(Remain_rcd)
        # self.OSprod_acc.append(OSprod_rcd)
        # self.SOHdiff_acc.append(SOHdiff_rcd)
        # self.purchaseG_acc.append(purchaseG_rcd)
        # self.purchaseB_acc.append(purchaseB_rcd)
        # self.purchaseX_acc.append(purchaseX_rcd)
        # self.powerMG_acc.append(powerMG_rcd)
        # self.powerDG_acc.append(powerDG_rcd)
        # self.powerPV_acc.append(powerPV_rcd)
        # self.G_CO2_central_total_acc.append(G_CO2_central_total_rcd)
        # self.G_CO2_dist_total_acc.append(G_CO2_dist_total_rcd)
        # self.B_CO2_total_acc.append(B_CO2_total_rcd)
        # self.X_CO2_total_acc.append(X_CO2_total_rcd)
        done = {"__all__": False}
        if self.steps_remaining_at_level == 0:
            done = {"__all__": True}
            profitX = sum([x * y for x, y in zip(self.X_pie, self.hydroXAccum)]) - self.Xpie * self.hydroX
            profitB = sum([x * y for x, y in zip(self.B_pie, self.hydroBAccum)]) - self.Bpie * self.hydroB
            profitG = sum([x * y for x, y in zip(self.G_pie, self.hydroGAccum)]) - self.Gpie * self.hydroG
            profitCV = sum(self.hydroGAccum) * self.G_CV * 0.3
            profit = profitX + profitB + profitG + profitCV
            logger.debug("high level final reward {}".format(profit))
            rew_HDS = profit - self.penalty_HDS
            self.profit_acc.extend([profit])
            rew["HDS"] = rew_HDS

            self.G_pie.append(self.Gpie)
            Green_info = self.G_pie
            self.B_pie.append(self.Bpie)
            Blue_info = self.B_pie
            self.X_pie.append(self.Xpie)
            Gray_info = self.X_pie
            demand_info = list(np.zeros(self.num_HRS))
            for n in range(self.num_HRS):
                for t in range(self.period):
                    tem = demand_info[n]
                    tem += self.H2buy_acc[t][n]
                    demand_info[n] = tem
            demand_info.append(0)
            obs["HDS"] = [Green_info, Blue_info, Gray_info, demand_info]


            # self.a.write("{:s}\n".format(str(self.HDSaction_acc)))
            # self.b.write("{:s}\n".format(str(self.HRSaction_acc)))
            # self.c.write("{:s}\n".format(str(self.cost_acc)))
            # self.d.write("{:s}\n".format(str(self.SOH_acc)))
            # self.e.write("{:s}\n".format(str(self.Remain_acc)))
            # self.f.write("{:s}\n".format(str(self.OSprod_acc)))
            # self.g.write("{:s}\n".format(str(self.SOHdiff_acc)))
            # self.h.write("{:s}\n".format(str(self.profit_acc)))
            # self.i.write("{:s}\n".format(str(self.H2buy_acc)))
            # self.j.write("{:s}\n".format(str(self.H2pro_acc)))
            # self.k.write("{:s}\n".format(str(self.powerMG_acc)))
            # self.l.write("{:s}\n".format(str(self.powerDG_acc)))
            # self.m.write("{:s}\n".format(str(self.powerPV_acc)))
            # self.n.write("{:s}\n".format(str(self.purchaseG_acc)))
            # self.o.write("{:s}\n".format(str(self.purchaseB_acc)))
            # self.p.write("{:s}\n".format(str(self.purchaseX_acc)))
            # self.q.write("{:s}\n".format(str(self.GH2buy_acc)))
            # self.r.write("{:s}\n".format(str(self.BH2buy_acc)))
            # self.s.write("{:s}\n".format(str(self.XH2buy_acc)))
            # self.t.write("{:s}\n".format(str(self.G_CO2_central_total_acc)))
            # self.u.write("{:s}\n".format(str(self.G_CO2_dist_total_acc)))
            # self.v.write("{:s}\n".format(str(self.B_CO2_total_acc)))
            # self.w.write("{:s}\n".format(str(self.X_CO2_total_acc)))
            # self.a.close()
            # self.b.close()
            # self.c.close()
            # self.d.close()
            # self.e.close()
            # self.f.close()
            # self.g.close()
            # self.h.close()
            # self.i.close()
            # self.j.close()
            # self.k.close()
            # self.l.close()
            # self.m.close()
            # self.n.close()
            # self.o.close()
            # self.p.close()
            # self.q.close()
            # self.r.close()
            # self.s.close()
            # self.t.close()
            # self.u.close()
            # self.v.close()
            # self.w.close()
        return obs, rew, done, {}



