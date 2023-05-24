# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:14:45 2023

@author: majid
"""
# Environment building
import math
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import pyomo as py
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
solvername ='cplex'
Path = '/opt/ibm/ILOG/CPLEX_Studio201/cplex/bin/x86-64_linux/cplex'
solver = SolverFactory(solvername, executable=Path)
# solver = SolverFactory('cplex', executable='E:/IBM/cplex/bin/x64_win64/cplex')
# solver_NLP = SolverFactory('ipopt', executable='E:/Ipopt-3.5.1-win32-icl10.0-debug/bin/ipopt')

#%%

t = 1000
Pmax = 10
Emax = 20
Eff = 0.95
Eini = 0

# df_price = pd.read_excel (r'C:\Users\majid\Box\Daily_files\Coding_S\.xlsx',sheet_name="DAM_LZ_SOUTH_2022",usecols ='A:E')

DA_price = pd.read_csv("DAM_LZ_SOUTH_2022.csv")  
DA_price = DA_price['Settlement Point Price'].values
#%%

model=pyo.AbstractModel()
model.dual=pyo.Suffix(direction=pyo.Suffix.IMPORT)

model.t = pyo.RangeSet(1,t)


#%% Defining the varibales

model.Pch = pyo.Var(model.t,domain=pyo.NonNegativeReals)
model.Pdch = pyo.Var(model.t,domain=pyo.NonNegativeReals)
model.Ich = pyo.Var(model.t,domain=pyo.Binary)
model.Idch = pyo.Var(model.t,domain=pyo.Binary)
model.Mch = pyo.Var(model.t,domain=pyo.Binary)
model.Mdch = pyo.Var(model.t,domain=pyo.Binary)
model.IMc = pyo.Var(model.t,domain=pyo.Binary)
model.IMdc = pyo.Var(model.t,domain=pyo.Binary)
model.E = pyo.Var(model.t, domain = pyo.NonNegativeReals)

#%% Equations     
def ESS_Cons1(model,t):
    if t==1: #the first time
        return model.E[t]== Eini + model.Pch[t] * Eff - model.Pdch[t] / Eff
    else: #the first time
        return model.E[t]== model.E[t-1] + model.Pch[t] * Eff - model.Pdch[t] / Eff
    return pyo.Constraint.Skip

def ESS_Cons2(model,t):
    return model.Pch[t] <= Pmax * model.Ich[t]

def ESS_Cons3(model,t):
    return model.Pdch[t] <= Pmax * model.Idch[t]

def ESS_Cons4(model,t):
    return model.E[t] <= Emax

def ESS_Cons5(model,t):
    return model.Ich[t] + model.Idch[t] == 1



def ESS_Lin_C_1(model,t):
    return model.Mch[t] == model.Ich[t] -  model.IMc[t]


def ESS_Lin_C_2(model,t):
    if t==1: 
        return model.IMc[t] <= 0
    else:
        return model.IMc[t] <= model.Ich[t-1]        
    return pyo.Constraint.Skip
    

def ESS_Lin_C_3(model,t):
    return  model.IMc[t] <= model.Ich[t] 

 
def ESS_Lin_C_4(model,t):
    if t==1: 
        return model.IMc[t] >= model.Ich[t] - 1
    else:
        return model.IMc[t] >= model.Ich[t] + model.Ich[t-1] - 1       
    return pyo.Constraint.Skip


def ESS_Lin_D_1(model,t):
    return model.Mdch[t] == model.Idch[t] -  model.IMdc[t]


def ESS_Lin_D_2(model,t):
    if t==1: 
        return model.IMdc[t] <= 0
    else:
        return model.IMdc[t] <= model.Idch[t-1]        
    return pyo.Constraint.Skip
    

def ESS_Lin_D_3(model,t):
    return  model.IMdc[t] <= model.Idch[t] 

 
def ESS_Lin_D_4(model,t):
    if t==1: 
        return model.IMdc[t] >= model.Idch[t] - 1
    else:
        return model.IMdc[t] >= model.Idch[t] + model.Idch[t-1] - 1       
    return pyo.Constraint.Skip


def ESS_cycle(model):
    return  sum((model.Mch[t]+model.Mdch[t]) for t in model.t) <= 100

def obj_func(model):
    return sum( (model.Pdch[t]-model.Pch[t])*DA_price[t-1] for t in model.t)

#%%

model.constraint1 = pyo.Constraint(model.t,rule=ESS_Cons1)
model.constraint2 = pyo.Constraint(model.t,rule=ESS_Cons2)
model.constraint3 = pyo.Constraint(model.t,rule=ESS_Cons3)
model.constraint4 = pyo.Constraint(model.t,rule=ESS_Cons4)
model.constraint5 = pyo.Constraint(model.t,rule=ESS_Cons5)
model.constraint6 = pyo.Constraint(model.t,rule=ESS_Lin_C_1)
model.constraint7 = pyo.Constraint(model.t,rule=ESS_Lin_C_2)
model.constraint8 = pyo.Constraint(model.t,rule=ESS_Lin_C_3)
model.constraint9 = pyo.Constraint(model.t,rule=ESS_Lin_C_4)
model.constraint10 = pyo.Constraint(model.t,rule=ESS_Lin_D_1)
model.constraint11 = pyo.Constraint(model.t,rule=ESS_Lin_D_2)
model.constraint12 = pyo.Constraint(model.t,rule=ESS_Lin_D_3)
model.constraint13 = pyo.Constraint(model.t,rule=ESS_Lin_D_4)
model.constraint14 = pyo.Constraint(rule=ESS_cycle)
model.OBJ = pyo.Objective(rule=obj_func, sense=maximize)  

#%%

instance = model.create_instance()
results = solver.solve(instance,tee=True)

#%% Results
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    feas_run="Feasible"
else: 
    feas_run="Not Feasible"


objective=instance.OBJ.expr()
print("The Objective function is:  $", objective)
print("The feasibility status is:", feas_run)

Pch_d = np.zeros(8760)
for t in range(1,t+1):
    Pch_d[t-1] = instance.Pch[t].value - instance.Pdch[t].value

        
Energy = np.zeros(8760)
for t in range(1,t+1):
    Energy[t-1] = instance.E[t].value 



Num_cyc = np.zeros(8760)
for t in range(1,t+1):
    Num_cyc[t-1] = instance.Mch[t].value + instance.Mdch[t].value

Total_Num_cyc = sum(Num_cyc)
print("The total Number of cycles is:", Total_Num_cyc)

Num_cyc_C = np.zeros(8760)
for t in range(1,t+1):
    Num_cyc_C[t-1] = instance.Mch[t].value 

Num_cyc_D = np.zeros(8760)
for t in range(1,t+1):
    Num_cyc_D[t-1] = instance.Mdch[t].value 



# #%% Plot

# plt.plot(Num_cyc)

# plt.legend(["Total cycle"])
# plt.show()


# plt.plot(Num_cyc_C)
# plt.legend(["Cycle:Chareg"])
# plt.show()

# plt.plot(Num_cyc_D)
# plt.legend(["Cycle:Discharge"])
# plt.show()

#%%
  


#%%