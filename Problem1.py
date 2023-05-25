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

t = 8670
Pmax = 10
Emax = 20
Eff = 0.9
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
model.Aux1 = pyo.Var(model.t,domain=pyo.NonNegativeReals)
model.Aux3 = pyo.Var(model.t,domain=pyo.NonNegativeReals)
model.Aux2 = pyo.Var(model.t,domain=pyo.NegativeReals)
model.Aux4 = pyo.Var(model.t,domain=pyo.NegativeReals)
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
    return model.Pdch[t] <= Pmax *(1-model.Ich[t])

def ESS_Cons4(model,t):
    return model.E[t] <= Emax




def ESS_Lin_C_1(model,t):
    if t==1: 
        return model.Aux1[t]  >= model.Ich[t]
    else:
        return model.Aux1[t] >= model.Ich[t] - model.Ich[t-1]       
    return pyo.Constraint.Skip
    
def ESS_Lin_D_1(model,t):
    if t==1: 
        return model.Aux1[t]  >= model.Ich[t]
    else:
        return model.Aux1[t] >= model.Ich[t-1] - model.Ich[t]       
    return pyo.Constraint.Skip
    


def ESS_cycle(model):
    return  sum((model.Aux1[t]) for t in model.t) <= 365

def obj_func(model):
    return sum( (model.Pdch[t]-model.Pch[t])*DA_price[t-1] for t in model.t)\
        +0.001*sum((-model.Aux1[t]) for t in model.t)

#%%

model.constraint1 = pyo.Constraint(model.t,rule=ESS_Cons1)
model.constraint2 = pyo.Constraint(model.t,rule=ESS_Cons2)
model.constraint3 = pyo.Constraint(model.t,rule=ESS_Cons3)
model.constraint4 = pyo.Constraint(model.t,rule=ESS_Cons4)
model.constraint6 = pyo.Constraint(model.t,rule=ESS_Lin_C_1)
model.constraint6 = pyo.Constraint(model.t,rule=ESS_Lin_D_1)
model.constraint14 = pyo.Constraint(rule=ESS_cycle)
model.OBJ = pyo.Objective(rule=obj_func, sense=maximize)  

#%%
solver.options['mipgap'] = 0.41
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
    Num_cyc[t-1] = instance.Aux1[t].value 
Total_Num_cyc = sum(Num_cyc)
print("The total Number of cycles is:", Total_Num_cyc)




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