import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
T = 365*12
Q_max = 100.0 # total capacity of battery
Q_s = 10.0 #battery-capacity for selling in one timeslot
Q_b = 10.0 #battery-capacity for buying in one timeslot


# include buying and selling prices for every day (here still randomly)
c_buy  = np.random.uniform(30, 50, T)
c_sell = c_buy - 10

# -----------------------------
# Model
# -----------------------------
m = gp.Model("battery_trading")

# Variables
b = m.addVars(T, name="buy", vtype=GRB.BINARY)
s = m.addVars(T, name="sell", vtype=GRB.BINARY)

# -----------------------------
# Battery dynamics
# -----------------------------
m.addConstr(s[0] == 0) 

for t in range(1,T):
    m.addConstr(s[t]+b[t] <= 1)
    m.addConstr(s[t]*Q_s <= gp.quicksum(b[t2]*Q_b-s[t2]*Q_s for t2 in range(t))) # sell only what we have
    m.addConstr(gp.quicksum(b[t2]*Q_b-s[t2]*Q_s for t2 in range(t)) <= Q_max) # buy only until battery capacity

# -----------------------------
# Objective (maximize profit)
# -----------------------------
obj = gp.quicksum(
    c_sell[t] * s[t]
    - c_buy[t] * b[t]
    for t in range(T)
)

m.setObjective(obj, GRB.MAXIMIZE)

# -----------------------------
# Solve
# -----------------------------
m.optimize()

# -----------------------------
# Results
# -----------------------------
b_opt = np.array([b[t].X for t in range(T)])
s_opt = np.array([s[t].X for t in range(T)])
bat_opt = np.array([ sum(b[t2].X*Q_b-s[t2].X*Q_s for t2 in range(t)) for t in range(T)])

plt.plot(range(T), bat_opt)
plt.xlabel('Time')
plt.ylabel('Battery Level')
plt.title('Battery Level Over Time')
plt.show()

print("Optimal profit:", m.ObjVal)