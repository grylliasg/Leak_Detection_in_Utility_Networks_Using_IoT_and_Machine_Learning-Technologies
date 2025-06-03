import wntr
import pandas as pd

# Load EPANET Network
wn = wntr.network.WaterNetworkModel("networks/Net6.inp")

# Add Leakage
# leak_node = wn.get_node('10')
# leak_node.add_leak(wn, area=5, start_time=3600, end_time=12*3600)

# Run Simulation
sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

# Get pressure & flow & demand
pressure = results.node['pressure']
flowrate = results.link['flowrate']
demand = results.node["demand"]

# Save to CSV
# pressure.to_csv('pressure_no_leak.csv')
# flowrate.to_csv('flowrate_no_leak.csv')
demand.to_csv("demand_no_leak.csv")

print("Simulation completed!")
