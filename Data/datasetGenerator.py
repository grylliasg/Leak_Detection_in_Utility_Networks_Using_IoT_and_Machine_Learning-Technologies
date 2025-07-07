import wntr
import pandas as pd

# Load EPANET Network
wn = wntr.network.WaterNetworkModel("networks/Net3.inp")

# # Add Leakage
# leak_node = wn.get_node('15')
# leak_node.add_leak(wn, area=0.05, start_time=2*3600, end_time=4*3600)

# # Connected links with leaked node:
# node_name = '15'
# connected_links = wn.get_links_for_node(node_name)
# print(f"Ο κόμβος {node_name} συνδέεται με τα links: {connected_links}")

# Run Simulation
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

# Get pressure & flow & demand
pressure = results.node['pressure']
flowrate = results.link['flowrate']
demand = results.node["demand"]

# Save to CSV
pressure.to_csv('leakage/pressure__leak(Net 3).csv')
flowrate.to_csv('leakage/flowrate__leak(Net3).csv')
demand.to_csv("leakage/demand__leak(Net 3).csv")

print("Simulation completed!")


