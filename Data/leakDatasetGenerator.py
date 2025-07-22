import wntr
import pandas as pd
import random
import numpy as np

# === 1. Load Network ===
wn = wntr.network.WaterNetworkModel("Data/Net3.inp")

# === 2. Define leak nodes and random leak windows ===
leak_nodes = ['15', '35', '601', '103', '119', '120', '201', '225', '257', '267']
leak_windows = {}

# Τυχαία χρονικά παράθυρα διαρροής για κάθε κόμβο (μέσα στις 12 ώρες)
for node in leak_nodes:
    start_hour = random.randint(1, 160)         # Ώρα έναρξης: μεταξύ 1h και 160h
    duration = random.randint(10, 15) * 3600      # Διάρκεια
    start_time = start_hour * 3600
    end_time = start_time + duration
    leak_windows[node] = (start_time, end_time)

    # === 3. Add Leak ===
    leak_node = wn.get_node(node)
    leak_node.add_leak(wn, area=5e-4, start_time=start_time, end_time=end_time)

# Εμφάνιση των χρονικών διαστημάτων διαρροής
print("Leak periods per node:")
for node, (start, end) in leak_windows.items():
    print(f"Node {node}: {start//3600}h–{end//3600}h")

# === 4. Run Simulation ===
sim = wntr.sim.WNTRSimulator(wn)
results = sim.run_sim()

# === 5. Extract Results ===
pressure = results.node['pressure']  # DataFrame: rows = time, columns = nodes

# === 6. Create 'leak' label with node name(s) ===
leak_labels = []

for t in pressure.index:
    leaking_nodes = [node for node, (start, end) in leak_windows.items() if start <= t <= end]
    if leaking_nodes:
        leak_labels.append(1)  # Πολλαπλοί κόμβοι αν υπάρχουν
    else:
        leak_labels.append(0)

pressure['leak'] = leak_labels

# === 7. Save to CSV ===
pressure.to_csv('Data/Leaks/pressure_leak(Net3).csv')
print("CSV saved: leaks/pressure_leak(Net3).csv")
