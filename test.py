import osmnx as ox
import pandas as pd

# 1. Download the street network (for driving)
# We use a small area to keep the download fast
G = ox.graph_from_point((48.1374, 11.5755), dist=15000, network_type='drive')

# 2. Impute missing maxspeeds
# Real-world data often has holes. This function fills missing 'maxspeed' 
# based on the type of street (e.g., 'residential' gets 30km/h by default).
G = ox.add_edge_speeds(G)

# 3. Calculate Travel Time
# Uses the filled-in maxspeed and the length of the edge to calculate 
# how many seconds it takes to traverse.
G = ox.add_edge_travel_times(G)

# 4. Convert to GeoDataFrame for easier analysis
# This lets us treat the edges like a spreadsheet table.
nodes, edges = ox.graph_to_gdfs(G)

# --- CUSTOM CALCULATION: ROAD CAPACITY ---

# 'Lanes' data can be messy (strings, lists, or missing/NaN).
# We need to clean it to perform math.

# Fill missing lane data with a conservative default (e.g., 1 lane)
edges["lanes"] = edges["lanes"].fillna(1)

# Sometimes 'lanes' is a list (e.g., if lane count changes mid-segment). 
# We take the first value or the max value.
def clean_lanes(lanes_value):
    if isinstance(lanes_value, list):
        # Even simpler: just take the max if multiple values exist
        return int(max(lanes_value))
    try:
        return int(lanes_value)
    except:
        return 1 # Fallback default

edges["lanes_cleaned"] = edges["lanes"].apply(clean_lanes)

# Calculate Capacity
# Assumption: Theoretical capacity of ~1800 vehicles per hour per lane
edges["capacity_vph"] = edges["lanes_cleaned"] * 1800

# --- INSPECTION ---

# Select relevant columns to view the result
view_cols = ["name", "highway", "maxspeed", "travel_time", "lanes_cleaned", "capacity_vph"]

print(f"Graph has {len(edges)} edges.")
print("\nSample of Real-World Data & Calculated Capacity:")
print(edges[view_cols].head(10))
# Output into file:
edges[view_cols].to_csv("edge_capacity_sample.csv", index=False)

# OPTIONAL: Convert back to Graph if you need to run routing on the new capacity weights
# G_with_capacity = ox.graph_from_gdfs(nodes, edges)