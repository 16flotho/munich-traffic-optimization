import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
PLACE_NAME = "Munich, Germany"
# We use a radius to keep the download and calculation fast enough for a POC
# 2km radius around Marienplatz covers the dense city center where traffic matters
CENTER_POINT = (48.1372, 11.5755) 
DIST = 10000 
NUM_DRIVERS = 10000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

class TrafficSimulation:
    def __init__(self):
        print(f"1. Downloading/Loading map data for {PLACE_NAME} (Center)...")
        # Download the graph
        self.G = ox.graph_from_point(CENTER_POINT, dist=DIST, network_type='drive')
        
        # Add basic edge attributes (speed, travel_time)
        self.G = ox.add_edge_speeds(self.G)
        self.G = ox.add_edge_travel_times(self.G)
        
        # Clean up and set capacities
        self.prepare_graph()
        
        # Generate Random Demand (Origin -> Destination nodes)
        print(f"2. Generating {NUM_DRIVERS} random trips...")
        nodes = list(self.G.nodes())
        self.trips = []
        for _ in range(NUM_DRIVERS):
            u = random.choice(nodes)
            v = random.choice(nodes)
            while u == v:
                v = random.choice(nodes)
            self.trips.append((u, v))

    def prepare_graph(self):
        """
        Estimates capacity for roads based on their type (highway tag).
        This is crucial for the congestion function.
        """
        default_capacity = 500  # cars per hour per lane equivalent
        
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # 1. Ensure travel_time exists
            if 'travel_time' not in data:
                # Fallback if length/speed is missing
                length = data.get('length', 100)
                speed = data.get('speed_kph', 30)
                if isinstance(speed, list): speed = float(speed[0])
                data['travel_time'] = (length / 1000) / speed * 3600

            # 2. Impute Capacity based on highway type
            highway = data.get('highway', 'residential')
            if isinstance(highway, list): highway = highway[0]
            
            if highway in ['motorway', 'trunk', 'primary']:
                cap = 2000
            elif highway in ['secondary', 'tertiary']:
                cap = 1200
            else:
                cap = 600
            
            data['capacity'] = cap
            data['flow'] = 0 # Reset flow

    def bpr_function(self, free_flow_time, flow, capacity):
        """
        Bureau of Public Roads (BPR) Congestion Function.
        Time = T0 * (1 + alpha * (Flow/Capacity)^beta)
        """
        alpha = 0.15
        beta = 4.0
        return free_flow_time * (1 + alpha * ((flow / capacity) ** beta))

    def run_selfish_scenario(self):
        """
        SELFISH APPROACH (User Equilibrium approximation):
        Drivers ignore others. They look at the map, see the shortest path 
        based on speed limits (empty roads), and take it.
        We then calculate how much traffic jam this causes.
        """
        print("\n--- Running Scenario A: Selfish (Uncoordinated) ---")
        
        # Reset flows
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
            
        # 1. Route everyone based on static 'travel_time' (Free Flow)
        start_time = time.time()
        routes_with_keys = []  # Store (route, edge_keys) tuples
        
        # Pre-calculate weights (static)
        weight = 'travel_time'
        
        for i, (orig, dest) in enumerate(self.trips):
            try:
                route = nx.shortest_path(self.G, orig, dest, weight=weight)
                edge_keys = []
                
                # Add flow to the network and record which edges were used
                for j in range(len(route)-1):
                    u, v_node = route[j], route[j+1]
                    # Add flow to the specific edge (taking the shortest key if parallel)
                    # Simplified: take the first edge key found or min weight
                    edge_data = self.G.get_edge_data(u, v_node)
                    key = min(edge_data, key=lambda k: edge_data[k]['travel_time'])
                    self.G[u][v_node][key]['flow'] += 1
                    edge_keys.append(key)
                    
                routes_with_keys.append((route, edge_keys))
            except nx.NetworkXNoPath:
                continue

        # 2. Calculate Resulting Travel Times (Congested)
        total_system_time = 0
        
        # We iterate over the ACTUAL routes taken and calculate the time experienced
        # considering the congestion created by everyone.
        for route, edge_keys in routes_with_keys:
            trip_time = 0
            for j in range(len(route)-1):
                u, v_node = route[j], route[j+1]
                key = edge_keys[j]
                data = self.G[u][v_node][key]
                
                # Apply BPR Congestion
                actual_edge_time = self.bpr_function(data['travel_time'], data['flow'], data['capacity'])
                trip_time += actual_edge_time
            total_system_time += trip_time
            
        avg_time = total_system_time / len(routes_with_keys) if routes_with_keys else 0
        print(f"Selfish Simulation Complete.")
        print(f"Average Travel Time per Driver: {avg_time/60:.2f} minutes")
        return avg_time

    def run_social_scenario(self):
        """
        SOCIAL APPROACH (System Optimal / Coordinated):
        The system routes drivers one by one. Before routing Driver X, 
        it checks the traffic caused by Drivers 1 to X-1.
        It updates edge costs to reflect current congestion.
        Driver X is sent on the *currently* fastest path, which diverts them
        from jams created by previous drivers.
        """
        print("\n--- Running Scenario B: Social (Coordinated Routing) ---")
        
        # Reset flows
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
            d['current_cost'] = d['travel_time'] # Reset dynamic weight
            
        total_system_time = 0
        num_successful_trips = 0
        
        # Route drivers incrementally and calculate their ACTUAL experienced time
        for i, (orig, dest) in enumerate(self.trips):
            try:
                # Route based on CURRENT dynamic cost
                route = nx.shortest_path(self.G, orig, dest, weight='current_cost')
                
                # Calculate the time THIS driver experiences with CURRENT congestion
                trip_time = 0
                
                # Update Network State Immediately
                for j in range(len(route)-1):
                    u, v_node = route[j], route[j+1]
                    # Find best edge
                    edge_data = self.G.get_edge_data(u, v_node)
                    # In social planning, we must select the edge consistent with the routing
                    key = min(edge_data, key=lambda k: edge_data[k]['current_cost'])
                    
                    data = self.G[u][v_node][key]
                    
                    # This driver experiences the CURRENT cost (before adding their own flow)
                    trip_time += data['current_cost']
                    
                    # Increment flow
                    self.G[u][v_node][key]['flow'] += 1
                    
                    # UPDATE COST for the next driver
                    new_time = self.bpr_function(data['travel_time'], data['flow'], data['capacity'])
                    data['current_cost'] = new_time
                
                total_system_time += trip_time
                num_successful_trips += 1
                    
            except nx.NetworkXNoPath:
                continue

        # Calculate average
        avg_time = total_system_time / num_successful_trips if num_successful_trips > 0 else 0
        print(f"Social Simulation Complete.")
        print(f"Average Travel Time per Driver: {avg_time/60:.2f} minutes")
        return avg_time
    
    def visualize_difference(self):
        print("\nGenerating visualization (this may take a moment)...")
        # Create a plot showing which roads are used
        # For simplicity in this script, we just print stats, 
        # but we can use ox.plot_graph_routes if needed.
        pass

if __name__ == "__main__":
    sim = TrafficSimulation()
    
    t_selfish = sim.run_selfish_scenario()
    t_social = sim.run_social_scenario()
    
    print(f"\n{'='*30}")
    print(f"FINAL RESULTS (Munich Center, {NUM_DRIVERS} drivers)")
    print(f"{'='*30}")
    print(f"Selfish Strategy Avg Time: {t_selfish/60:.2f} min")
    print(f"Social Strategy Avg Time:  {t_social/60:.2f} min")
    
    diff = t_selfish - t_social
    improvement = (diff / t_selfish) * 100
    
    print(f"Time Saved: {diff/60:.2f} min per person")
    print(f"Improvement: {improvement:.2f}%")
    
    if t_social < t_selfish:
        print("\nCONCLUSION: By coordinating routes and diverting some traffic")
        print("to 'sub-optimal' free roads, the overall system moved faster.")
    else:
        print("\nNOTE: Increase NUM_DRIVERS to see the congestion effects more clearly.")