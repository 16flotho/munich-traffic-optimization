"""
Munich to Taufkirchen Traffic Simulation
=========================================

Simulates traffic participants with random start/goal locations and arrival times.
Compares two routing strategies:
1. SELFISH: Each driver takes their shortest path (ignoring others)
2. SOCIAL: Coordinated routing that diverts some traffic to alternative routes
   to minimize overall travel time

The simulation covers Munich center and the tangent highway (A995) to Taufkirchen.
"""

import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time as time_module

# --- CONFIGURATION ---
MUNICH_CENTER = (48.1374, 11.5755)  # Marienplatz

# Area covering Munich center to Taufkirchen (roughly 15km radius from Munich center)
DIST_FROM_CENTER = 15000  # meters

NUM_PARTICIPANTS = 1000  # Number of traffic participants
SEED = 42

# Time window for arrivals (morning rush hour)
START_TIME = datetime(2025, 11, 22, 7, 0)  # 7:00 AM
END_TIME = datetime(2025, 11, 22, 9, 0)    # 9:00 AM

# Congestion modeling parameters
ENABLE_TRAFFIC_LIGHTS = True  # Add delay at intersections
ENABLE_INTERSECTION_DELAY = True  # Model intersection complexity
BPR_ALPHA = 0.5  # Increased from 0.15 for more pronounced congestion
BPR_BETA = 4.0   # Power factor for congestion curve

random.seed(SEED)
np.random.seed(SEED)


class TrafficParticipant:
    """Represents a single traffic participant with origin, destination, and desired arrival time."""
    
    def __init__(self, participant_id, origin_node, dest_node, arrival_time):
        self.id = participant_id
        self.origin = origin_node
        self.destination = dest_node
        self.desired_arrival = arrival_time
        self.route = None
        self.actual_travel_time = 0  # seconds
        self.free_flow_time = 0  # seconds
        
    def __repr__(self):
        return f"Participant {self.id}: {self.origin} -> {self.destination} @ {self.desired_arrival.strftime('%H:%M')}"


class MunichTaufkirchenSimulation:
    """Traffic simulation for Munich to Taufkirchen area."""
    
    def __init__(self, num_participants=NUM_PARTICIPANTS):
        self.num_participants = num_participants
        self.G = None
        self.participants = []
        

        print(f"Using parameters: BPR_ALPHA={self.bpr_alpha}, BPR_BETA={self.bpr_beta}, "
              f"Traffic Lights={'Enabled' if self.enable_traffic_lights else 'Disabled'}, "
              f"Intersection Delay={'Enabled' if self.enable_intersection_delay else 'Disabled'}")
        print("="*60)
        print("Munich-Taufkirchen Traffic Simulation")
        print("="*60)
        
        # Step 1: Load road network
        self.load_network()
        
        # Step 2: Generate participants
        self.generate_participants()
        
    def load_network(self):
        """Download and prepare the road network from Munich to Taufkirchen."""
        print(f"\n[1/4] Loading road network (Munich + A995 to Taufkirchen)...")
        print(f"      This covers approx. {DIST_FROM_CENTER/1000:.1f}km radius from center...")
        
        try:
            # Download road network centered on Munich
            self.G = ox.graph_from_point(
                MUNICH_CENTER, 
                dist=DIST_FROM_CENTER, 
                network_type='drive',
                simplify=True
            )
            
            print(f"      ✓ Loaded network with {len(self.G.nodes())} nodes and {len(self.G.edges())} edges")
            
            # Add speed and travel time attributes
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            
            # Set capacity for each road segment
            self._prepare_graph()
            
        except Exception as e:
            print(f"      ✗ Error loading network: {e}")
            raise
    
    def _prepare_graph(self):
        """Add capacity estimates to each edge based on road type."""
        # print(f"      Adding capacity estimates based on road types...")
        
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # Ensure travel_time exists
            if 'travel_time' not in data:
                length = data.get('length', 100)  # meters
                speed = data.get('speed_kph', 30)
                if isinstance(speed, list): 
                    speed = float(speed[0])
                data['travel_time'] = (length / 1000) / speed * 3600  # seconds
            
            # Estimate capacity based on road type
            highway = data.get('highway', 'residential')
            if isinstance(highway, list): 
                highway = highway[0]
            
            # Capacity in vehicles per hour (reduced for more realistic congestion)
            if highway in ['motorway', 'motorway_link']:
                capacity = 1800  # Highway lanes (reduced for more congestion)
            elif highway in ['trunk', 'trunk_link']:
                capacity = 1400  # Major arterial
            elif highway in ['primary', 'primary_link']:
                capacity = 1000
            elif highway in ['secondary', 'secondary_link']:
                capacity = 600
            elif highway in ['tertiary', 'tertiary_link']:
                capacity = 400
            else:
                capacity = 250  # Residential and smaller roads
            
            data['capacity'] = capacity
            data['flow'] = 0  # Initialize flow counter
            data['free_flow_time'] = data['travel_time']  # Store original time
            
            # Add traffic light delay for non-highway roads
            data['has_traffic_light'] = highway not in ['motorway', 'motorway_link', 'trunk', 'trunk_link']
            
            # Intersection penalty based on road type
            if highway in ['primary', 'primary_link']:
                data['intersection_penalty'] = 15  # seconds
            elif highway in ['secondary', 'secondary_link']:
                data['intersection_penalty'] = 20
            elif highway in ['tertiary', 'tertiary_link']:
                data['intersection_penalty'] = 25
            else:
                data['intersection_penalty'] = 30  # Residential - more complex intersections
    
    def generate_participants(self):
        """Generate random traffic participants with origins, destinations, and arrival times."""
        print(f"\n[2/4] Generating {self.num_participants} traffic participants...")
        
        nodes = list(self.G.nodes())
        
        # Filter nodes to ensure they're reasonably distributed
        # Get nodes in different zones
        north_munich = []
        south_munich = []
        east_munich = []
        west_munich = []
        taufkirchen_area = []
        
        for node in nodes:
            lat = self.G.nodes[node]['y']
            lon = self.G.nodes[node]['x']
            
            # Categorize nodes
            if lat > 48.15:  # North Munich
                north_munich.append(node)
            elif lat < 48.05:  # South (including Taufkirchen)
                if lat < 48.06 and abs(lon - 11.6175) < 0.05:
                    taufkirchen_area.append(node)
                else:
                    south_munich.append(node)
            
            if lon < 11.5:  # West
                west_munich.append(node)
            elif lon > 11.6:  # East
                east_munich.append(node)
        
        print(f"      Node distribution:")
        print(f"      - North Munich: {len(north_munich)} nodes")
        print(f"      - South Munich: {len(south_munich)} nodes")
        print(f"      - Taufkirchen area: {len(taufkirchen_area)} nodes")
        print(f"      - East: {len(east_munich)} nodes")
        print(f"      - West: {len(west_munich)} nodes")
        
        # Generate participants with diverse origin-destination pairs
        for i in range(self.num_participants):
            # Random arrival time in rush hour window
            minutes_offset = random.randint(0, int((END_TIME - START_TIME).total_seconds() / 60))
            arrival_time = START_TIME + timedelta(minutes=minutes_offset)
            
            # Create diverse traffic patterns
            pattern = random.choice(['commute_to_center', 'commute_from_center', 'cross_city', 'random'])
            
            if pattern == 'commute_to_center' and (taufkirchen_area or south_munich):
                # From suburbs/Taufkirchen to center
                origin_pool = taufkirchen_area if taufkirchen_area and random.random() > 0.5 else south_munich
                if origin_pool:
                    origin = random.choice(origin_pool)
                    dest = random.choice(north_munich) if north_munich else random.choice(nodes)
                else:
                    origin, dest = self._random_pair(nodes)
            
            elif pattern == 'commute_from_center' and north_munich and south_munich:
                # From center to suburbs
                origin = random.choice(north_munich)
                dest = random.choice(south_munich + taufkirchen_area) if (south_munich + taufkirchen_area) else random.choice(nodes)
            
            elif pattern == 'cross_city' and east_munich and west_munich:
                # East-West traffic
                origin = random.choice(east_munich)
                dest = random.choice(west_munich)
            
            else:
                # Random pair
                origin, dest = self._random_pair(nodes)
            
            participant = TrafficParticipant(i, origin, dest, arrival_time)
            self.participants.append(participant)
        
        print(f"      ✓ Generated {len(self.participants)} participants")
        
        # Show sample participants
        print(f"\n      Sample participants:")
        for p in self.participants[:5]:
            print(f"      {p}")
    
    def _random_pair(self, nodes):
        """Generate a random origin-destination pair ensuring they're different."""
        origin = random.choice(nodes)
        dest = random.choice(nodes)
        attempts = 0
        while origin == dest and attempts < 10:
            dest = random.choice(nodes)
            attempts += 1
        return origin, dest
    
    def bpr_congestion_function(self, free_flow_time, flow, capacity):
        """
        Bureau of Public Roads (BPR) congestion function.
        
        Calculates actual travel time based on traffic flow:
        time = free_flow_time * (1 + α * (flow/capacity)^β)
        
        Enhanced parameters for more pronounced congestion effects.
        """
        if capacity <= 0:
            return free_flow_time
        
        congestion_ratio = flow / capacity
        congestion_factor = 1 + BPR_ALPHA * (congestion_ratio ** BPR_BETA)
        
        return free_flow_time * congestion_factor
    
    def calculate_edge_time(self, edge_data, include_delays=True):
        """
        Calculate total time for an edge including congestion, traffic lights, and intersections.
        
        Args:
            edge_data: Dictionary with edge attributes
            include_delays: Whether to include traffic light and intersection delays
        
        Returns:
            Total travel time in seconds
        """
        # Base congested time
        base_time = self.bpr_congestion_function(
            edge_data['free_flow_time'],
            edge_data['flow'],
            edge_data['capacity']
        )
        
        if not include_delays:
            return base_time
        
        # Add traffic light delay (average wait time)
        traffic_light_delay = 0
        if ENABLE_TRAFFIC_LIGHTS and edge_data.get('has_traffic_light', True):
            # Average wait is half the cycle time, increases with congestion
            flow_ratio = edge_data['flow'] / edge_data['capacity']
            base_wait = 20  # seconds
            traffic_light_delay = base_wait * (1 + flow_ratio * 0.5)
            print(f"      Adding traffic light delay for edge with flow {edge_data['flow']} and capacity {edge_data['capacity']}")
        
        # Add intersection complexity delay
        intersection_delay = 0
        if ENABLE_INTERSECTION_DELAY:
            intersection_delay = edge_data.get('intersection_penalty', 0)
            # Scale by congestion
            flow_ratio = edge_data['flow'] / edge_data['capacity']
            intersection_delay *= (1 + flow_ratio * 0.3)
        
        return base_time + traffic_light_delay + intersection_delay
    
    def run_selfish_routing(self):
        """
        SELFISH ROUTING: Everyone takes their individually optimal shortest path.
        
        Each participant routes based on free-flow travel times (empty roads),
        ignoring the impact on other drivers. We then calculate the resulting
        congestion and actual travel times.
        """
        print(f"\n[3/4] Running SELFISH routing simulation...")
        print(f"      (Each driver takes their shortest path, ignoring others)")
        
        # Reset all flows
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
        
        # Route everyone on shortest paths (free-flow)
        successful_routes = 0
        failed_routes = 0
        
        print(f"      Calculating shortest paths for all participants...")
        
        for participant in self.participants:
            try:
                # Find shortest path based on free-flow travel time
                route = nx.shortest_path(
                    self.G, 
                    participant.origin, 
                    participant.destination, 
                    weight='travel_time'
                )
                
                participant.route = route
                
                # Add this route's flow to the network
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    # Get edge data (handle multi-edges)
                    edge_data = self.G.get_edge_data(u, v)
                    # Pick edge with minimum travel time
                    key = min(edge_data, key=lambda k: edge_data[k]['travel_time'])
                    self.G[u][v][key]['flow'] += 1
                
                successful_routes += 1
                
            except nx.NetworkXNoPath:
                failed_routes += 1
                participant.route = None
        
        print(f"      ✓ Routed {successful_routes} participants ({failed_routes} failed)")
        
        # Calculate actual travel times with congestion
        print(f"      Calculating actual travel times with congestion...")
        
        total_time = 0
        total_free_flow = 0
        
        for participant in self.participants:
            if participant.route is None:
                continue
            
            route = participant.route
            actual_time = 0
            free_time = 0
            
            for j in range(len(route) - 1):
                u, v = route[j], route[j + 1]
                edge_data = self.G.get_edge_data(u, v)
                key = min(edge_data, key=lambda k: edge_data[k]['travel_time'])
                data = self.G[u][v][key]
                
                # Calculate congested travel time with all delays
                congested_time = self.calculate_edge_time(data, include_delays=True)
                free_time_edge = data['free_flow_time']
                
                actual_time += congested_time
                free_time += free_time_edge
            
            participant.actual_travel_time = actual_time
            participant.free_flow_time = free_time
            
            total_time += actual_time
            total_free_flow += free_time
        
        avg_time = total_time / successful_routes if successful_routes > 0 else 0
        avg_free_flow = total_free_flow / successful_routes if successful_routes > 0 else 0
        
        # Calculate congestion statistics
        max_flow_ratio = 0
        congested_edges = 0
        for u, v, k, d in self.G.edges(keys=True, data=True):
            flow_ratio = d['flow'] / d['capacity']
            max_flow_ratio = max(max_flow_ratio, flow_ratio)
            if flow_ratio > 0.7:  # Over 70% capacity = congested
                congested_edges += 1
        
        print(f"\n      SELFISH ROUTING RESULTS:")
        print(f"      - Average free-flow time: {avg_free_flow/60:.2f} minutes")
        print(f"      - Average actual time:    {avg_time/60:.2f} minutes")
        print(f"      - Average delay:          {(avg_time - avg_free_flow)/60:.2f} minutes")
        print(f"      - Congestion factor:      {avg_time/avg_free_flow:.2f}x")
        print(f"      - Max edge utilization:   {max_flow_ratio*100:.1f}%")
        print(f"      - Congested edges:        {congested_edges}")
        
        # Store flow data for visualization
        flow_data = {}
        for u, v, k, d in self.G.edges(keys=True, data=True):
            flow_data[(u, v, k)] = d['flow']
        
        return {
            'avg_time': avg_time,
            'avg_free_flow': avg_free_flow,
            'total_time': total_time,
            'successful_routes': successful_routes,
            'flow_data': flow_data
        }
    
    def run_social_routing(self):
        """
        SOCIAL ROUTING: Coordinated routing to minimize total system travel time.
        
        Participants are routed sequentially, each considering the current traffic
        from previous participants. This naturally diverts some traffic to alternative
        routes, reducing congestion on main corridors.
        """
        print(f"\n[4/4] Running SOCIAL (coordinated) routing simulation...")
        print(f"      (Routes consider current congestion, naturally balancing load)")
        
        # Reset all flows
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
            d['current_cost'] = d['travel_time']  # Dynamic cost
        
        # Sort participants by desired arrival time (earlier arrivals get priority)
        # sorted_participants = sorted(self.participants, key=lambda p: p.desired_arrival)
        # make random order to simulate simultaneous departures
        sorted_participants = random.sample(self.participants, len(self.participants))

        successful_routes = 0
        failed_routes = 0
        
        total_time = 0
        total_free_flow = 0
        
        print(f"      Routing participants in order of arrival time...")
        
        for i, participant in enumerate(sorted_participants):
            try:
                # Find shortest path based on CURRENT congestion
                route = nx.shortest_path(
                    self.G, 
                    participant.origin, 
                    participant.destination, 
                    weight='current_cost'
                )
                
                participant.route = route
                
                # Calculate travel time and update network state
                actual_time = 0
                free_time = 0
                
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    edge_data = self.G.get_edge_data(u, v)
                    # Pick edge with minimum current cost
                    key = min(edge_data, key=lambda k: edge_data[k]['current_cost'])
                    data = self.G[u][v][key]
                    
                    # Record time this participant experiences
                    actual_time += data['current_cost']
                    free_time += data['free_flow_time']
                    
                    # Update flow
                    data['flow'] += 1
                    
                    # Update cost for next participant (with all delays)
                    new_cost = self.calculate_edge_time(data, include_delays=True)
                    data['current_cost'] = new_cost
                
                participant.actual_travel_time = actual_time
                participant.free_flow_time = free_time
                
                total_time += actual_time
                total_free_flow += free_time
                successful_routes += 1
                
            except nx.NetworkXNoPath:
                failed_routes += 1
                participant.route = None
        
        print(f"      ✓ Routed {successful_routes} participants ({failed_routes} failed)")
        
        avg_time = total_time / successful_routes if successful_routes > 0 else 0
        avg_free_flow = total_free_flow / successful_routes if successful_routes > 0 else 0
        
        # Calculate congestion statistics
        max_flow_ratio = 0
        congested_edges = 0
        for u, v, k, d in self.G.edges(keys=True, data=True):
            flow_ratio = d['flow'] / d['capacity']
            max_flow_ratio = max(max_flow_ratio, flow_ratio)
            if flow_ratio > 0.7:  # Over 70% capacity = congested
                congested_edges += 1
        
        print(f"\n      SOCIAL ROUTING RESULTS:")
        print(f"      - Average free-flow time: {avg_free_flow/60:.2f} minutes")
        print(f"      - Average actual time:    {avg_time/60:.2f} minutes")
        print(f"      - Average delay:          {(avg_time - avg_free_flow)/60:.2f} minutes")
        print(f"      - Congestion factor:      {avg_time/avg_free_flow:.2f}x")
        print(f"      - Max edge utilization:   {max_flow_ratio*100:.1f}%")
        print(f"      - Congested edges:        {congested_edges}")
        
        # Store flow data for visualization
        flow_data = {}
        for u, v, k, d in self.G.edges(keys=True, data=True):
            flow_data[(u, v, k)] = d['flow']
        
        return {
            'avg_time': avg_time,
            'avg_free_flow': avg_free_flow,
            'total_time': total_time,
            'successful_routes': successful_routes,
            'flow_data': flow_data
        }
    
    def compare_results(self, selfish_results, social_results):
        """Compare and display results from both routing strategies."""
        print(f"\n{'='*60}")
        print(f"FINAL COMPARISON")
        print(f"{'='*60}")
        
        print(f"\nParticipants: {self.num_participants}")
        print(f"Network size: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges")
        
        print(f"\n{'Strategy':<20} {'Avg Time':<15} {'Total Time':<15} {'vs Free-Flow':<15}")
        print(f"{'-'*65}")
        
        selfish_avg = selfish_results['avg_time'] / 60
        social_avg = social_results['avg_time'] / 60
        selfish_total = selfish_results['total_time'] / 3600
        social_total = social_results['total_time'] / 3600
        
        selfish_ff = selfish_results['avg_free_flow'] / 60
        social_ff = social_results['avg_free_flow'] / 60
        
        print(f"{'SELFISH':<20} {selfish_avg:>8.2f} min   {selfish_total:>8.2f} hrs    +{(selfish_avg-selfish_ff):>6.2f} min")
        print(f"{'SOCIAL':<20} {social_avg:>8.2f} min   {social_total:>8.2f} hrs    +{(social_avg-social_ff):>6.2f} min")
        
        # Calculate improvement
        time_saved = selfish_avg - social_avg
        percent_improvement = (time_saved / selfish_avg * 100) if selfish_avg > 0 else 0
        total_saved = selfish_total - social_total
        
        print(f"\n{'IMPROVEMENT':<20} {time_saved:>8.2f} min   {total_saved:>8.2f} hrs    {percent_improvement:>6.2f}%")
        
        print(f"\n{'='*60}")
        
        if time_saved > 0:
            print(f"✓ SOCIAL routing is BETTER!")
            print(f"  By coordinating routes and diverting some traffic to")
            print(f"  alternative paths, average travel time decreased by")
            print(f"  {time_saved:.2f} minutes ({percent_improvement:.1f}%).")
            print(f"\n  Total time saved across all participants:")
            print(f"  {total_saved*60:.1f} minutes = {total_saved:.2f} hours")
        elif time_saved < 0:
            print(f"✗ SELFISH routing was better in this simulation.")
            print(f"  (May indicate insufficient congestion or network structure)")
        else:
            print(f"= Both strategies performed equally.")
        
        print(f"\n{'='*60}")
    
    def visualize_results(self, scenario='selfish', flow_data=None):
        """Create visualization of traffic flow (optional).
        
        Args:
            scenario: Name of the scenario ('selfish' or 'social')
            flow_data: Dictionary mapping (u, v, k) tuples to flow values
        """
        print(f"\n[Optional] Generating visualization for {scenario} scenario...")
        
        try:
            # Use provided flow data instead of current graph state
            if flow_data is None:
                # Fallback to current graph state if no flow data provided
                edge_flows = {}
                for u, v, k, d in self.G.edges(keys=True, data=True):
                    flow = d.get('flow', 0)
                    edge_flows[(u, v, k)] = flow
            else:
                edge_flows = flow_data
            
            # Create color map based on flow
            max_flow = max(edge_flows.values()) if edge_flows else 1
            
            edge_colors = []
            edge_widths = []
            
            for u, v, k in self.G.edges(keys=True):
                flow = edge_flows.get((u, v, k), 0)
                
                if flow == 0:
                    edge_colors.append('#e0e0e0')
                    edge_widths.append(0.3)
                else:
                    # Color intensity based on flow
                    intensity = min(flow / max_flow, 1.0)
                    if intensity < 0.2:
                        edge_colors.append('#90EE90')  # Light green
                        edge_widths.append(0.5)
                    elif intensity < 0.5:
                        edge_colors.append('#FFD700')  # Yellow
                        edge_widths.append(1.0)
                    elif intensity < 0.8:
                        edge_colors.append('#FF8C00')  # Orange
                        edge_widths.append(1.5)
                    else:
                        edge_colors.append('#FF0000')  # Red
                        edge_widths.append(2.0)
            
            fig, ax = ox.plot_graph(
                self.G,
                node_size=0,
                edge_color=edge_colors,
                edge_linewidth=edge_widths,
                bgcolor='white',
                show=False,
                close=False,
                figsize=(12, 12)
            )
            
            ax.set_title(f"Traffic Flow - {scenario.upper()} Routing\n(Munich to Taufkirchen)", 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            filename = f"traffic_flow_{scenario}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"      ✓ Saved visualization to {filename}")
            plt.close(fig)
            
        except Exception as e:
            print(f"      ✗ Visualization error: {e}")


def main():
    """Main simulation entry point."""
    
    # Create simulation
    sim = MunichTaufkirchenSimulation(num_participants=NUM_PARTICIPANTS)
    
    # Run both scenarios
    selfish_results = sim.run_selfish_routing()
    social_results = sim.run_social_routing()
    
    # Compare results
    sim.compare_results(selfish_results, social_results)
    
    # Optional: Visualize (comment out if not needed)
    sim.visualize_results('selfish', flow_data=selfish_results['flow_data'])
    sim.visualize_results('social', flow_data=social_results['flow_data'])
    
    print(f"\n✓ Simulation complete!")


if __name__ == "__main__":
    main()
