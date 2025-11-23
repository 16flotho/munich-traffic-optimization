"""
Munich Traffic Simulation
=========================

This module implements the MunichSimulation class, which inherits from BaseSimulation.
It includes methods for running the traffic simulation from Munich to Taufkirchen,
comparing selfish and social routing strategies.
"""

from src.core.base_simulation import BaseSimulation
from src.core.traffic_participant import TrafficParticipant
from src.core.config import MUNICH_CENTER, DIST_FROM_CENTER, NUM_PARTICIPANTS
import random
import numpy as np
import copy
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache


class MunichSimulation(BaseSimulation):
    """Traffic simulation for Munich to Taufkirchen area."""
    
    def __init__(self, num_participants=NUM_PARTICIPANTS, bpr_alpha=0.5, bpr_beta=4.0, 
                 enable_traffic_lights=True, enable_intersection_delay=True, seed=42):
        """
        Initialize Munich simulation with custom parameters.
        
        Args:
            num_participants: Number of traffic participants
            bpr_alpha: BPR congestion sensitivity parameter
            bpr_beta: BPR congestion power parameter
            enable_traffic_lights: Whether to enable traffic light delays
            enable_intersection_delay: Whether to enable intersection delays
            seed: Random seed for reproducibility
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Validate parameters
        if not 0.0 <= bpr_alpha <= 2.0:
            raise ValueError(f"BPR Alpha must be between 0.0 and 2.0, got {bpr_alpha}")
        if not 1.0 <= bpr_beta <= 10.0:
            raise ValueError(f"BPR Beta must be between 1.0 and 10.0, got {bpr_beta}")
        
        # Store parameters
        self.bpr_alpha = bpr_alpha
        self.bpr_beta = bpr_beta
        self.enable_traffic_lights = enable_traffic_lights
        self.enable_intersection_delay = enable_intersection_delay
        self.seed = seed
        
        # Initialize base simulation
        super().__init__(num_participants)
        
        print(f"Using parameters: BPR_ALPHA={self.bpr_alpha}, BPR_BETA={self.bpr_beta}, "
              f"Traffic Lights={'Enabled' if self.enable_traffic_lights else 'Disabled'}, "
              f"Intersection Delay={'Enabled' if self.enable_intersection_delay else 'Disabled'}")
    
    def load_network(self):
        """Download and prepare the road network from Munich to Taufkirchen."""
        print(f"\n[1/4] Loading road network (Munich + A995 to Taufkirchen)...")
        print(f"      This covers approx. {DIST_FROM_CENTER/1000:.1f}km radius from center...")
        
        try:
            # Download street network
            import osmnx as ox
            self.G = ox.graph_from_point(
                MUNICH_CENTER,
                dist=DIST_FROM_CENTER,
                network_type='drive',
                simplify=True
            )
            
            print(f"      ✓ Network loaded: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
            
            # Prepare graph with capacities
            self._prepare_graph()
            
        except Exception as e:
            print(f"      ✗ Error loading network: {e}")
            raise
    
    def _prepare_graph(self):
        """Add capacity estimates to each edge based on road type."""
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # Set capacity based on road type
            highway_type = data.get('highway', 'residential')
            
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            # Capacity estimates (vehicles per hour per lane)
            # Reduced values to increase congestion sensitivity
            capacity_map = {
                'motorway': 80,
                'motorway_link': 72,
                'trunk': 60,
                'trunk_link': 48,
                'primary': 40,
                'primary_link': 30,
                'secondary': 30,
                'secondary_link': 24,
                'tertiary': 24,
                'residential': 16,
                'unclassified': 16,
                'living_street': 8
            }
            
            capacity = capacity_map.get(highway_type, 10)
            lanes = data.get('lanes', 1)
            
            if isinstance(lanes, list):
                lanes = int(lanes[0]) if lanes[0].isdigit() else 1
            elif isinstance(lanes, str):
                lanes = int(lanes) if lanes.isdigit() else 1
            
            data['capacity'] = capacity * lanes
            
            # Store original travel time (free flow)
            if 'travel_time' not in data:
                length = data.get('length', 100)
                speed = data.get('maxspeed', 50)
                
                if isinstance(speed, list):
                    speed = speed[0]
                if isinstance(speed, str):
                    speed = float(speed.split()[0]) if speed.split()[0].replace('.', '').isdigit() else 50
                
                data['travel_time'] = (length / 1000) / speed * 3600  # seconds
    
    def generate_participants(self):
        """Generate random traffic participants with origins, destinations, and arrival times."""
        from src.core.traffic_participant import TrafficParticipant
        from src.core.config import START_TIME, END_TIME
        from datetime import timedelta
        
        print(f"\n[2/4] Generating {self.num_participants} traffic participants...")
        
        nodes = list(self.G.nodes())
        
        # Filter nodes by location zones
        north_munich = []
        south_munich = []
        taufkirchen_area = []
        
        for node in nodes:
            lat, lon = self.G.nodes[node]['y'], self.G.nodes[node]['x']
            
            # Simple geographic filtering
            if lat > MUNICH_CENTER[0] + 0.05:
                north_munich.append(node)
            elif lat < MUNICH_CENTER[0] - 0.08:
                taufkirchen_area.append(node)
            else:
                south_munich.append(node)
        
        print(f"      Node distribution:")
        print(f"      - North Munich: {len(north_munich)} nodes")
        print(f"      - South Munich: {len(south_munich)} nodes")
        print(f"      - Taufkirchen area: {len(taufkirchen_area)} nodes")
        
        # Generate participants
        for i in range(self.num_participants):
            # 50% north to south traffic, 50% reverse
            if random.random() < 0.5:
                origin = random.choice(north_munich) if north_munich else random.choice(nodes)
                destination = random.choice(taufkirchen_area) if taufkirchen_area else random.choice(nodes)
            else:
                origin = random.choice(taufkirchen_area) if taufkirchen_area else random.choice(nodes)
                destination = random.choice(north_munich) if north_munich else random.choice(nodes)



            # Random arrival time in time window
            time_diff = (END_TIME - START_TIME).total_seconds()
            random_seconds = random.uniform(0, time_diff)
            arrival_time = START_TIME + timedelta(seconds=random_seconds)
            
            participant = TrafficParticipant(i, origin, destination, arrival_time)
            self.participants.append(participant)
        
        print(f"      ✓ Generated {len(self.participants)} participants")
    
    def bpr_congestion_function(self, free_flow_time, flow, capacity):
        """
        Bureau of Public Roads (BPR) congestion function with amplified impact.
        
        Args:
            free_flow_time: Travel time with no congestion (seconds)
            flow: Current traffic flow on the edge
            capacity: Maximum capacity of the edge
        
        Returns:
            Congested travel time (seconds)
        """
        if capacity == 0:
            return free_flow_time
        
        flow_ratio = flow / capacity
        
        # Amplified BPR function: multiply alpha effect by 2x for more dramatic impact
        # This makes the parameter changes more visible in results
        congestion_factor = 1 + (2.0 * self.bpr_alpha) * (flow_ratio ** self.bpr_beta)
        
        return free_flow_time * congestion_factor
    
    def calculate_edge_time(self, edge_data, include_delays=True):
        """
        Calculate actual travel time for an edge considering congestion and delays.
        
        Args:
            edge_data: Dictionary containing edge attributes
            include_delays: Whether to include traffic light and intersection delays
        
        Returns:
            Total travel time (seconds)
        """
        # Base travel time
        free_flow_time = edge_data.get('travel_time', 60)
        flow = edge_data.get('flow', 0)
        capacity = edge_data.get('capacity', 1000)
        
        # Apply BPR congestion function
        congested_time = self.bpr_congestion_function(free_flow_time, flow, capacity)
        # print(f"Calculated congested time: {congested_time} seconds (flow: {flow}, capacity: {capacity})")


        if not include_delays:
            return congested_time
        
        # Add traffic light delays
        delay = 0
        if self.enable_traffic_lights:
            # Simple model: major roads have traffic lights
            highway_type = edge_data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            
            if highway_type in ['primary', 'secondary', 'tertiary']:
                delay += random.uniform(10, 30)  # seconds
        
        # Add intersection delays
        if self.enable_intersection_delay:
            delay += random.uniform(2, 8)  # seconds
        
        return congested_time + delay
    
    def _get_random_delay(self, edge_data):
        """Calculate random delay for an edge (traffic lights + intersection)."""
        delay = 0
        if self.enable_traffic_lights:
            highway_type = edge_data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            if highway_type in ['primary', 'secondary', 'tertiary']:
                delay += random.uniform(5, 15)
        
        if self.enable_intersection_delay:
            delay += random.uniform(2, 8)
        
        return delay
    
    def find_shortest_path(self, origin, destination, weight='travel_time'):
        """
        Find shortest path between two nodes.
        
        Args:
            origin: Origin node ID
            destination: Destination node ID
            weight: Edge attribute to use as weight
        
        Returns:
            List of nodes representing the path
        """
        try:
            (_, path)= nx.bidirectional_dijkstra(self.G, origin, destination, weight=weight)
            return path
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found between {origin} and {destination}")
    
    def run_simulation(self):
        """
        Run both SELFISH and SOCIAL routing simulations in parallel.
        
        Returns:
            Tuple of (selfish_results, social_results) dictionaries
        """
        # Load network
        self.load_network()
        
        # Generate participants
        self.generate_participants()
        
        print("\n[3/4] Running SELFISH and SOCIAL routing simulations in parallel...")
        
        # Create deep copies of graph and participants for parallel execution
        G_selfish = copy.deepcopy(self.G)
        G_social = copy.deepcopy(self.G)
        participants_selfish = copy.deepcopy(self.participants)
        participants_social = copy.deepcopy(self.participants)
        
        # Run simulations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_selfish = executor.submit(
                self._run_selfish_routing_parallel,
                G_selfish,
                participants_selfish
            )
            future_social = executor.submit(
                self._run_social_routing_parallel,
                G_social,
                participants_social
            )
            
            # Collect results as they complete
            results = {}
            for future in as_completed([future_selfish, future_social]):
                result = future.result()
                if result['type'] == 'selfish':
                    results['selfish'] = result
                    print("      ✓ SELFISH simulation completed")
                else:
                    results['social'] = result
                    print("      ✓ SOCIAL simulation completed")
        
        selfish_results = results['selfish']
        social_results = results['social']
        
        # Compare results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        print(f"SELFISH routing: {selfish_results['avg_time']/60:.2f} min average")
        print(f"SOCIAL routing:  {social_results['avg_time']/60:.2f} min average")
        improvement = (selfish_results['avg_time'] - social_results['avg_time']) / selfish_results['avg_time'] * 100
        print(f"Improvement:     {improvement:.1f}%")
        print("="*60)
        
        return selfish_results, social_results
    
    def _run_selfish_routing_parallel(self, G, participants):
        """Run SELFISH routing strategy on provided graph and participants."""
        # Reset flows
        for u, v, k, d in G.edges(keys=True, data=True):
            d['flow'] = 0
        
        successful = 0
        failed = 0
        total_time = 0
        flow_data = {}
        
        # Pre-generate random delays if enabled (reduces RNG calls in hot loop)
        delay_cache = {}
        if self.enable_traffic_lights or self.enable_intersection_delay:
            for u, v, k in G.edges(keys=True):
                delay_cache[(u, v, k)] = self._get_random_delay(G[u][v][k])
        
        for participant in participants:
            try:
                # Find shortest path based on free-flow travel time
                route = nx.shortest_path(G, participant.origin, participant.destination, weight='travel_time')
                participant.route = route
                
                # Calculate travel time and update flows
                travel_time = 0
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    edge_data = G.get_edge_data(u, v)
                    
                    # Find the best edge if multiple edges exist
                    key = min(edge_data.keys(), key=lambda k, ed=edge_data: ed[k]['travel_time'])
                    data = G[u][v][key]
                    
                    # Update flow
                    data['flow'] += 1
                    flow_data[(u, v, key)] = data['flow']
                    
                    # Calculate time with current congestion
                    free_flow_time = data['travel_time']
                    flow = data['flow']
                    capacity = data['capacity']
                    
                    # BPR function inline for performance (amplified version)
                    if capacity > 0:
                        flow_ratio = flow / capacity
                        # Amplified: 2x alpha multiplier for stronger parameter impact
                        congestion_factor = 1 + (2.0 * self.bpr_alpha) * (flow_ratio ** self.bpr_beta)
                        congested_time = free_flow_time * congestion_factor
                    else:
                        congested_time = free_flow_time
                    
                    # Add pre-calculated delays
                    if self.enable_traffic_lights or self.enable_intersection_delay:
                        congested_time += delay_cache.get((u, v, key), 0)
                    
                    travel_time += congested_time
                
                participant.actual_travel_time = travel_time
                total_time += travel_time
                successful += 1
                
            except Exception:
                failed += 1
                participant.route = None
        
        avg_time = total_time / successful if successful > 0 else 0
        
        # Calculate congestion statistics (optimized)
        if flow_data:
            flows = list(flow_data.values())
            max_flow = max(flows)
            congested_edges = sum(f > 50 for f in flows)
        else:
            max_flow = 0
            congested_edges = 0
        
        return {
            'type': 'selfish',
            'avg_time': avg_time,
            'successful': successful,
            'failed': failed,
            'flow_data': flow_data,
            'max_flow': max_flow,
            'congested_edges': congested_edges,
            'stats': {
                'successful': successful,
                'failed': failed,
                'avg_time_min': avg_time/60,
                'max_flow': max_flow,
                'congested_edges': congested_edges,
                'bpr_alpha': self.bpr_alpha,
                'bpr_beta': self.bpr_beta
            }
        }
    
    def _run_social_routing_parallel(self, G, participants):
        """Run SOCIAL routing strategy on provided graph and participants."""
        # Reset flows and set initial costs
        for u, v, k, d in G.edges(keys=True, data=True):
            d['flow'] = 0
            d['current_cost'] = d['travel_time']
        
        successful = 0
        failed = 0
        total_time = 0
        flow_data = {}
        
        # Sort participants by arrival time (use itemgetter for speed)
        from operator import attrgetter
        sorted_participants = sorted(participants, key=attrgetter('desired_arrival'))
        
        for participant in sorted_participants:
            try:
                # Find shortest path based on current costs (includes congestion)
                route = nx.shortest_path(G, participant.origin, participant.destination, weight='current_cost')
                participant.route = route
                
                # Calculate travel time and update flows and costs
                travel_time = 0
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    edge_data = G.get_edge_data(u, v)
                    
                    # Find the best edge based on current cost
                    key = min(edge_data.keys(), key=lambda k, ed=edge_data: ed[k].get('current_cost', ed[k]['travel_time']))
                    data = G[u][v][key]
                    
                    # Use current cost for this segment
                    travel_time += data['current_cost']
                    
                    # Update flow
                    data['flow'] += 1
                    flow_data[(u, v, key)] = data['flow']
                    
                    # Recalculate cost for next participants
                    data['current_cost'] = self.calculate_edge_time(data, include_delays=True)
                
                participant.actual_travel_time = travel_time
                total_time += travel_time
                successful += 1
                
            except Exception:
                failed += 1
                participant.route = None
        
        avg_time = total_time / successful if successful > 0 else 0
        
        # Calculate congestion statistics (optimized)
        if flow_data:
            flows = list(flow_data.values())
            max_flow = max(flows)
            congested_edges = sum(f > 50 for f in flows)
        else:
            max_flow = 0
            congested_edges = 0
        
        return {
            'type': 'social',
            'avg_time': avg_time,
            'successful': successful,
            'failed': failed,
            'flow_data': flow_data,
            'max_flow': max_flow,
            'congested_edges': congested_edges,
            'stats': {
                'successful': successful,
                'failed': failed,
                'avg_time_min': avg_time/60,
                'max_flow': max_flow,
                'congested_edges': congested_edges,
                'bpr_alpha': self.bpr_alpha,
                'bpr_beta': self.bpr_beta
            }
        }