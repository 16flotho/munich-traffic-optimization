import osmnx as ox
import networkx as nx
import streamlit as st
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Constants for morning rush hour simulation
MORNING_RUSH_START = datetime.strptime("07:00", "%H:%M")
MORNING_RUSH_PEAK = datetime.strptime("08:00", "%H:%M")
MORNING_RUSH_END = datetime.strptime("09:00", "%H:%M")
BASE_SPEED_KMH = 30  # Average speed in city
CONGESTION_SLOWDOWN = 0.25  # Speed reduction per car on same edge (25% per car)
RANDOM_SEED = 42  # Seed for reproducibility

# 1. GET DATA (Cached for speed)
@st.cache_data
def get_map():
    # Download Munich inner ring area (centered on Marienplatz)
    return ox.graph_from_point((48.1374, 11.5755), dist=2000, network_type='drive')

@st.cache_data
def get_ring_crossing_nodes(_graph, num_pairs=15, seed=RANDOM_SEED):
    """Get node pairs that are likely to cross the inner ring."""
    random.seed(seed)
    nodes = list(_graph.nodes())
    center_lat, center_lon = 48.1374, 11.5755
    
    # Categorize nodes by their position relative to center
    north_nodes = []
    south_nodes = []
    east_nodes = []
    west_nodes = []
    
    for node in nodes:
        lat = _graph.nodes[node]['y']
        lon = _graph.nodes[node]['x']
        
        if lat > center_lat + 0.005:
            north_nodes.append(node)
        elif lat < center_lat - 0.005:
            south_nodes.append(node)
        
        if lon > center_lon + 0.005:
            east_nodes.append(node)
        elif lon < center_lon - 0.005:
            west_nodes.append(node)
    
    pairs = []
    directions = [
        (north_nodes, south_nodes),
        (south_nodes, north_nodes),
        (east_nodes, west_nodes),
        (west_nodes, east_nodes),
    ]
    
    # Generate pairs that cross through the center
    attempts = 0
    max_attempts = num_pairs * 10
    
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        source_pool, dest_pool = random.choice(directions)
        
        if source_pool and dest_pool:
            source = random.choice(source_pool)
            dest = random.choice(dest_pool)
            
            if source != dest:
                try:
                    nx.shortest_path(_graph, source, dest, weight='length')
                    pairs.append((source, dest))
                except nx.NetworkXNoPath:
                    continue
    
    return pairs

def generate_start_times(num_cars, rush_start, rush_peak, rush_end, seed=RANDOM_SEED):
    """Generate start times following a normal distribution around rush hour peak."""
    rng = np.random.default_rng(seed)
    start_times = []
    
    # Convert to minutes for easier calculation
    start_min = 0
    peak_min = (rush_peak - rush_start).total_seconds() / 60
    end_min = (rush_end - rush_start).total_seconds() / 60
    
    # Generate times with normal distribution around peak
    for _ in range(num_cars):
        # Normal distribution centered at peak with std dev of 20 minutes
        time_offset = rng.normal(peak_min, 20)
        time_offset = max(start_min, min(end_min, time_offset))  # Clamp to range
        
        start_time = rush_start + timedelta(minutes=time_offset)
        start_times.append(start_time)
    
    return sorted(start_times)

class TripRequest:
    """Represents a user's trip request in the database."""
    def __init__(self, user_id, source, dest, desired_start_time):
        self.user_id = user_id
        self.source = source
        self.dest = dest
        self.desired_start_time = desired_start_time
        self.assigned_route = None
        self.actual_start_time = None
        self.arrival_time = None
        self.travel_time_minutes = None

def calculate_edge_time(graph, edge, num_cars_on_edge):
    """Calculate time to traverse an edge considering congestion."""
    u, v = edge
    
    # Get edge length (in meters)
    edge_data = graph.get_edge_data(u, v)
    if edge_data:
        length_m = list(edge_data.values())[0].get('length', 100)
    else:
        length_m = 100
    
    # Calculate base time (no congestion)
    speed_ms = (BASE_SPEED_KMH * 1000) / 3600  # Convert km/h to m/s
    
    # Apply congestion penalty
    congestion_factor = max(0.2, 1.0 - (num_cars_on_edge * CONGESTION_SLOWDOWN))
    actual_speed_ms = speed_ms * congestion_factor
    
    time_seconds = length_m / actual_speed_ms
    return time_seconds / 60  # Return minutes

def calculate_route_time(graph, route, edge_loads):
    """Calculate total travel time for a route given edge loads."""
    total_time = 0
    for i in range(len(route) - 1):
        edge = (route[i], route[i+1])
        num_cars = edge_loads.get(edge, 0)
        total_time += calculate_edge_time(graph, edge, num_cars)
    return total_time

def selfish_routing(graph, trip_requests):
    """Selfish routing: Everyone takes shortest path, no coordination."""
    edge_loads = defaultdict(int)
    
    for request in trip_requests:
        try:
            # Each user independently finds shortest path
            route = nx.shortest_path(graph, request.source, request.dest, weight='length')
            request.assigned_route = route
            
            # Update edge loads
            for i in range(len(route) - 1):
                edge = (route[i], route[i+1])
                edge_loads[edge] += 1
        except nx.NetworkXNoPath:
            request.assigned_route = None
    
    return edge_loads

def social_optimum_routing(graph, trip_requests, max_iterations=15):
    """Social optimum: Centralized routing that minimizes total system travel time.
    
    Uses very strong congestion penalties to force route diversity.
    """
    edge_loads = defaultdict(int)
    
    # Initialize with shortest paths
    for request in trip_requests:
        try:
            route = nx.shortest_path(graph, request.source, request.dest, weight='length')
            request.assigned_route = route
            for i in range(len(route) - 1):
                edge = (route[i], route[i+1])
                edge_loads[edge] += 1
        except nx.NetworkXNoPath:
            request.assigned_route = None
    
    # Iteratively improve
    for iteration in range(max_iterations):
        improved = False
        
        # Calculate current total system time
        current_system_time = sum(
            calculate_route_time(graph, req.assigned_route, edge_loads)
            for req in trip_requests if req.assigned_route
        )
        
        # Shuffle to avoid bias
        import random
        requests_to_optimize = [req for req in trip_requests if req.assigned_route]
        random.shuffle(requests_to_optimize)
        
        # Try to reroute each user
        for request in requests_to_optimize:
            current_route = request.assigned_route
            
            # Create weighted graph with VERY STRONG congestion penalties
            g_weighted = graph.copy()
            for u, v, key, data in graph.edges(keys=True, data=True):
                edge = (u, v)
                load = edge_loads.get(edge, 0)
                base_length = data.get('length', 100)
                
                # MUCH stronger penalty - exponential growth with load
                # This forces the algorithm to spread traffic
                if load > 0:
                    # Exponential penalty: each additional car makes it exponentially worse
                    congestion_factor = 1.0 + (load ** 2.0) * 0.5
                else:
                    congestion_factor = 1.0
                    
                g_weighted[u][v][key]['congestion_weight'] = base_length * congestion_factor
            
            # Find alternative route
            try:
                alternative_route = nx.shortest_path(
                    g_weighted,
                    request.source,
                    request.dest,
                    weight='congestion_weight'
                )
                
                # Skip if same as current
                if alternative_route == current_route:
                    continue
                
                # Calculate impact of switching
                test_loads = edge_loads.copy()
                
                # Remove current route
                for i in range(len(current_route) - 1):
                    edge = (current_route[i], current_route[i+1])
                    test_loads[edge] = max(0, test_loads[edge] - 1)
                
                # Add alternative route
                for i in range(len(alternative_route) - 1):
                    edge = (alternative_route[i], alternative_route[i+1])
                    test_loads[edge] += 1
                
                # Calculate new total system time with new loads
                new_system_time = 0
                for req in trip_requests:
                    if not req.assigned_route:
                        continue
                    # Use correct route for each request
                    route_to_use = alternative_route if req.user_id == request.user_id else req.assigned_route
                    new_system_time += calculate_route_time(graph, route_to_use, test_loads)
                
                # Accept if it reduces total system time
                # Be more aggressive in early iterations
                threshold = 1.0 if iteration < 5 else 0.995
                if new_system_time < current_system_time * threshold:
                    request.assigned_route = alternative_route
                    edge_loads = test_loads
                    current_system_time = new_system_time
                    improved = True
                    
            except nx.NetworkXNoPath:
                continue
        
        # If no improvements, we've converged
        if not improved:
            break
    
    return edge_loads

def simulate_traffic_with_database(graph, strategy, trip_requests):
    """Simulate traffic execution with pre-assigned routes from database."""
    
    # Routes are already assigned by the routing algorithm
    current_time = MORNING_RUSH_START
    simulation_end = MORNING_RUSH_END + timedelta(minutes=90)
    
    edge_occupancy = defaultdict(list)
    edge_traffic_history = defaultdict(int)
    edge_time_remaining = {}
    
    # Track active requests by start time
    pending_requests = sorted(trip_requests, key=lambda r: r.desired_start_time)
    active_requests = []
    completed_requests = []
    
    request_idx = 0
    
    while current_time < simulation_end:
        # Start new trips that are scheduled for this time
        while request_idx < len(pending_requests):
            request = pending_requests[request_idx]
            if request.desired_start_time <= current_time:
                if request.assigned_route:
                    request.actual_start_time = current_time
                    request.current_edge_index = 0
                    active_requests.append(request)
                request_idx += 1
            else:
                break
        
        # Process active trips
        for request in list(active_requests):
            if not request.assigned_route:
                continue
            
            # Check if trip is complete
            if request.current_edge_index >= len(request.assigned_route) - 1:
                if request.arrival_time is None:
                    request.arrival_time = current_time
                    request.travel_time_minutes = (request.arrival_time - request.actual_start_time).total_seconds() / 60
                    completed_requests.append(request)
                    active_requests.remove(request)
                continue
            
            # Get current edge
            current_edge = (
                request.assigned_route[request.current_edge_index],
                request.assigned_route[request.current_edge_index + 1]
            )
            
            # Car just arrived at this edge
            if request.user_id not in edge_occupancy[current_edge]:
                num_cars_on_edge = len(edge_occupancy[current_edge])
                edge_time = calculate_edge_time(graph, current_edge, num_cars_on_edge)
                
                # Add car to edge
                edge_occupancy[current_edge].append(request.user_id)
                edge_time_remaining[(request.user_id, current_edge)] = edge_time
                edge_traffic_history[current_edge] += 1
            
            # Decrease remaining time
            if (request.user_id, current_edge) in edge_time_remaining:
                edge_time_remaining[(request.user_id, current_edge)] -= 1
                
                # Car finished this edge
                if edge_time_remaining[(request.user_id, current_edge)] <= 0:
                    edge_occupancy[current_edge].remove(request.user_id)
                    del edge_time_remaining[(request.user_id, current_edge)]
                    request.current_edge_index += 1
        
        current_time += timedelta(minutes=1)
    
    # Ensure all have arrival times
    for request in trip_requests:
        if request.assigned_route and request.arrival_time is None:
            request.arrival_time = simulation_end
            if request.actual_start_time:
                request.travel_time_minutes = (request.arrival_time - request.actual_start_time).total_seconds() / 60
    
    return edge_traffic_history

G = get_map()

# 3. VISUALIZE
st.title("🚗 Munich Traffic Optimizer - Centralized Routing Database")
st.write("""
**Scenario**: A centralized database knows all users' trip requests in advance (origin, destination, desired departure time).

**Strategies**:
- **Selfish**: Each user independently takes their shortest path → causes congestion
- **Social Optimum**: Central algorithm assigns routes to minimize total travel time → reroutes some users to balance load
""")

col1, col2 = st.columns(2)
with col1:
    strategy = st.radio("Choose Strategy", ["selfish", "social_optimum"])
with col2:
    st.info("""
    **Key Difference**: 
    - Selfish = No coordination
    - Social Optimum = Pre-planned routes that distribute traffic efficiently
    """)

# Add settings in sidebar
with st.sidebar:
    st.subheader("Simulation Settings")
    seed = st.number_input("Random Seed", min_value=0, value=RANDOM_SEED, step=1)
    num_od_pairs = st.slider("Number of Origin-Destination Pairs", 5, 30, 15)
    cars_per_pair = st.slider("Cars per OD Pair", 5, 20, 10)
    st.info("Same seed = same simulation results")

if st.button("Run Database Simulation"):
    with st.spinner("Loading trip database and computing optimal routes..."):
        # Generate node pairs (OD pairs in database)
        node_pairs = get_ring_crossing_nodes(G, num_pairs=num_od_pairs, seed=seed)
        
        # Generate trip requests (database entries)
        total_cars = len(node_pairs) * cars_per_pair
        start_times = generate_start_times(total_cars, MORNING_RUSH_START, MORNING_RUSH_PEAK, MORNING_RUSH_END, seed)
        
        trip_requests = []
        time_idx = 0
        for source, dest in node_pairs:
            for _ in range(cars_per_pair):
                request = TripRequest(time_idx, source, dest, start_times[time_idx])
                trip_requests.append(request)
                time_idx += 1
        
        # Apply routing strategy
        st.info(f"🗄️ Database contains {len(trip_requests)} trip requests")
        
        if strategy == "selfish":
            st.write("⚡ Computing selfish routes (each user finds shortest path)...")
            edge_loads = selfish_routing(G, trip_requests)
        else:
            st.write("🧠 Computing social optimum routes (load balancing)...")
            edge_loads = social_optimum_routing(G, trip_requests, max_iterations=15)
        
        # Simulate traffic execution
        st.write("🚦 Simulating traffic flow with assigned routes...")
        edge_traffic = simulate_traffic_with_database(G, strategy, trip_requests)
        
        # Calculate metrics
        valid_requests = [r for r in trip_requests if r.travel_time_minutes is not None]
        travel_times = [r.travel_time_minutes for r in valid_requests]
        
        if travel_times:
            avg_travel_time = np.mean(travel_times)
            median_travel_time = np.median(travel_times)
            max_travel_time = np.max(travel_times)
            min_travel_time = np.min(travel_times)
            std_travel_time = np.std(travel_times)
            total_system_time = sum(travel_times)
            
            st.success(f"✅ Simulation complete! Processed {len(valid_requests)} trips")
            
            # Main metrics
            st.subheader("📊 System Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Avg Travel Time", 
                    value=f"{avg_travel_time:.1f} min",
                    help="Lower is better for commuters"
                )
            with col2:
                st.metric(
                    label="Total System Time", 
                    value=f"{total_system_time:.0f} min",
                    help="Sum of all travel times - key optimization target"
                )
            with col3:
                st.metric(
                    label="Std Deviation", 
                    value=f"{std_travel_time:.1f} min",
                    help="Lower = more predictable commutes"
                )
            with col4:
                max_congestion = max(edge_traffic.values()) if edge_traffic else 0
                st.metric(
                    label="Peak Congestion", 
                    value=f"{max_congestion} cars",
                    help="Max cars on any single street"
                )
            
            # Show route diversity
            unique_routes = len(set(tuple(r.assigned_route) for r in valid_requests if r.assigned_route))
            st.metric(
                label="Route Diversity", 
                value=f"{unique_routes} unique routes",
                help="More diversity = better load distribution"
            )
            
            # Travel time distribution
            st.subheader("⏱️ Travel Time Distribution")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
            ax_hist.hist(travel_times, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax_hist.axvline(avg_travel_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_travel_time:.1f} min')
            ax_hist.axvline(median_travel_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_travel_time:.1f} min')
            ax_hist.set_xlabel('Travel Time (minutes)')
            ax_hist.set_ylabel('Number of Users')
            ax_hist.set_title('How long did trips take?')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
            st.pyplot(fig_hist)
            plt.close(fig_hist)
            
            # Route visualization
            st.subheader("🗺️ Assigned Routes")
            all_routes = [r.assigned_route for r in valid_requests if r.assigned_route]
            
            try:
                colors = plt.cm.rainbow(np.linspace(0, 1, len(all_routes)))
                route_colors = [mcolors.rgb2hex(c) for c in colors]
                
                fig_routes, ax_routes = ox.plot_graph_routes(
                    G, 
                    all_routes, 
                    route_colors=route_colors, 
                    route_linewidth=0.8,
                    route_alpha=0.3,
                    node_size=0,
                    bgcolor='white',
                    show=False,
                    close=False
                )
                st.pyplot(fig_routes)
                plt.close(fig_routes)
            except Exception as e:
                st.error(f"Visualization error: {e}")
            
            # Congestion heatmap
            st.subheader("🔥 Traffic Congestion Heatmap")
            try:
                edge_colors = []
                edge_linewidths = []
                
                max_traffic = max(edge_traffic.values()) if edge_traffic else 1
                
                for u, v, k in G.edges(keys=True):
                    traffic = edge_traffic.get((u, v), 0)
                    
                    if traffic == 0:
                        edge_colors.append('#e0e0e0')
                        edge_linewidths.append(0.5)
                    else:
                        intensity = traffic / max_traffic
                        
                        if intensity <= 0.2:
                            edge_colors.append('#90EE90')
                            edge_linewidths.append(1.0)
                        elif intensity <= 0.4:
                            edge_colors.append('#FFD700')
                            edge_linewidths.append(1.5)
                        elif intensity <= 0.6:
                            edge_colors.append('#FFA500')
                            edge_linewidths.append(2.0)
                        elif intensity <= 0.8:
                            edge_colors.append('#FF8C00')
                            edge_linewidths.append(2.5)
                        else:
                            edge_colors.append('#FF0000')
                            edge_linewidths.append(3.0)
                
                fig_heat, ax_heat = ox.plot_graph(
                    G,
                    node_size=0,
                    edge_color=edge_colors,
                    edge_linewidth=edge_linewidths,
                    bgcolor='white',
                    show=False,
                    close=False
                )
                
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#e0e0e0', label='No traffic'),
                    Patch(facecolor='#90EE90', label='Light (0-20%)'),
                    Patch(facecolor='#FFD700', label='Moderate (20-40%)'),
                    Patch(facecolor='#FFA500', label='Heavy (40-60%)'),
                    Patch(facecolor='#FF8C00', label='Very Heavy (60-80%)'),
                    Patch(facecolor='#FF0000', label=f'Extreme (>80%, max={max_traffic})')
                ]
                ax_heat.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
                st.pyplot(fig_heat)
                plt.close(fig_heat)
            except Exception as e:
                st.error(f"Heatmap error: {e}")
            
            # Key insights
            st.subheader("💡 Analysis")
            efficiency = (min_travel_time / avg_travel_time) * 100 if avg_travel_time > 0 else 0
            
            if strategy == "selfish":
                st.write(f"""
                **Selfish Routing Results**:
                - Everyone took their shortest path independently
                - No coordination → likely congestion on popular routes
                - Total system time: **{total_system_time:.0f} minutes**
                - Average trip: **{avg_travel_time:.1f} minutes**
                - Efficiency: {efficiency:.1f}% (ratio of fastest to average)
                
                💭 *Try "Social Optimum" to see if centralized routing can improve these metrics!*
                """)
            else:
                st.write(f"""
                **Social Optimum Results**:
                - Routes pre-assigned by central algorithm to minimize total travel time
                - Some users rerouted (up to 15% longer personal path) to reduce overall congestion
                - Total system time: **{total_system_time:.0f} minutes**
                - Average trip: **{avg_travel_time:.1f} minutes**
                - Route diversity: {unique_routes} different paths used
                
                ✅ *Compare with "Selfish" to measure the benefit of coordination!*
                """)
        else:
            st.warning("No valid trips completed.")
