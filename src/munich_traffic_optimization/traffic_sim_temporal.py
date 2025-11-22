import osmnx as ox
import networkx as nx
import streamlit as st
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime, timedelta

# Constants for morning rush hour simulation
MORNING_RUSH_START = datetime.strptime("07:00", "%H:%M")
MORNING_RUSH_PEAK = datetime.strptime("08:00", "%H:%M")
MORNING_RUSH_END = datetime.strptime("09:00", "%H:%M")
BASE_SPEED_KMH = 30  # Average speed in city
CONGESTION_SLOWDOWN = 0.7  # Speed reduction per car on same edge
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

class Car:
    """Represents a car with route and timing information."""
    def __init__(self, car_id, source, dest, start_time, route):
        self.car_id = car_id
        self.source = source
        self.dest = dest
        self.start_time = start_time
        self.route = route
        self.current_edge_index = 0
        self.arrival_time = None
        self.travel_time_minutes = None
        self.edges_traveled = []
        
    def is_finished(self):
        return self.current_edge_index >= len(self.route) - 1
    
    def get_current_edge(self):
        if self.is_finished():
            return None
        return (self.route[self.current_edge_index], self.route[self.current_edge_index + 1])

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

def simulate_temporal_traffic(graph, strategy, node_pairs, seed=RANDOM_SEED):
    """Simulate traffic with temporal dynamics."""
    random.seed(seed)
    np.random.seed(seed)
    cars = []
    car_id = 0
    
    # Generate cars for each origin-destination pair
    num_cars_per_pair = 10
    total_cars = len(node_pairs) * num_cars_per_pair
    
    start_times = generate_start_times(total_cars, MORNING_RUSH_START, MORNING_RUSH_PEAK, MORNING_RUSH_END, seed)
    
    time_idx = 0
    for source_node, dest_node in node_pairs:
        for _ in range(num_cars_per_pair):
            try:
                # Pre-calculate routes for selfish strategy
                if strategy == "selfish":
                    route = nx.shortest_path(graph, source_node, dest_node, weight='length')
                else:
                    # For social optimum, route is calculated when car starts
                    route = None
                
                car = Car(car_id, source_node, dest_node, start_times[time_idx], route)
                cars.append(car)
                car_id += 1
                time_idx += 1
            except nx.NetworkXNoPath:
                continue
    
    # Simulation
    current_time = MORNING_RUSH_START
    simulation_end = MORNING_RUSH_END + timedelta(minutes=60)  # Extra time for stragglers
    edge_occupancy = {}  # Tracks which cars are on which edges at current time
    edge_traffic_history = {}  # Total traffic that has passed through each edge
    edge_time_remaining = {}  # Track how much time each car needs on current edge
    
    while current_time < simulation_end:
        # Process each car
        for car in cars:
            if car.arrival_time is not None:
                continue  # Car has finished
            
            if car.start_time > current_time:
                continue  # Car hasn't started yet
            
            # For social optimum, calculate route when car starts (uses current traffic state)
            if strategy == "social_optimum" and car.route is None:
                g_weighted = graph.copy()
                
                # Consider both current occupancy AND historical traffic
                all_edges_with_traffic = set(edge_occupancy.keys()) | set(edge_traffic_history.keys())
                
                for edge_key in all_edges_with_traffic:
                    u, v = edge_key
                    if u in g_weighted and v in g_weighted[u]:
                        for k in g_weighted[u][v]:
                            base_length = g_weighted[u][v][k].get('length', 1)
                            
                            # Current congestion (real-time)
                            current_traffic = len(edge_occupancy.get(edge_key, []))
                            
                            # Historical congestion (predictive)
                            historical_traffic = edge_traffic_history.get(edge_key, 0)
                            
                            # Weight combines both factors
                            # Current traffic has higher weight as it's more immediate
                            base_weight = base_length
                            current_congestion = base_length * current_traffic * 1.0
                            historical_congestion = base_length * (historical_traffic / 50.0) * 0.5
                            
                            g_weighted[u][v][k]['adjusted_weight'] = base_weight + current_congestion + historical_congestion
                
                try:
                    car.route = nx.shortest_path(g_weighted, car.source, car.dest, weight='adjusted_weight')
                    car.current_edge_index = 0
                except nx.NetworkXNoPath:
                    car.arrival_time = current_time
                    continue
            
            if car.is_finished():
                if car.arrival_time is None:
                    car.arrival_time = current_time
                    car.travel_time_minutes = (car.arrival_time - car.start_time).total_seconds() / 60
                continue
            
            current_edge = car.get_current_edge()
            if current_edge is None:
                continue
            
            # Initialize edge occupancy tracking
            if current_edge not in edge_occupancy:
                edge_occupancy[current_edge] = []
            
            # Car just arrived at this edge
            if car.car_id not in edge_occupancy[current_edge]:
                num_cars_on_edge = len(edge_occupancy[current_edge])
                edge_time = calculate_edge_time(graph, current_edge, num_cars_on_edge)
                
                # Add car to this edge
                edge_occupancy[current_edge].append(car.car_id)
                car.edges_traveled.append(current_edge)
                edge_time_remaining[(car.car_id, current_edge)] = edge_time
                
                # Track in history
                if current_edge not in edge_traffic_history:
                    edge_traffic_history[current_edge] = 0
                edge_traffic_history[current_edge] += 1
            
            # Decrease remaining time on this edge
            if (car.car_id, current_edge) in edge_time_remaining:
                edge_time_remaining[(car.car_id, current_edge)] -= 1  # 1 minute time step
                
                # Car has finished this edge
                if edge_time_remaining[(car.car_id, current_edge)] <= 0:
                    # Remove car from this edge
                    if car.car_id in edge_occupancy[current_edge]:
                        edge_occupancy[current_edge].remove(car.car_id)
                    
                    # Clean up time tracking
                    del edge_time_remaining[(car.car_id, current_edge)]
                    
                    # Move to next edge
                    car.current_edge_index += 1
        
        # Advance time by 1 minute
        current_time += timedelta(minutes=1)
    
    # Ensure all cars have arrival times
    for car in cars:
        if car.arrival_time is None:
            car.arrival_time = simulation_end
            car.travel_time_minutes = (car.arrival_time - car.start_time).total_seconds() / 60
    
    return cars, edge_traffic_history

G = get_map()

# 3. VISUALIZE
st.title("Munich Inner Ring Traffic Optimizer - Temporal Simulation")
st.write("""
Simulate morning rush hour traffic (7:00-9:00 AM) crossing Munich's inner ring. 
Cars depart at different times with peak traffic around 8:00 AM.
Travel times account for real-time congestion.
""")

strategy = st.radio("Choose Strategy", ["selfish", "social_optimum"])

# Add seed input in sidebar
with st.sidebar:
    st.subheader("Simulation Settings")
    seed = st.number_input("Random Seed (for reproducibility)", min_value=0, value=RANDOM_SEED, step=1)
    st.info("Same seed = same simulation results")

if st.button("Run Temporal Simulation"):
    with st.spinner("Simulating morning rush hour traffic..."):
        node_pairs = get_ring_crossing_nodes(G, num_pairs=15, seed=seed)
        
        cars, edge_traffic = simulate_temporal_traffic(G, strategy, node_pairs, seed=seed)
        
        if cars:
            # Calculate comprehensive metrics
            finished_cars = [car for car in cars if car.arrival_time is not None]
            travel_times = [car.travel_time_minutes for car in finished_cars if car.travel_time_minutes is not None]
            
            if travel_times:
                avg_travel_time = np.mean(travel_times)
                median_travel_time = np.median(travel_times)
                max_travel_time = np.max(travel_times)
                min_travel_time = np.min(travel_times)
                std_travel_time = np.std(travel_times)
                
                st.success(f"Simulated {len(cars)} cars through morning rush hour!")
                
                # Main metrics
                st.subheader("📊 Travel Time Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Average Travel Time", value=f"{avg_travel_time:.1f} min")
                with col2:
                    st.metric(label="Median Travel Time", value=f"{median_travel_time:.1f} min")
                with col3:
                    st.metric(label="Fastest Trip", value=f"{min_travel_time:.1f} min")
                with col4:
                    st.metric(label="Slowest Trip", value=f"{max_travel_time:.1f} min")
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric(label="Std Deviation", value=f"{std_travel_time:.1f} min")
                with col6:
                    st.metric(label="Total Cars", value=len(cars))
                with col7:
                    total_system_time = sum(travel_times)
                    st.metric(label="Total System Time", value=f"{total_system_time:.0f} min")
                
                # Travel time distribution
                st.subheader("⏱️ Travel Time Distribution")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                ax_hist.hist(travel_times, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                ax_hist.axvline(avg_travel_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_travel_time:.1f} min')
                ax_hist.axvline(median_travel_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_travel_time:.1f} min')
                ax_hist.set_xlabel('Travel Time (minutes)')
                ax_hist.set_ylabel('Number of Cars')
                ax_hist.set_title('Distribution of Travel Times')
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                st.pyplot(fig_hist)
                plt.close(fig_hist)
                
                # Start times distribution
                st.subheader("🚗 Departure Times Distribution")
                start_times_minutes = [(car.start_time - MORNING_RUSH_START).total_seconds() / 60 for car in cars]
                fig_start, ax_start = plt.subplots(figsize=(10, 4))
                ax_start.hist(start_times_minutes, bins=30, color='orange', edgecolor='black', alpha=0.7)
                ax_start.set_xlabel('Minutes from 7:00 AM')
                ax_start.set_ylabel('Number of Cars Departing')
                ax_start.set_title('Morning Rush Hour Departure Pattern')
                ax_start.grid(True, alpha=0.3)
                st.pyplot(fig_start)
                plt.close(fig_start)
                
                # Collect all routes for visualization
                all_routes = [car.route for car in cars if car.route]
                
                # Visualize routes with different colors
                st.subheader("🗺️ All Routes Visualization")
                try:
                    colors = plt.cm.rainbow(np.linspace(0, 1, len(all_routes)))
                    route_colors = [mcolors.rgb2hex(c) for c in colors]
                    
                    fig_routes, ax_routes = ox.plot_graph_routes(
                        G, 
                        all_routes, 
                        route_colors=route_colors, 
                        route_linewidth=1.0,
                        route_alpha=0.4,
                        node_size=0,
                        bgcolor='white',
                        show=False,
                        close=False
                    )
                    st.pyplot(fig_routes)
                    plt.close(fig_routes)
                except Exception as e:
                    st.error(f"Route visualization error: {e}")
                
                # Congestion heatmap
                st.subheader("🔥 Cumulative Traffic Congestion Heatmap")
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
                            # Normalize traffic to 0-1 scale
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
                        Patch(facecolor='#FF0000', label=f'Extreme (80-100%, max={max_traffic})')
                    ]
                    ax_heat.legend(handles=legend_elements, loc='upper right', fontsize=8)
                    
                    st.pyplot(fig_heat)
                    plt.close(fig_heat)
                except Exception as e:
                    st.error(f"Heatmap visualization error: {e}")
                
                # Key insights
                st.subheader("💡 Key Insights")
                efficiency = (min_travel_time / avg_travel_time) * 100 if avg_travel_time > 0 else 0
                
                st.write(f"""
                - **System Efficiency**: {efficiency:.1f}% (ratio of fastest to average trip)
                - **Time Variance**: ±{std_travel_time:.1f} minutes (lower is better)
                - **Peak Load**: {max_traffic} cars on busiest street segment
                - **Strategy**: {"Selfish routing - everyone takes shortest path" if strategy == "selfish" else "Social optimum - routes adapt to congestion"}
                """)
                
            else:
                st.warning("No completed trips to analyze.")
        else:
            st.warning("No valid routes found. Try running the simulation again.")
