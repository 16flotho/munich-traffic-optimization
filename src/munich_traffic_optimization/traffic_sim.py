import osmnx as ox
import networkx as nx
import streamlit as st
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 1. GET DATA (Cached for speed)
@st.cache_data
def get_map():
    # Download Munich inner ring area (centered on Marienplatz)
    # The inner ring is roughly 3km in diameter
    return ox.graph_from_point((48.1374, 11.5755), dist=2000, network_type='drive')

@st.cache_data
def get_ring_crossing_nodes(_graph, num_pairs=15):
    """Get node pairs that are likely to cross the inner ring.
    
    Selects pairs where source is on one side and destination on opposite side,
    making it likely routes will go through the ring roads.
    """
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
        # Pick a random direction pair
        source_pool, dest_pool = random.choice(directions)
        
        if source_pool and dest_pool:
            source = random.choice(source_pool)
            dest = random.choice(dest_pool)
            
            if source != dest:
                try:
                    # Verify there's a path between nodes
                    nx.shortest_path(_graph, source, dest, weight='length')
                    pairs.append((source, dest))
                except nx.NetworkXNoPath:
                    continue
    
    return pairs

G = get_map()

# 2. DEFINE ROUTES
def calculate_routes(graph, strategy="selfish", node_pairs=None):
    """Calculate routes based on the chosen strategy."""
    routes = []
    edge_traffic = {}  # Track traffic on each edge
    
    if node_pairs is None:
        node_pairs = get_ring_crossing_nodes(graph, num_pairs=15)
    
    # Simulate 10 cars per origin-destination pair
    for source_node, dest_node in node_pairs:
        for _ in range(10):
            if strategy == "selfish":
                # Everyone takes the shortest path
                try:
                    route = nx.shortest_path(graph, source_node, dest_node, weight='length')
                    routes.append(route)
                    # Track traffic
                    for i in range(len(route) - 1):
                        edge = (route[i], route[i+1])
                        edge_traffic[edge] = edge_traffic.get(edge, 0) + 1
                except nx.NetworkXNoPath:
                    continue
                    
            elif strategy == "social_optimum":
                # Use traffic-aware routing
                # Create a copy of the graph with adjusted weights
                G_weighted = graph.copy()
                
                # Increase weight for congested edges
                for edge in G_weighted.edges():
                    u, v, k = edge if len(edge) == 3 else (*edge, 0)
                    base_length = G_weighted[u][v][k].get('length', 1)
                    traffic = edge_traffic.get((u, v), 0)
                    # Add congestion penalty: more traffic = higher weight
                    G_weighted[u][v][k]['adjusted_weight'] = base_length * (1 + traffic * 0.1)
                
                try:
                    route = nx.shortest_path(G_weighted, source_node, dest_node, weight='adjusted_weight')
                    routes.append(route)
                    # Track traffic
                    for i in range(len(route) - 1):
                        edge = (route[i], route[i+1])
                        edge_traffic[edge] = edge_traffic.get(edge, 0) + 1
                except nx.NetworkXNoPath:
                    continue
        
    return routes, edge_traffic

# 3. VISUALIZE
st.title("Munich Inner Ring Traffic Optimizer")
st.write("Simulate traffic crossing Munich's inner ring. Compare selfish routing (shortest path through the ring) vs. social optimum (traffic-aware routing that distributes load).")

strategy = st.radio("Choose Strategy", ["selfish", "social_optimum"])

if st.button("Run Simulation"):
    with st.spinner("Loading Munich inner ring map and calculating routes..."):
        # Get node pairs that cross the ring
        node_pairs = get_ring_crossing_nodes(G, num_pairs=15)
        
        results, edge_traffic = calculate_routes(G, strategy, node_pairs)
        
        if results:
            st.success(f"Calculated {len(results)} routes!")
            
            # Calculate metrics
            total_traffic = sum(edge_traffic.values())
            avg_edge_traffic = total_traffic / len(edge_traffic) if edge_traffic else 0
            max_congestion = max(edge_traffic.values()) if edge_traffic else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Total Routes", value=len(results))
            with col2:
                st.metric(label="Avg Edge Traffic", value=f"{avg_edge_traffic:.1f}")
            with col3:
                st.metric(label="Max Congestion", value=max_congestion)
            
            # Visualize routes with different colors
            try:
                # Create a color map for different routes
                colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
                route_colors = [mcolors.rgb2hex(c) for c in colors]
                
                fig, ax = ox.plot_graph_routes(
                    G, 
                    results, 
                    route_colors=route_colors, 
                    route_linewidth=1.5,
                    route_alpha=0.6,
                    node_size=0,
                    bgcolor='white',
                    show=False,
                    close=False
                )
                st.subheader("All Routes Visualization")
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Route visualization error: {e}")
            
            # Create congestion heatmap
            try:
                st.subheader("Traffic Congestion Heatmap")
                
                # Create edge colors based on traffic
                edge_colors = []
                edge_linewidths = []
                
                for u, v, k in G.edges(keys=True):
                    traffic = edge_traffic.get((u, v), 0)
                    
                    if traffic == 0:
                        # No traffic - light gray
                        edge_colors.append('#e0e0e0')
                        edge_linewidths.append(0.5)
                    elif traffic <= 5:
                        # Low traffic - light green
                        edge_colors.append('#90EE90')
                        edge_linewidths.append(1.0)
                    elif traffic <= 10:
                        # Medium traffic - yellow
                        edge_colors.append('#FFD700')
                        edge_linewidths.append(1.5)
                    elif traffic <= 20:
                        # High traffic - orange
                        edge_colors.append('#FF8C00')
                        edge_linewidths.append(2.0)
                    else:
                        # Very high traffic - red
                        edge_colors.append('#FF0000')
                        edge_linewidths.append(2.5)
                
                fig, ax = ox.plot_graph(
                    G,
                    node_size=0,
                    edge_color=edge_colors,
                    edge_linewidth=edge_linewidths,
                    bgcolor='white',
                    show=False,
                    close=False
                )
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#e0e0e0', label='No traffic'),
                    Patch(facecolor='#90EE90', label='Low (1-5 cars)'),
                    Patch(facecolor='#FFD700', label='Medium (6-10 cars)'),
                    Patch(facecolor='#FF8C00', label='High (11-20 cars)'),
                    Patch(facecolor='#FF0000', label='Very High (>20 cars)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Heatmap visualization error: {e}")
                st.write("Routes calculated but heatmap couldn't be visualized.")
        else:
            st.warning("No valid routes found. Try running the simulation again.")
