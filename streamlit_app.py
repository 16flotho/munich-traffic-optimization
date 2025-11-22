"""
Streamlit Web App for Munich Traffic Simulation
================================================

Interactive comparison of SELFISH vs SOCIAL routing strategies.
"""

import streamlit as st
import osmnx as ox
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# Import the simulation class
from munich_taufkirchen_simulation import (
    MunichTaufkirchenSimulation, 
    TrafficParticipant,
    MUNICH_CENTER,
    DIST_FROM_CENTER
)

# Page configuration
st.set_page_config(
    page_title="Munich Traffic Optimization",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_network():
    """Load and cache the road network."""
    with st.spinner("Loading Munich road network... This may take a minute..."):
        G = ox.graph_from_point(
            MUNICH_CENTER, 
            dist=DIST_FROM_CENTER, 
            network_type='drive',
            simplify=True
        )
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    return G


def create_visualization(G, flow_data, scenario, max_flow):
    """Create traffic flow visualization."""
    edge_colors = []
    edge_widths = []
    
    for u, v, k in G.edges(keys=True):
        flow = flow_data.get((u, v, k), 0)
        
        if flow == 0:
            edge_colors.append('#e0e0e0')
            edge_widths.append(0.3)
        else:
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
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        bgcolor='white',
        show=False,
        close=False,
        figsize=(10, 10)
    )
    
    ax.set_title(f"{scenario.upper()} Routing Strategy", 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Low traffic (< 20%)'),
        Patch(facecolor='#FFD700', label='Moderate (20-50%)'),
        Patch(facecolor='#FF8C00', label='Heavy (50-80%)'),
        Patch(facecolor='#FF0000', label='Congested (> 80%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf


def run_simulation(num_participants, bpr_alpha, bpr_beta, enable_traffic_lights, enable_intersection_delay, seed):
    """Run the simulation with custom parameters."""
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create custom simulation class with parameters
    class CustomSimulation(MunichTaufkirchenSimulation):
        def __init__(self, num_participants, G, bpr_alpha, bpr_beta, enable_lights, enable_intersection):
            self.num_participants = num_participants
            self.G = G
            self.participants = []
            self.bpr_alpha = bpr_alpha
            self.bpr_beta = bpr_beta
            self.enable_traffic_lights = enable_lights
            self.enable_intersection_delay = enable_intersection
            
        def bpr_congestion_function(self, free_flow_time, flow, capacity):
            if capacity <= 0:
                return free_flow_time
            congestion_ratio = flow / capacity
            congestion_factor = 1 + self.bpr_alpha * (congestion_ratio ** self.bpr_beta)
            return free_flow_time * congestion_factor
        
        def calculate_edge_time(self, edge_data, include_delays=True):
            base_time = self.bpr_congestion_function(
                edge_data['free_flow_time'],
                edge_data['flow'],
                edge_data['capacity']
            )
            
            if not include_delays:
                return base_time
            
            traffic_light_delay = 0
            if self.enable_traffic_lights and edge_data.get('has_traffic_light', True):
                flow_ratio = edge_data['flow'] / edge_data['capacity']
                base_wait = 20
                traffic_light_delay = base_wait * (1 + flow_ratio * 0.5)
            
            intersection_delay = 0
            if self.enable_intersection_delay:
                intersection_delay = edge_data.get('intersection_penalty', 0)
                flow_ratio = edge_data['flow'] / edge_data['capacity']
                intersection_delay *= (1 + flow_ratio * 0.3)
            
            return base_time + traffic_light_delay + intersection_delay
    
    # Load network
    G = load_network()
    
    # Prepare graph
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'travel_time' not in data:
            length = data.get('length', 100)
            speed = data.get('speed_kph', 30)
            if isinstance(speed, list): 
                speed = float(speed[0])
            data['travel_time'] = (length / 1000) / speed * 3600
        
        highway = data.get('highway', 'residential')
        if isinstance(highway, list): 
            highway = highway[0]
        
        if highway in ['motorway', 'motorway_link']:
            capacity = 1800
        elif highway in ['trunk', 'trunk_link']:
            capacity = 1400
        elif highway in ['primary', 'primary_link']:
            capacity = 1000
        elif highway in ['secondary', 'secondary_link']:
            capacity = 600
        elif highway in ['tertiary', 'tertiary_link']:
            capacity = 400
        else:
            capacity = 250
        
        data['capacity'] = capacity
        data['flow'] = 0
        data['free_flow_time'] = data['travel_time']
        data['has_traffic_light'] = highway not in ['motorway', 'motorway_link', 'trunk', 'trunk_link']
        
        if highway in ['primary', 'primary_link']:
            data['intersection_penalty'] = 15
        elif highway in ['secondary', 'secondary_link']:
            data['intersection_penalty'] = 20
        elif highway in ['tertiary', 'tertiary_link']:
            data['intersection_penalty'] = 25
        else:
            data['intersection_penalty'] = 30
    
    # Create simulation
    sim = CustomSimulation(num_participants, G, bpr_alpha, bpr_beta, enable_traffic_lights, enable_intersection_delay)
    sim.generate_participants()
    
    # Run both scenarios
    progress_bar = st.progress(0, text="Running SELFISH routing simulation...")
    selfish_results = sim.run_selfish_routing()
    
    progress_bar.progress(50, text="Running SOCIAL routing simulation...")
    social_results = sim.run_social_routing()
    
    progress_bar.progress(100, text="Simulation complete!")
    progress_bar.empty()
    
    return sim, selfish_results, social_results


def main():
    # Header
    st.markdown('<div class="main-header">🚗 Munich Traffic Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing SELFISH vs SOCIAL Routing Strategies</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Simulation Parameters")
    
    num_participants = st.sidebar.slider(
        "Number of Participants",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Total number of traffic participants in the simulation"
    )
    
    st.sidebar.subheader("Congestion Model")
    bpr_alpha = st.sidebar.slider(
        "BPR Alpha (Congestion Sensitivity)",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Controls how quickly congestion builds up"
    )
    
    bpr_beta = st.sidebar.slider(
        "BPR Beta (Congestion Power)",
        min_value=2.0,
        max_value=6.0,
        value=4.0,
        step=0.5,
        help="Power factor for congestion curve"
    )
    
    st.sidebar.subheader("Additional Delays")
    enable_traffic_lights = st.sidebar.checkbox(
        "Enable Traffic Lights",
        value=True,
        help="Add traffic light delays"
    )
    
    enable_intersection_delay = st.sidebar.checkbox(
        "Enable Intersection Delays",
        value=True,
        help="Model intersection complexity"
    )
    
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=1,
        max_value=999,
        value=42,
        help="Set seed for reproducible results"
    )
    
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Main content
    if run_button:
        with st.spinner("Running simulation..."):
            sim, selfish_results, social_results = run_simulation(
                num_participants, bpr_alpha, bpr_beta, 
                enable_traffic_lights, enable_intersection_delay, seed
            )
        
        st.success("✅ Simulation completed successfully!")
        
        # Metrics comparison
        st.header("📊 Performance Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        selfish_avg = selfish_results['avg_time'] / 60
        social_avg = social_results['avg_time'] / 60
        time_saved = selfish_avg - social_avg
        percent_improvement = (time_saved / selfish_avg * 100) if selfish_avg > 0 else 0
        
        with col1:
            st.metric(
                label="SELFISH Avg. Time",
                value=f"{selfish_avg:.2f} min",
                delta=None
            )
        
        with col2:
            st.metric(
                label="SOCIAL Avg. Time",
                value=f"{social_avg:.2f} min",
                delta=f"{-time_saved:.2f} min" if time_saved > 0 else f"{-time_saved:.2f} min",
                delta_color="normal" if time_saved > 0 else "inverse"
            )
        
        with col3:
            st.metric(
                label="Improvement",
                value=f"{abs(percent_improvement):.1f}%",
                delta="SOCIAL is better" if time_saved > 0 else "SELFISH is better",
                delta_color="normal" if time_saved > 0 else "inverse"
            )
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**SELFISH Routing**")
            st.write(f"- Average travel time: **{selfish_avg:.2f} min**")
            st.write(f"- Free-flow time: **{selfish_results['avg_free_flow']/60:.2f} min**")
            st.write(f"- Average delay: **{(selfish_avg - selfish_results['avg_free_flow']/60):.2f} min**")
            st.write(f"- Total system time: **{selfish_results['total_time']/3600:.2f} hrs**")
            st.write(f"- Successful routes: **{selfish_results['successful_routes']}**")
        
        with col2:
            st.markdown("**SOCIAL Routing**")
            st.write(f"- Average travel time: **{social_avg:.2f} min**")
            st.write(f"- Free-flow time: **{social_results['avg_free_flow']/60:.2f} min**")
            st.write(f"- Average delay: **{(social_avg - social_results['avg_free_flow']/60):.2f} min**")
            st.write(f"- Total system time: **{social_results['total_time']/3600:.2f} hrs**")
            st.write(f"- Successful routes: **{social_results['successful_routes']}**")
        
        # Time savings summary
        if time_saved > 0:
            total_saved = selfish_results['total_time'] / 3600 - social_results['total_time'] / 3600
            st.info(f"""
            **✅ SOCIAL routing is {percent_improvement:.1f}% better!**
            
            By coordinating routes and naturally balancing traffic load, the average travel time 
            decreased by **{time_saved:.2f} minutes** per participant.
            
            **Total time saved:** {total_saved*60:.1f} minutes = {total_saved:.2f} hours across all participants
            """)
        elif time_saved < 0:
            st.warning(f"""
            **⚠️ SELFISH routing performed better in this simulation.**
            
            This may indicate insufficient congestion or specific network characteristics.
            """)
        
        # Visualizations
        st.header("🗺️ Traffic Flow Visualization")
        st.write("Heat maps showing traffic distribution across the network")
        
        # Calculate max flow for consistent color scaling
        all_flows = list(selfish_results['flow_data'].values()) + list(social_results['flow_data'].values())
        max_flow = max(all_flows) if all_flows else 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Generating SELFISH visualization..."):
                img_selfish = create_visualization(
                    sim.G, 
                    selfish_results['flow_data'], 
                    "selfish",
                    max_flow
                )
                st.image(img_selfish, caption="SELFISH Routing - Everyone takes shortest path", use_container_width=True)
        
        with col2:
            with st.spinner("Generating SOCIAL visualization..."):
                img_social = create_visualization(
                    sim.G, 
                    social_results['flow_data'], 
                    "social",
                    max_flow
                )
                st.image(img_social, caption="SOCIAL Routing - Coordinated traffic distribution", use_container_width=True)
        
        # Flow distribution analysis
        st.header("📈 Flow Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selfish_flows = list(selfish_results['flow_data'].values())
            selfish_nonzero = [f for f in selfish_flows if f > 0]
            st.metric("SELFISH - Edges with traffic", f"{len(selfish_nonzero)}")
            st.metric("SELFISH - Max edge flow", f"{max(selfish_flows)}")
            st.metric("SELFISH - Avg edge flow (non-zero)", f"{np.mean(selfish_nonzero):.1f}" if selfish_nonzero else "0")
        
        with col2:
            social_flows = list(social_results['flow_data'].values())
            social_nonzero = [f for f in social_flows if f > 0]
            st.metric("SOCIAL - Edges with traffic", f"{len(social_nonzero)}")
            st.metric("SOCIAL - Max edge flow", f"{max(social_flows)}")
            st.metric("SOCIAL - Avg edge flow (non-zero)", f"{np.mean(social_nonzero):.1f}" if social_nonzero else "0")
        
        # Insights
        st.header("💡 Key Insights")
        
        flow_spread_selfish = len(selfish_nonzero)
        flow_spread_social = len(social_nonzero)
        
        insights = []
        
        if flow_spread_social > flow_spread_selfish:
            insights.append("✅ SOCIAL routing utilizes more edges, distributing traffic more evenly")
        
        if max(social_flows) < max(selfish_flows):
            insights.append("✅ SOCIAL routing reduces peak congestion on individual edges")
        
        if time_saved > 0:
            insights.append(f"✅ Coordination saves {time_saved:.2f} minutes per trip on average")
        
        if insights:
            for insight in insights:
                st.write(insight)
        
    else:
        # Initial state - show instructions
        st.info("""
        👈 **Configure simulation parameters in the sidebar and click 'Run Simulation' to begin.**
        
        ### What does this simulation show?
        
        This tool compares two traffic routing strategies:
        
        1. **SELFISH Routing**: Each driver independently chooses their shortest path, ignoring the impact on others.
           This leads to congestion on popular routes.
        
        2. **SOCIAL Routing**: A coordinated approach where routing decisions consider current traffic conditions,
           naturally distributing vehicles across alternative paths.
        
        ### Key Features:
        - Real road network from OpenStreetMap (Munich to Taufkirchen area)
        - Realistic congestion modeling using BPR (Bureau of Public Roads) function
        - Traffic light and intersection delays
        - Visual heat maps showing traffic distribution
        - Detailed performance metrics
        
        ### Expected Results:
        SOCIAL routing typically reduces average travel time by **5-15%** compared to SELFISH routing,
        demonstrating the benefits of coordinated traffic management.
        """)
        
        # Show example visualization
        st.image("https://raw.githubusercontent.com/gboeing/osmnx-examples/main/notebooks/images/graph-simple.png", 
                caption="Example: Road network visualization (actual simulation uses Munich area)", 
                use_container_width=True)


if __name__ == "__main__":
    main()
