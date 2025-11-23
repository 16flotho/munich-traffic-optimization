"""
Streamlit Web App for Munich Traffic Simulation
================================================

Interactive comparison of SELFISH vs SOCIAL routing strategies.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(src_dir))

import streamlit as st
from core.config import MUNICH_CENTER, DIST_FROM_CENTER
from simulation.munich_simulation import MunichSimulation
from visualization.graph_plotter import create_visualization

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


def main():
    # Header
    st.markdown('<div class="main-header">🚗 Munich Traffic Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing SELFISH vs SOCIAL Routing Strategies</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Simulation Parameters")
    
    num_participants = st.sidebar.slider(
        "Number of Participants",
        min_value=100,
        max_value=20000,
        value=10000,
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
            print(f"Running simulation with parameters: num_participants={num_participants}, bpr_alpha={bpr_alpha}, bpr_beta={bpr_beta}, enable_traffic_lights={enable_traffic_lights}, enable_intersection_delay={enable_intersection_delay}, seed={seed}")
            sim = MunichSimulation(num_participants, bpr_alpha, bpr_beta, enable_traffic_lights, enable_intersection_delay, seed)
            selfish_results, social_results = sim.run_simulation()
        
        st.success("✅ Simulation completed successfully!")
        
        # Display simulation parameters used
        st.info(f"🔧 **Simulation Parameters:** α={bpr_alpha}, β={bpr_beta}, "
                f"Traffic Lights={'On' if enable_traffic_lights else 'Off'}, "
                f"Intersection Delays={'On' if enable_intersection_delay else 'Off'}, "
                f"Participants={num_participants}, Seed={seed}")
        
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
                delta_color="inverse" if time_saved > 0 else "normal"
            )
        
        with col3:
            st.metric(
                label="Improvement",
                value=f"{abs(percent_improvement):.1f}%",
                delta="SOCIAL is better" if time_saved > 0 else "SELFISH is better",
                delta_color="normal" if time_saved > 0 else "inverse"
            )
        
        # Congestion statistics
        st.subheader("🚦 Congestion Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("SELFISH Max Flow", selfish_results.get('max_flow', 0))
        with col2:
            st.metric("SOCIAL Max Flow", social_results.get('max_flow', 0))
        with col3:
            st.metric("SELFISH Congested Edges", selfish_results.get('congested_edges', 0))
        with col4:
            st.metric("SOCIAL Congested Edges", social_results.get('congested_edges', 0))
        
        # Visualizations
        st.header("🗺️ Traffic Flow Visualization")
        st.write("Heat maps showing traffic distribution across the network")
        
        # Calculate max flow for consistent color scaling
        all_flows = list(selfish_results['flow_data'].values()) + list(social_results['flow_data'].values())
        max_flow = max(all_flows) if all_flows else 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Generating SELFISH visualization..."):
                img_selfish = create_visualization(sim.G, selfish_results['flow_data'], "selfish", max_flow)
                st.image(img_selfish, caption="SELFISH Routing - Everyone takes shortest path", use_container_width=True)
        
        with col2:
            with st.spinner("Generating SOCIAL visualization..."):
                img_social = create_visualization(sim.G, social_results['flow_data'], "social", max_flow)
                st.image(img_social, caption="SOCIAL Routing - Coordinated traffic distribution", use_container_width=True)
        
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

if __name__ == "__main__":
    main()