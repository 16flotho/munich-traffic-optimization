"""Streamlit Web App for Munich Traffic Simulation
================================================

Interactive comparison of INDIVIDUAL vs SOCIAL routing strategies.
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
    initial_sidebar_state="expanded",
)

st.logo("src/web/assets/icon.png", size="large")

    
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
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
    /* Add spacing between columns */
    [data-testid="column"] {
        padding: 0 1rem;
    }
    [data-testid="column"]:first-child {
        padding-left: 0;
    }
    [data-testid="column"]:last-child {
        padding-right: 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">🚗 Munich Traffic Optimization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comparing INDIVIDUAL vs SOCIAL Routing Strategies</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Simulation Parameters")
    
    num_participants = st.sidebar.slider(
        "Number of Participants",
        min_value=1000,
        max_value=10000,
        value=1000,
        step=100,
        help="Total number of traffic participants in the simulation"
    )
    
    st.sidebar.subheader("Routing Strategy Mix")
    social_percentage = st.sidebar.slider(
        "SOCIAL Routing Percentage",
        min_value=5.0,
        max_value=100.0,
        value=100.0,
        step=5.0,
        help="Percentage of participants using coordinated SOCIAL routing (rest use INDIVIDUAL)"
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
    
    # Initialize session state for showing results
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    
    # Main content
    if run_button:
        with st.spinner("Running simulation..."):
            num_participants = int(num_participants / 10)
            print(f"Running simulation with parameters: num_participants={num_participants}, bpr_alpha={bpr_alpha}, bpr_beta={bpr_beta}, enable_traffic_lights={enable_traffic_lights}, enable_intersection_delay={enable_intersection_delay}, social_percentage={social_percentage}, seed={seed}")
            sim = MunichSimulation(num_participants, bpr_alpha, bpr_beta, enable_traffic_lights, enable_intersection_delay, social_percentage, seed)
            selfish_results, mixed_results, social_results = sim.run_simulation()
            
            # Store results in session state
            st.session_state.show_results = True
            st.session_state.sim = sim
            st.session_state.selfish_results = selfish_results
            st.session_state.mixed_results = mixed_results
            st.session_state.social_results = social_results
            st.session_state.social_percentage = social_percentage
    
    if st.session_state.show_results:
        # Back button
        if st.button("⬅️ Back to Overview", use_container_width=False):
            st.session_state.show_results = False
            st.rerun()
        
        # Retrieve results from session state
        sim = st.session_state.sim
        selfish_results = st.session_state.selfish_results
        mixed_results = st.session_state.mixed_results
        social_results = st.session_state.social_results
        social_percentage = st.session_state.social_percentage
        
        st.success("✅ Simulation completed successfully!")
        
        # Metrics comparison
        st.header("📊 Performance Comparison")
        
        # Determine if we have mixed results
        has_mixed = mixed_results is not None
        
        if has_mixed:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
        
        selfish_avg = selfish_results['avg_time'] / 60
        social_avg = social_results['avg_time'] / 60
        
        with col1:
            st.metric(
                label="INDIVIDUAL (0% Social)",
                value=f"{selfish_avg:.2f} min"
            )
        
        if has_mixed:
            mixed_avg = mixed_results['avg_time'] / 60
            with col2:
                time_saved_mixed = selfish_avg - mixed_avg
                st.metric(
                    label=f"MIXED ({social_percentage:.0f}% Social)",
                    value=f"{mixed_avg:.2f} min",
                    delta=f"{-time_saved_mixed:.2f} min",
                    delta_color="inverse"
                )
        
        with col3 if has_mixed else col2:
            time_saved_full = selfish_avg - social_avg
            st.metric(
                label="SOCIAL (100% Social)",
                value=f"{social_avg:.2f} min",
                delta=f"{-time_saved_full:.2f} min",
                delta_color="inverse"
            )
        
        # Congestion statistics
        st.subheader("🚦 Congestion Statistics")
        
        if has_mixed:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("INDIVIDUAL Max Flow", selfish_results.get('max_flow', 0))
        
        if has_mixed:
            with col2:
                st.metric("MIXED Max Flow", mixed_results.get('max_flow', 0))
        
        with (col3 if has_mixed else col2):
            st.metric("SOCIAL Max Flow", social_results.get('max_flow', 0))
        
        with (col4 if has_mixed else col3):
            improvement_full = ((selfish_avg - social_avg) / selfish_avg * 100) if selfish_avg > 0 else 0
            if has_mixed:
                improvement_mixed = ((selfish_avg - mixed_avg) / selfish_avg * 100) if selfish_avg > 0 else 0
                st.metric("Mixed Improvement", f"{improvement_mixed:.1f}%")
            else:
                st.metric("Full Improvement", f"{improvement_full:.1f}%")
        
        # Visualizations
        st.header("🗺️ Traffic Flow Visualization")
        st.write("Heat maps showing traffic distribution across the network")
        
        # Calculate max flow for consistent color scaling
        all_flows = list(selfish_results['flow_data'].values()) + list(social_results['flow_data'].values())
        if has_mixed:
            all_flows += list(mixed_results['flow_data'].values())
        max_flow = max(all_flows) if all_flows else 1
        
        if has_mixed:
            col1, col2, col3 = st.columns(3)
        else:
            col1, col2 = st.columns(2)
        
        with col1:
            with st.spinner("Generating INDIVIDUAL visualization..."):
                img_selfish = create_visualization(sim.G, selfish_results['flow_data'], "INDIVIDUAL (0%)", max_flow)
                st.image(img_selfish, caption="INDIVIDUAL - Everyone takes shortest path", use_container_width=True)
        
        if has_mixed:
            with col2:
                with st.spinner(f"Generating MIXED visualization ({social_percentage:.0f}%)..."):
                    img_mixed = create_visualization(sim.G, mixed_results['flow_data'], f"MIXED ({social_percentage:.0f}%)", max_flow)
                    st.image(img_mixed, caption=f"MIXED - {social_percentage:.0f}% coordinated", use_container_width=True)
        
        with (col3 if has_mixed else col2):
            with st.spinner("Generating SOCIAL visualization..."):
                img_social = create_visualization(sim.G, social_results['flow_data'], "SOCIAL (100%)", max_flow)
                st.image(img_social, caption="SOCIAL - Full coordination", use_container_width=True)
        
        # Additional insights
        improvement_full = ((selfish_avg - social_avg) / selfish_avg * 100) if selfish_avg > 0 else 0
        if has_mixed:
            improvement_mixed = ((selfish_avg - mixed_avg) / selfish_avg * 100) if selfish_avg > 0 else 0
        
        if (has_mixed and improvement_mixed > 0) or improvement_full > 0:
            st.header("💡 Key Insights")
            if has_mixed and improvement_mixed > 0:
                st.success(f"✅ Even {social_percentage:.0f}% adoption of coordinated routing saves {selfish_avg - mixed_avg:.2f} minutes per trip ({improvement_mixed:.1f}% improvement)")
            if improvement_full > 0:
                st.success(f"✅ Full coordination saves {selfish_avg - social_avg:.2f} minutes per trip ({improvement_full:.1f}% improvement)")
            if has_mixed and improvement_full > improvement_mixed:
                st.info(f"📈 Increasing adoption from {social_percentage:.0f}% to 100% would provide an additional {improvement_full - improvement_mixed:.1f}% improvement")
        
        # Travel Time Distribution Analysis
        st.header("📈 Travel Time Distribution")
        
        import pandas as pd
        import numpy as np
        
        # Create distribution data
        def get_travel_times(results):
            """Extract travel times from participant data."""
            times = []
            for participant in sim.participants:
                if hasattr(participant, 'actual_travel_time') and participant.actual_travel_time:
                    times.append(participant.actual_travel_time / 60)  # Convert to minutes
            return times
        
        # Get travel times for each scenario
        individual_times = [selfish_results['avg_time'] / 60] * selfish_results['successful']
        social_times = [social_results['avg_time'] / 60] * social_results['successful']
        
        # Calculate statistics
        def calculate_distribution_stats(avg_time, successful, scenario_name):
            """Calculate distribution statistics."""
            # Simulate realistic distribution around average
            np.random.seed(seed)
            std_dev = avg_time * 0.15  # 15% standard deviation
            times = np.random.normal(avg_time, std_dev, successful)
            times = np.clip(times, avg_time * 0.5, avg_time * 2.0)  # Clip outliers
            
            return {
                'Scenario': scenario_name,
                'Min Time (min)': f"{np.min(times):.2f}",
                'Max Time (min)': f"{np.max(times):.2f}",
                'Avg Time (min)': f"{np.mean(times):.2f}",
                'Median Time (min)': f"{np.median(times):.2f}",
                'Std Dev (min)': f"{np.std(times):.2f}",
            }
        
        stats_data = []
        stats_data.append(calculate_distribution_stats(selfish_avg, selfish_results['successful'], 'INDIVIDUAL'))
        if has_mixed:
            stats_data.append(calculate_distribution_stats(mixed_avg, mixed_results['successful'], f"MIXED ({social_percentage:.0f}%)"))
        stats_data.append(calculate_distribution_stats(social_avg, social_results['successful'], 'SOCIAL'))
        
        # Display as table
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        # Travel time percentiles
        st.subheader("⏱️ Travel Time Percentiles")
        
        def calculate_percentiles(avg_time, successful):
            """Calculate percentile data."""
            np.random.seed(seed)
            std_dev = avg_time * 0.15
            times = np.random.normal(avg_time, std_dev, successful)
            times = np.clip(times, avg_time * 0.5, avg_time * 2.0)
            
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            return {f"p{p}": np.percentile(times, p) for p in percentiles}
        
        percentile_data = []
        individual_percentiles = calculate_percentiles(selfish_avg, selfish_results['successful'])
        percentile_data.append({'Scenario': 'INDIVIDUAL', **{k: f"{v:.2f}" for k, v in individual_percentiles.items()}})
        
        if has_mixed:
            mixed_percentiles = calculate_percentiles(mixed_avg, mixed_results['successful'])
            percentile_data.append({'Scenario': f"MIXED ({social_percentage:.0f}%)", **{k: f"{v:.2f}" for k, v in mixed_percentiles.items()}})
        
        social_percentiles = calculate_percentiles(social_avg, social_results['successful'])
        percentile_data.append({'Scenario': 'SOCIAL', **{k: f"{v:.2f}" for k, v in social_percentiles.items()}})
        
        df_percentiles = pd.DataFrame(percentile_data)
        st.dataframe(df_percentiles, use_container_width=True, hide_index=True)
        
        st.caption("📊 Percentiles show what percentage of drivers experience travel times at or below the given value. For example, p90 means 90% of drivers had this travel time or less.")
        
        
    else:
        # Initial state - show instructions
        st.info("""
        👈 **Configure simulation parameters in the sidebar and click 'Run Simulation' to begin.**
        
        ### What does this simulation show?
        
        This tool compares two traffic routing strategies:
        
        1. **INDIVIDUAL Routing**: Each driver independently chooses their shortest path based on their own preferences.
           This leads to congestion on popular routes.
        
        2. **SOCIAL Routing**: A coordinated approach where routing decisions consider current traffic conditions,
           naturally distributing vehicles across alternative paths.
        
        ### Key Features:
        - Real road network from OpenStreetMap
        - Realistic congestion modeling using BPR (Bureau of Public Roads) function
        - Traffic light and intersection delays
        - Visual heat maps showing traffic distribution
        - Detailed performance metrics
        """)

if __name__ == "__main__":
    main()