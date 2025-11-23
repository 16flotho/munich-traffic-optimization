"""
Graph Plotter for Traffic Simulation Visualization
==================================================

This module provides classes and functions for visualizing traffic flow and network graphs.
It includes methods for generating heat maps and plotting results from traffic simulations.
"""

import osmnx as ox
import matplotlib.pyplot as plt
import io

class GraphPlotter:
    """Class for plotting traffic flow on a network graph."""
    
    def __init__(self, graph):
        self.graph = graph

    def create_visualization(self, flow_data, scenario, max_flow):
        """Create traffic flow visualization."""
        edge_colors = []
        edge_widths = []
        
        for u, v, k in self.graph.edges(keys=True):
            flow = flow_data.get((u, v, k), 0)
            
            if flow == 0:
                edge_colors.append('#e0e0e0')
                edge_widths.append(0.3)
            else:
                intensity = min(flow / max_flow, 1.0)
                if intensity < 0.2:
                    edge_colors.append("#158E00")  # Light green
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
            self.graph,
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
    
# Standalone function for convenience (wraps GraphPlotter)
def create_visualization(graph, flow_data, scenario, max_flow):
    """
    Create traffic flow visualization.
    
    Args:
        graph: NetworkX graph
        flow_data: Dictionary mapping (u, v, k) tuples to flow values
        scenario: String describing the scenario ('selfish' or 'social')
        max_flow: Maximum flow value for color scaling
    
    Returns:
        BytesIO buffer containing the PNG image
    """
    plotter = GraphPlotter(graph)
    return plotter.create_visualization(flow_data, scenario, max_flow)