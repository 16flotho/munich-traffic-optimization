"""
Routing Strategies for Traffic Simulation
==========================================

This module defines routing strategy classes for traffic simulations, specifically
the SELFISH and SOCIAL routing strategies.
"""

from core.base_simulation import BaseSimulation

class SelfishRouting(BaseSimulation):
    """Implements the SELFISH routing strategy where each participant takes their shortest path."""
    
    def run(self):
        """Run the selfish routing simulation."""
        # Reset all flows
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
        
        successful_routes = 0
        failed_routes = 0
        
        for participant in self.participants:
            try:
                route = self.find_shortest_path(participant.origin, participant.destination)
                participant.route = route
                
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    edge_data = self.G.get_edge_data(u, v)
                    key = min(edge_data, key=lambda k: edge_data[k]['travel_time'])
                    self.G[u][v][key]['flow'] += 1
                
                successful_routes += 1
                
            except Exception:
                failed_routes += 1
                participant.route = None
        
        return successful_routes, failed_routes

class SocialRouting(BaseSimulation):
    """Implements the SOCIAL routing strategy where routing decisions consider current traffic conditions."""
    
    def run(self):
        """Run the social routing simulation."""
        for u, v, k, d in self.G.edges(keys=True, data=True):
            d['flow'] = 0
            d['current_cost'] = d['travel_time']
        
        successful_routes = 0
        failed_routes = 0
        
        sorted_participants = sorted(self.participants, key=lambda p: p.desired_arrival)
        
        for participant in sorted_participants:
            try:
                route = self.find_shortest_path(participant.origin, participant.destination, weight='current_cost')
                participant.route = route
                
                actual_time = 0
                for j in range(len(route) - 1):
                    u, v = route[j], route[j + 1]
                    edge_data = self.G.get_edge_data(u, v)
                    key = min(edge_data, key=lambda k: edge_data[k]['current_cost'])
                    data = self.G[u][v][key]
                    
                    actual_time += data['current_cost']
                    data['flow'] += 1
                    
                    new_cost = self.calculate_edge_time(data, include_delays=True)
                    data['current_cost'] = new_cost
                
                participant.actual_travel_time = actual_time
                successful_routes += 1
                
            except Exception:
                failed_routes += 1
                participant.route = None
        
        return successful_routes, failed_routes