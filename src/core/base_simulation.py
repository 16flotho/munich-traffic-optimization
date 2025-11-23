class BaseSimulation:
    def __init__(self, num_participants):
        self.num_participants = num_participants
        self.G = None
        self.participants = []

    def load_network(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_participants(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def run_simulation(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def calculate_edge_time(self, edge_data, include_delays=True):
        raise NotImplementedError("Subclasses should implement this method.")

    def bpr_congestion_function(self, free_flow_time, flow, capacity):
        raise NotImplementedError("Subclasses should implement this method.")