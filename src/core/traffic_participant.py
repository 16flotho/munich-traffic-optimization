class TrafficParticipant:
    """Represents a single traffic participant with origin, destination, and desired arrival time."""
    
    def __init__(self, participant_id, origin_node, dest_node, arrival_time):
        self.id = participant_id
        self.origin = origin_node
        self.destination = dest_node
        self.desired_arrival = arrival_time
        self.route = None
        self.actual_travel_time = 0  # seconds
        self.free_flow_time = 0  # seconds
        
    def __repr__(self):
        return f"Participant {self.id}: {self.origin} -> {self.destination} @ {self.desired_arrival.strftime('%H:%M')}"