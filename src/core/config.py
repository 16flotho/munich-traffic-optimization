"""
Configuration Constants for Munich Traffic Simulation
=====================================================

This module contains configuration constants and parameters used throughout the simulation,
such as the center of Munich, distance from the center, and congestion modeling parameters.
"""

from datetime import datetime

# Geographic configuration
MUNICH_CENTER = (48.1374, 11.5755)  # Latitude and longitude of Munich center
DIST_FROM_CENTER = 15000  # Distance from center in meters

# Simulation parameters
NUM_PARTICIPANTS = 1000  # Default number of traffic participants
SEED = 42  # Random seed for reproducibility

# Time window for arrivals (morning rush hour)
START_TIME = datetime.strptime("07:00", "%H:%M")  # Start time for traffic simulation
END_TIME = datetime.strptime("09:00", "%H:%M")    # End time for traffic simulation

# Congestion modeling parameters
ENABLE_TRAFFIC_LIGHTS = True  # Flag to enable traffic light delays
ENABLE_INTERSECTION_DELAY = True  # Flag to enable intersection delays
BPR_ALPHA = 0.5  # BPR congestion sensitivity parameter
BPR_BETA = 4.0   # BPR congestion power parameter
