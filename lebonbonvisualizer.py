
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load keystroke data
with open("raw_data\jouman_attempt_1_raw.json", "r") as f:
    data = json.load(f)

# Process data to calculate dwell and flight times
dwell_times = {}
flight_times = []
key_press_times = {}
previous_release_time = None

for event in data:
    key = event["key"]
    event_type = event["event"]
    time = event["time"]

    if event_type == "press":
        key_press_times[key] = time
    elif event_type == "release" and key in key_press_times:
        # Calculate dwell time
        dwell_time = time - key_press_times[key]
        dwell_times[key] = dwell_times.get(key, []) + [dwell_time]
        del key_press_times[key]

        # Calculate flight time
        if previous_release_time is not None:
            flight_time = time - previous_release_time
            flight_times.append(flight_time)
        previous_release_time = time

# Flatten dwell time data for easier visualization
keys = []
dwell_time_values = []
for key, times in dwell_times.items():
    keys.extend([key] * len(times))
    dwell_time_values.extend(times)

# Visualization 1: Dwell times for each key
plt.figure(figsize=(12, 6))
sns.barplot(x=keys, y=dwell_time_values, palette="viridis")
plt.title("Dwell Times for Each Key")
plt.xlabel("Key")
plt.ylabel("Dwell Time (seconds)")
plt.show()

# Visualization 2: Flight time distribution
plt.figure(figsize=(10, 5))
sns.histplot(flight_times, bins=20, kde=True, color="blue")
plt.title("Distribution of Flight Times")
plt.xlabel("Flight Time (seconds)")
plt.ylabel("Frequency")
plt.show()
