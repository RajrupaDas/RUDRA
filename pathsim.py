import numpy as np
import matplotlib.pyplot as plt
import csv

# Simulation parameters
radius = 5  # Radius of the circle in meters
center = (0, 0)  # Center of the circle
dt = 0.1  # Time step in seconds

# Velocity settings
base_velocity = 1  # Average velocity in m/s
velocity_variation = 0.2  # Maximum variation in velocity

# Circle properties
total_angle = 2 * np.pi  # Full circle in radians
angle = 0  # Start angle in radians

# Initialize position, velocity, and acceleration
x, y = [], []
prev_velocity = base_velocity  # Initial velocity
prev_angular_velocity = base_velocity / radius  # Initial angular velocity
# File for CSV logging
csv_file = "simulation_data.csv"

# Open CSV file and write the header
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["gps_x", "gps_y", "imu_linear_acc", "imu_angular_vel", "linear_vel", "angular_accel", "additional_accel", "yaw_rate"])  # CSV Header

# Simulation loop
while angle < total_angle:
    # Vary the velocity randomly
    velocity = base_velocity + np.random.uniform(-velocity_variation, velocity_variation)
    
    # Calculate linear acceleration (Δv / Δt)
    imu_linear_acc = (velocity - prev_velocity) / dt  # Linear acceleration
    
    # Calculate angular velocity (v = r * ω --> ω = v / r)
    angular_velocity = velocity / radius

    additional_accel = imu_linear_acc + np.random.uniform(-0.2, 0.2)
    
    # Calculate angular acceleration (Δω / Δt)
    imu_angular_accel = (angular_velocity - prev_angular_velocity) / dt  # Angular acceleration
    
    yaw_rate = imu_angular_accel + np.random.uniform(-0.2, 0.2)

    # Update previous values for the next iteration
    prev_velocity = velocity
    prev_angular_velocity = angular_velocity
    
    # Update the angle
    angle += angular_velocity * dt
    
    # Cap the angle at the full circle
    if angle > total_angle:
        angle = total_angle  # Snap to exactly 2π
    
    # Calculate new position
    new_x = center[0] + radius * np.cos(angle)
    new_y = center[1] + radius * np.sin(angle)
    x.append(new_x)
    y.append(new_y)
    
    # Simulate GPS and IMU readings
    gps_x = new_x
    gps_y = new_y
    gps_yaw = np.degrees(angle) % 360  # Convert angle to degrees

    # Write to CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([gps_x, gps_y, imu_linear_acc, angular_velocity, velocity, imu_angular_accel, additional_accel, yaw_rate])
    
    # Visualization
    plt.clf()
    plt.xlim(-radius - 1, radius + 1)
    plt.ylim(-radius - 1, radius + 1)
    plt.grid(True)
    plt.plot(x, y, '-o', markersize=4, label="Path")
    plt.scatter(new_x, new_y, color="red", label="Current Position")
    plt.legend()
    plt.title(f"Angle: {gps_yaw:.1f}° | Velocity: {velocity:.2f} m/s | Accel: {imu_linear_acc:.2f} m/s²")
    plt.pause(0.05)  # Pause to simulate real-time movement

plt.show()

print(f"Simulation complete! Data saved to '{csv_file}'.")

