import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('2outukf.csv')

# Extract the data for plotting
actual_x = df['Actual x']  # X coordinates for actual x
actual_y = df['Actual y']  # Y coordinates for actual y
estimated_x = df['Estimated x']  # X coordinates for estimated x
estimated_y = df['Estimated y']  # Y coordinates for estimated y

# Create a figure and axis object
plt.figure(figsize=(10, 6))

# Plot Actual X vs Actual Y as a single curve
plt.plot(actual_x, actual_y, label='Actual Position', marker='o', color='b', linestyle='-', linewidth=2)

# Plot Estimated X vs Estimated Y as a single curve
plt.plot(estimated_x, estimated_y, label='Estimated Position', marker='x', color='r', linestyle='--', linewidth=2)

# Add labels and title
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('UKF Position Estimates vs Actual Positions')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.tight_layout()  # Make sure everything fits
plt.show()

