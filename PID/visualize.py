import pandas as pd
import matplotlib.pyplot as plt

def plot_data(filename):
    df = pd.read_csv(filename)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot yaw data
    axes[0].plot(df['ActualYaw'], label='Actual Yaw', color='blue')
    axes[0].plot(df['SensorYaw'], label='Sensor Yaw (with noise)', color='red', linestyle='dashed')
    axes[0].set_title('Yaw Control')
    axes[0].set_ylabel('Yaw (radians)')
    axes[0].legend()
    
    # Plot position data
    axes[1].plot(df['ActualX'], df['ActualY'], label='Actual Position', color='blue')
    axes[1].plot(df['SensorX'], df['SensorY'], label='Sensor Position (with noise)', color='red', linestyle='dashed')
    axes[1].set_title('Position Control')
    axes[1].set_xlabel('X Position')
    axes[1].set_ylabel('Y Position')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_data("motion_data.csv")
