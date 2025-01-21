#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include "UKF.h"

const size_t BUFFER_SIZE = 10; // Define the size of the circular buffer
std::vector<Vector3d> circular_buffer(BUFFER_SIZE, Vector3d::Zero()); // Initialize with zeros
size_t buffer_index = 0; // Start index for the buffer

// Function to update the circular buffer with new state data
void updateBuffer(const Vector3d& new_state) {
    circular_buffer[buffer_index] = new_state;
    buffer_index = (buffer_index + 1) % BUFFER_SIZE;
}

// Function to retrieve the most recent state from the buffer
Vector3d getLatestState() {
    size_t latest_index = (buffer_index + BUFFER_SIZE - 1) % BUFFER_SIZE;
    return circular_buffer[latest_index];
}

int main() {
    std::thread ukf_thread([]() {
        try {
            while (true) {
                Vector3d state = locate();
                updateBuffer(state);

                // Simulate data processing delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } catch (const std::exception& e) {
            std::cerr << "UKF Exception: " << e.what() << std::endl;
        }
    });

    while (true) {
        Vector3d latest_state = getLatestState();
        std::cout << "Latest State - x: " << latest_state(0) 
                  << ", y: " << latest_state(1) 
                  << ", yaw: " << latest_state(2) << std::endl;

        // Simulate heavy processing or external code execution
        std::this_thread::sleep_for(std::chrono::seconds(5)); // Simulate lag
    }

    ukf_thread.join(); // This line is never reached due to infinite loops
    return 0;
}

