#include <iostream>
#include <thread>
#include <chrono>
#include "UKF.h"

int main() {
    try {
        while (true) {
            Vector3d state = locate();
            std::cout << "x: " << state(0) << ", y: " << state(1) << ", yaw: " << state(2) << std::endl;

            // Sleep for a short duration to simulate real-time data processing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}

