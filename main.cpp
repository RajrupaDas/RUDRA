#include <iostream>
#include "UKF.h" // Include your header

int main() {
    try {
        Vector3d result = locate();
        std::cout << "x: " << result(0) << ", y: " << result(1) << ", yaw: " << result(2) << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

