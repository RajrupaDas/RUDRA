# RUDRA

C++ code for performing localization via sensor integration using Unscented Kalman Filter. 

Sensors used here are:
1. GPS (x and y coordinates)
2. IMU (linear accleration, angular velocity)
3. Control Inputs (acceleration and rate of change of yaw)

Configuration:

src/1main.cpp
lib/UKF.cpp
include/UKF.h

Compile with:
g++ -Iinclude -I/usr/include/eigen3 src/1main.cpp lib/UKF.cpp -o output_executable

*change path to eigen as needed
