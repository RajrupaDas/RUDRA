#include <iostream>
#include <thread>
#include <fstream>
#include <cmath>
#include <random>
#include <chrono>

using namespace std;

class PID {
public:
    double Kp, Ki, Kd;
    double prev_error, integral;
    PID(double p, double i, double d) : Kp(p), Ki(i), Kd(d), prev_error(0), integral(0) {}
    double compute(double setpoint, double measured, double dt) {
        double error = setpoint - measured;
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;
        return Kp * error + Ki * integral + Kd * derivative;
    }
};

struct Pose {
    double x, y, yaw;
};

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> noise(-0.05, 0.05);

void logData(ofstream &file, double actual_yaw, double sensor_yaw, double actual_x, double actual_y, double sensor_x, double sensor_y) {
    file << actual_yaw << "," << sensor_yaw << "," << actual_x << "," << actual_y << "," << sensor_x << "," << sensor_y << "\n";
}

int main() {
    // PID gains
    PID yawPID(0.5, 0.01, 0.1);
    PID posPID(0.5, 0.01, 0.1);

    // Initial conditions
    Pose current = {0, 0, 0};
    Pose destination = {5, 5};
    double linear_vel = 0.1;
    double angular_vel = 0.1;
    double dt = 0.1;
    
    ofstream file("motion_data.csv");
    file << "ActualYaw,SensorYaw,ActualX,ActualY,SensorX,SensorY\n";
    
    double target_yaw = atan2(destination.y - current.y, destination.x - current.x);
    
    // Turning phase
    while (abs(current.yaw - target_yaw) > 0.01) {
        double yaw_control = yawPID.compute(target_yaw, current.yaw, dt);
        current.yaw += yaw_control * angular_vel * dt;
        logData(file, current.yaw, current.yaw + noise(gen), current.x, current.y, current.x + noise(gen), current.y + noise(gen));
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    // Moving phase
    while (sqrt(pow(destination.x - current.x, 2) + pow(destination.y - current.y, 2)) > 0.1) {
        double pos_control = posPID.compute(sqrt(pow(destination.x - current.x, 2) + pow(destination.y - current.y, 2)), 0, dt);
        current.x += pos_control * linear_vel * cos(current.yaw) * dt;
        current.y += pos_control * linear_vel * sin(current.yaw) * dt;
        logData(file, current.yaw, current.yaw + noise(gen), current.x, current.y, current.x + noise(gen), current.y + noise(gen));
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    file.close();
    cout << "Simulation complete. Data stored in motion_data.csv" << endl;
    return 0;
}
