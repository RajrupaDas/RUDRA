#include <iostream>
#include <cmath>
#include <unistd.h>
using namespace std;

double kp_turn = 1.0, ki_turn = 0.0, kd_turn = 0.0;
double kp_trans = 1.0, ki_trans = 0.0, kd_trans = 0.0;
double angularVelocity = 0.1;
double linearVelocity = 0.1;

class PIDController {
private:
    double kp, ki, kd;
    double integral, prev_error;

public:
    PIDController(double p, double i, double d) 
        : kp(p), ki(i), kd(d), integral(0.0), prev_error(0.0) {}

    double compute(double setpoint, double current) {
        double error = setpoint - current;
        integral += error;
        double derivative = error - prev_error;
        prev_error = error;

        return (kp * error) + (ki * integral) + (kd * derivative);
    }

    void reset() {
        integral = 0.0;
        prev_error = 0.0;
    }
};

class MotionController {
private:
    PIDController turnController;
    PIDController translationController;

public:
    MotionController() 
        : turnController(kp_turn, ki_turn, kd_turn), 
          translationController(kp_trans, ki_trans, kd_trans) {}

    void moveToTarget(double current_x, double current_y, double current_yaw, double target_x, double target_y) {
        double delta_x = target_x - current_x;
        double delta_y = target_y - current_y;
        double required_yaw = atan2(delta_y, delta_x) * 180 / M_PI;

        // Turning phase
        while (abs(required_yaw - current_yaw) > 0.5) {
            double turn_output = turnController.compute(required_yaw, current_yaw);
            cout << "Turning... Current Yaw: " << current_yaw 
                 << " Angular Velocity: " << angularVelocity * (turn_output / abs(turn_output)) << endl;
            current_yaw += angularVelocity * (turn_output / abs(turn_output));
            sleep(0.01);
        }

        cout << "Turning complete. Now translating..." << endl;

        // Translation phase
        while (sqrt(delta_x * delta_x + delta_y * delta_y) > 0.5) {
            double trans_output = translationController.compute(sqrt(delta_x * delta_x + delta_y * delta_y), 0);
            cout << "Moving... Current Position: (" << current_x << ", " << current_y 
                 << ") Linear Velocity: " << linearVelocity * (trans_output / abs(trans_output)) << endl;

            current_x += linearVelocity * (delta_x / sqrt(delta_x * delta_x + delta_y * delta_y));
            current_y += linearVelocity * (delta_y / sqrt(delta_x * delta_x + delta_y * delta_y));
            delta_x = target_x - current_x;
            delta_y = target_y - current_y;
            sleep(0.01);
        }
        cout << "Destination reached!" << endl;
    }
};

int main() {
    double start_x, start_y, start_yaw, end_x, end_y;

    cout << "Enter start X: ";
    cin >> start_x;
    cout << "Enter start Y: ";
    cin >> start_y;
    cout << "Enter start Yaw: ";
    cin >> start_yaw;
    cout << "Enter target X: ";
    cin >> end_x;
    cout << "Enter target Y: ";
    cin >> end_y;

    MotionController controller;
    controller.moveToTarget(start_x, start_y, start_yaw, end_x, end_y);

    return 0;
}
