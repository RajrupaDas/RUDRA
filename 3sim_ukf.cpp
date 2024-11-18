#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <random>

using namespace Eigen;
using namespace std;

// Define the state dimension and measurement dimension
const int STATE_DIM = 6;  // State vector: [x, y, vel_x, vel_y, yaw, yaw_rate]
const int MEASUREMENT_DIM = 2;  // Measurement vector: [x, y]

// Define the time step (dt)
double dt = 0.1;  // Time step in seconds

// UKF Class
class UKF {
public:
    VectorXd state_;   // State vector [x, y, vel_x, vel_y, yaw, yaw_rate]
    MatrixXd P_;       // State covariance matrix
    MatrixXd Q_;       // Process noise covariance matrix
    MatrixXd R_;       // Measurement noise covariance matrix

    UKF() {
        state_ = VectorXd(STATE_DIM);  // Initialize state vector
        P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1;  // Initial state covariance matrix
        Q_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01;  // Process noise covariance matrix
        R_ = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 1.0;  // Measurement noise covariance matrix

        // Initial state (position at origin, velocity 0, and yaw 0)
        state_ << 0, 0, 1, 1, 0, 0;  // initial state [x, y, vel_x, vel_y, yaw, yaw_rate]
    }

    // Prediction Step
    void predict(const VectorXd& control_input) {
        // Control input (acceleration_x, acceleration_y, yaw_rate)
        double acc_x = control_input(0);
        double acc_y = control_input(1);
        double yaw_rate = control_input(2);

        // State prediction using motion model (constant velocity model)
        state_(0) += state_(2) * dt;  // x = x + vel_x * dt
        state_(1) += state_(3) * dt;  // y = y + vel_y * dt
        state_(2) += acc_x * dt;      // vel_x = vel_x + acc_x * dt
        state_(3) += acc_y * dt;      // vel_y = vel_y + acc_y * dt
        state_(4) += yaw_rate * dt;   // yaw = yaw + yaw_rate * dt
        state_(5) = yaw_rate;         // yaw_rate = yaw_rate (constant)

        // Update covariance (Process noise)
        P_ += Q_;
    }

    // Update Step (Measurement update using position data)
    void update(const VectorXd& measurement) {
        // Measurement matrix (maps state to measurement)
        MatrixXd H = MatrixXd(MEASUREMENT_DIM, STATE_DIM);
        H << 1, 0, 0, 0, 0, 0,   // x measurement
             0, 1, 0, 0, 0, 0;   // y measurement

        // Innovation (measurement residual)
        VectorXd z = measurement - H * state_;

        // Calculate innovation covariance
        MatrixXd S = H * P_ * H.transpose() + R_;  // Innovation covariance
        MatrixXd K = P_ * H.transpose() * S.inverse();  // Kalman gain

        // Update state with measurement
        state_ += K * z;

        // Update covariance
        P_ = (MatrixXd::Identity(STATE_DIM, STATE_DIM) - K * H) * P_;
    }

    // Getter for state vector
    VectorXd getState() {
        return state_;
    }
};

// Simulate the process and perform UKF updates
int main() {
    // Create UKF object
    UKF ukf;

    // Simulated control inputs (acceleration_x, acceleration_y, yaw_rate)
    VectorXd control_input(3);

    // Simulate a circular path for 10 seconds
    double radius = 10.0;  // Radius of the circle
    double angular_velocity = 0.1;  // Angular velocity (rad/s)

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> gps_noise(0, 0.5); // Simulated GPS noise in meters
    normal_distribution<> imu_noise(0, 0.1); // Simulated IMU noise (velocity, yaw rate)

    // Loop over time and simulate data
    for (int i = 0; i < 100; ++i) {
        // Update control input (acceleration and yaw rate)
        control_input << 0, 0, angular_velocity;

        // Predict the next state based on the control input
        ukf.predict(control_input);

        // Simulate the circular path
        double angle = angular_velocity * i * dt;
        double x_actual = radius * cos(angle);
        double y_actual = radius * sin(angle);

        // Simulate noisy GPS measurements
        VectorXd gps_measurement(2);
        gps_measurement << x_actual + gps_noise(gen), y_actual + gps_noise(gen);

        // Update UKF with noisy GPS measurement
        ukf.update(gps_measurement);

        // Get predicted state and print it
        VectorXd predicted_state = ukf.getState();
        cout << "Actual Coordinates (Simulated Path): " << endl;
        cout << "x: " << x_actual << " y: " << y_actual << endl;
        cout << "Predicted Coordinates from UKF: " << endl;
        cout << "x: " << predicted_state(0) << " y: " << predicted_state(1) << endl;
    }

    return 0;
}

