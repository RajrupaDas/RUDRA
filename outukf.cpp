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
        P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.5;  // Initial state covariance matrix (adjusted)
        Q_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01;  // Process noise covariance matrix (adjusted)
        R_ = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.5;  // Measurement noise covariance matrix (adjusted)

        // Initial state (on the circular path)
        state_ << 10, 0, 0, 0, 0, 0;  // initial state [x, y, vel_x, vel_y, yaw, yaw_rate]
    }

    // Prediction Step
    void predict(const VectorXd& control_input) {
        // Control input (acceleration_x, acceleration_y, yaw_rate)
        double acc_x = control_input(0);
        double acc_y = control_input(1);
        double yaw_rate = control_input(2);

        // Update state using the correct motion model for circular motion
        double vel = sqrt(state_(2)*state_(2) + state_(3)*state_(3)); // speed (magnitude of velocity)

        // Position update (considering velocity and yaw)
        state_(0) += vel * cos(state_(4)) * dt;  // x = x + vel * cos(yaw) * dt
        state_(1) += vel * sin(state_(4)) * dt;  // y = y + vel * sin(yaw) * dt
        state_(2) += acc_x * dt;      // vel_x = vel_x + acc_x * dt
        state_(3) += acc_y * dt;      // vel_y = vel_y + acc_y * dt
        state_(4) += yaw_rate * dt;   // yaw = yaw + yaw_rate * dt
        state_(5) = yaw_rate;         // yaw_rate = yaw_rate (constant)

        // Update covariance (Process noise)
        P_ += Q_;
    }

    /* Update Step (Measurement update using position data)
    void update(const VectorXd& gps_measurements, const VectorXd& imu_measurement) {
        // Measurement matrix (maps state to measurement)
        MatrixXd H_gps = MatrixXd(MEASUREMENT_DIM, STATE_DIM);
        H_gps << 1, 0, 0, 0, 0, 0,   // x measurement
                 0, 1, 0, 0, 0, 0;   // y measurement

        // Innovation (measurement residual)
        VectorXd z_gps = gps_measurement - H_gps * state_;

        // Calculate innovation covariance
        MatrixXd S_gps = H_gps * P_ * H_gps.transpose() + R_;  // Innovation covariance
        MatrixXd K_gps = P_ * H_gps.transpose() * S_gps.inverse();  // Kalman gain

        // Update state with measurement
        state_ += K_gps * z_gps;

        // Update covariance
        P_ = (MatrixXd::Identity(STATE_DIM, STATE_DIM) - K_gps * H_gps) * P_;

	MatrixXd H_imu = MatrixXd(2, STATE_DIM);
	H_imu << 0, 0, 1, 0, 0, 0,
	         0, 0, 0, 1, 0, 0;

	VectorXd z_imu(2);
	z_imu << imu_measurement(0) - state_(2),
	         imu_measurement(1) - state_(3);

	MatrixXd S_imu = H_imu * P_ * H_imu.transpose() + R_;
 	MatrixXd K_imu = P_ * H_imu.transpose() * S_imu.inverse();

	state_ += K_imu * z_imu;

	P_ = (MatrixXd::Identity(STATE_DIM, STATE_DIM) - K_imu * H_imu) * P_;
    }

    // Getter for state vector
    VectorXd getState() {
        return state_;
    }
};*/

void update(const VectorXd& gps_measurement, const VectorXd& imu_measurement) {
    // Measurement matrix for GPS (x, y)
    MatrixXd H_gps = MatrixXd(MEASUREMENT_DIM, STATE_DIM);
    H_gps << 1, 0, 0, 0, 0, 0,   // x measurement
             0, 1, 0, 0, 0, 0;   // y measurement

    // Innovation (measurement residual) for GPS
    VectorXd z_gps = gps_measurement - H_gps * state_;

    // Innovation covariance for GPS
    MatrixXd S_gps = H_gps * P_ * H_gps.transpose() + R_;
    MatrixXd K_gps = P_ * H_gps.transpose() * S_gps.inverse();

    // Update state with GPS measurement
    state_ += K_gps * z_gps;

    // Update covariance with GPS measurement
    P_ = (MatrixXd::Identity(STATE_DIM, STATE_DIM) - K_gps * H_gps) * P_;

    // Measurement matrix for IMU (velocity, yaw rate)
    MatrixXd H_imu(2, STATE_DIM);
    H_imu << 0, 0, 1, 0, 0, 0,   // vel_x measurement
             0, 0, 0, 0, 0, 1;   // yaw_rate measurement

    // Innovation for IMU
    VectorXd z_imu(2);
    z_imu << imu_measurement(0) - state_(2),   // vel_x residual
             imu_measurement(1) - state_(5);   // yaw_rate residual

    // Innovation covariance for IMU
    MatrixXd S_imu = H_imu * P_ * H_imu.transpose() + R_;
    MatrixXd K_imu = P_ * H_imu.transpose() * S_imu.inverse();

    // Update state with IMU measurement
    state_ += K_imu * z_imu;

    // Update covariance with IMU measurement
    P_ = (MatrixXd::Identity(STATE_DIM, STATE_DIM) - K_imu * H_imu) * P_;
  }
}I

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
        control_input << 0, 0, angular_velocity;  // no linear acceleration, just constant yaw

        // Predict the next state based on the control input
        ukf.predict(control_input);

        // Simulate the circular path
        double angle = angular_velocity * i * dt;
        double x_actual = radius * cos(angle);
        double y_actual = radius * sin(angle);

        // Simulate noisy GPS measurements
        VectorXd gps_measurement(2);
        gps_measurement << x_actual + gps_noise(gen), y_actual + gps_noise(gen);

	VectorXd imu_measurement(2);
        double vel_x = -radius * angular_velocity * sin(angle);  // velocity_x = -r * omega * sin(angle)
        double vel_y = radius * angular_velocity * cos(angle);   // velocity_y = r * omega * cos(angle)
        imu_measurement << vel_x + imu_noise(gen), vel_y + imu_noise(gen);

        // Update UKF with noisy GPS measurement
        ukf.update(gps_measurement, imu_measurement);

        // Get predicted state and print it
        VectorXd predicted_state = ukf.getState();
        cout << "Actual Coordinates (Simulated Path): " << endl;
        cout << "x: " << x_actual << " y: " << y_actual << endl;
        cout << "Predicted Coordinates from UKF: " << endl;
        cout << "x: " << predicted_state(0) << " y: " << predicted_state(1) << endl;
    }

    return 0;
}
