#include <Eigen/Dense>
#include <ctime>
#include <iostream>
#include <cmath>
#include <random>

// UKF parameters
const int n_x = 6;   // State dimension
const int n_aug = 8;  // Augmented state dimension
const double lambda = 3 - n_x;  // Sigma point spreading parameter

// Initial state vector (x, y, vel_x, vel_y, yaw, yaw rate)
Eigen::VectorXd x(6);

// Initial covariance matrix P
Eigen::MatrixXd P(6, 6);

// Process noise and measurement noise covariance matrices
Eigen::MatrixXd Q(6, 6);
Eigen::MatrixXd R(2, 2);  // Measurement noise (for GPS)

// Random number generator for sensor noise
std::default_random_engine generator;
std::normal_distribution<double> gps_noise(0, 0.5);  // GPS noise (mean = 0, stddev = 0.5)

void initializeUKF() {
    // Initial state (x, y, vel_x, vel_y, yaw, yaw rate)
    x << 0, 0, 1, 0, 0, 0.1;

    // Initial covariance matrix (P)
    P << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 0.5, 0, 0, 0,
         0, 0, 0, 0.5, 0, 0,
         0, 0, 0, 0, 0.1, 0,
         0, 0, 0, 0, 0, 0.01;

    // Process noise covariance matrix (Q)
    Q << 0.1, 0, 0, 0, 0, 0,
         0, 0.1, 0, 0, 0, 0,
         0, 0, 0.1, 0, 0, 0,
         0, 0, 0, 0.1, 0, 0,
         0, 0, 0, 0, 0.05, 0,
         0, 0, 0, 0, 0, 0.001;

    // Measurement noise covariance matrix (R) for GPS
    R << 0.5, 0,
         0, 0.5;  // GPS noise covariance (position uncertainty)
}

// Generate sigma points
void generateSigmaPoints(Eigen::MatrixXd& Xsig_aug) {
    Eigen::MatrixXd A = P.llt().matrixL();  // Cholesky decomposition
    Xsig_aug.col(0) = x;  // first sigma point is the state itself

    for (int i = 0; i < n_x; ++i) {
        Xsig_aug.col(i + 1) = x + sqrt(lambda + n_x) * A.col(i);
        Xsig_aug.col(i + 1 + n_x) = x - sqrt(lambda + n_x) * A.col(i);
    }
}

// Process model to predict the next state
void predictState(Eigen::MatrixXd& Xsig_aug, Eigen::MatrixXd& Xsig_pred) {
    for (int i = 0; i < 2 * n_aug + 1; ++i) {
        double delta_t = 1.0;  // Time step, can be adjusted

        // Extract state variables
        double px = Xsig_aug(0, i);
        double py = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(4, i);
        double yaw_rate = Xsig_aug(5, i);

        // Predict the state based on a simple kinematic model
        double px_pred = px + v * cos(yaw) * delta_t;
        double py_pred = py + v * sin(yaw) * delta_t;
        double v_pred = v;
        double yaw_pred = yaw + yaw_rate * delta_t;
        double yaw_rate_pred = yaw_rate;

        // Store the predicted sigma points
        Xsig_pred(0, i) = px_pred;
        Xsig_pred(1, i) = py_pred;
        Xsig_pred(2, i) = v_pred;
        Xsig_pred(4, i) = yaw_pred;
        Xsig_pred(5, i) = yaw_rate_pred;
    }
}

// Measurement update (for GPS: only position x, y)
void updateMeasurement(const Eigen::VectorXd& z, Eigen::MatrixXd& Xsig_pred, Eigen::VectorXd& x, Eigen::MatrixXd& P) {
    Eigen::MatrixXd Zsig(2, 2 * n_aug + 1);  // Transformed measurement sigma points

    for (int i = 0; i < 2 * n_aug + 1; ++i) {
        Zsig(0, i) = Xsig_pred(0, i);  // x position
        Zsig(1, i) = Xsig_pred(1, i);  // y position
    }

    Eigen::VectorXd z_pred = Zsig.rowwise().mean();  // Predicted measurement
    Eigen::MatrixXd S = R + Zsig * Zsig.transpose();  // Measurement covariance

    // Kalman gain
    Eigen::MatrixXd K = P * Zsig.transpose() * S.inverse();

    // Update the state vector
    x = x + K * (z - z_pred);
    P = P - K * S * K.transpose();
}

// Simulate noisy sensor data (IMU + GPS)
Eigen::VectorXd simulateSensors() {
    Eigen::VectorXd sensor_data(2);

    // Simulate actual position (circular path)
    double actual_x = 10.0 * cos(0.1 * std::time(0));  // Circular motion simulation
    double actual_y = 10.0 * sin(0.1 * std::time(0));

    // Add noise (e.g., GPS)
    sensor_data << actual_x + gps_noise(generator), actual_y + gps_noise(generator);

    return sensor_data;
}

int main() {
    initializeUKF();

    // State for the UKF (x, y, vel_x, vel_y, yaw, yaw rate)
    Eigen::VectorXd x_pred(6);

    // Main loop for UKF with sensor data update
    for (int i = 0; i < 100; ++i) {
        // Simulate sensor data (e.g., GPS measurements)
        Eigen::VectorXd z = simulateSensors();

        // Generate sigma points and predict state
        Eigen::MatrixXd Xsig_aug(6, 2 * n_aug + 1);
        generateSigmaPoints(Xsig_aug);

        Eigen::MatrixXd Xsig_pred(6, 2 * n_aug + 1);
        predictState(Xsig_aug, Xsig_pred);

        // Measurement update
        updateMeasurement(z, Xsig_pred, x, P);

        // Print out the predicted state (Position)
        std::cout << "Predicted Position: " << x(0) << ", " << x(1) << std::endl;
    }

    return 0;
}

