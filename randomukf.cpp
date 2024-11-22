#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;

const int STATE_DIM = 6; // [x, y, vel, yaw, yaw_rate, accn]
const int MEASUREMENT_DIM = 3; // [x, y, yaw]
double dt = 0.05; // time step in seconds
double prev_v = 0.0;
double prev_yaw = 0.0;
double alpha = 0.01;
double beta = 2;
double kappa = 0;
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

// Noise variables
std::default_random_engine generator;
std::normal_distribution<double> gps_noise(0.0, 0.1);  // Gaussian noise for GPS
std::normal_distribution<double> imu_noise(0.0, 0.01); // Gaussian noise for IMU

double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// Initial state and covariance
VectorXd state_ = VectorXd::Zero(STATE_DIM);
MatrixXd P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);
MatrixXd Q_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01; // process noise covariance

// Process and measurement noise
MatrixXd Q = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.03; // process noise covariance
MatrixXd R = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.08; // measurement noise covariance

void generateSigmaPoints(const VectorXd& state, const MatrixXd& P, MatrixXd& sigma_points) {
    int n_sig = 2 * STATE_DIM + 1;
    sigma_points.col(0) = state;

    MatrixXd A = P.llt().matrixL();
    A *= sqrt(lambda + STATE_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        sigma_points.col(i + 1) = state + A.col(i);
        sigma_points.col(i + 1 + STATE_DIM) = state - A.col(i);
    }
}

void predictSigmaPoints(MatrixXd& sigma_points, double dt, double actual_acceleration) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i); // velocity
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);
        double acc = sigma_points(5, i); // acceleration

        double v_new = v + actual_acceleration * dt;

        // Predict position and yaw
        if (fabs(yaw_rate) > 1e-5) {
            sigma_points(0, i) += v / yaw_rate * (sin(yaw + yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += v / yaw_rate * (-cos(yaw + yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += v * cos(yaw) * dt;
            sigma_points(1, i) += v * sin(yaw) * dt;
        }

        sigma_points(2, i) = v_new;
        sigma_points(3, i) = normalizeAngle(yaw + yaw_rate * dt);
        sigma_points(5, i) = actual_acceleration;
    }
}

VectorXd predictMeanAndCovariance(MatrixXd& sigma_points, MatrixXd& P_pred) {
    VectorXd weights = VectorXd(2 * STATE_DIM + 1);
    weights(0) = lambda / (lambda + STATE_DIM);

    for (int i = 1; i < 2 * STATE_DIM + 1; ++i) {
        weights(i) = 1 / (2 * (lambda + STATE_DIM));
    }

    VectorXd x_pred = VectorXd::Zero(STATE_DIM);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        x_pred += weights(i) * sigma_points.col(i);
    }

    P_pred = MatrixXd::Zero(STATE_DIM, STATE_DIM);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        VectorXd diff = sigma_points.col(i) - x_pred;
        P_pred += weights(i) * diff * diff.transpose();
    }
    P_pred += Q; // Add process noise

    return x_pred;
}

void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z) {
    // Measurement matrix
    MatrixXd H = MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    H(0, 0) = 1; // x
    H(1, 1) = 1; // y
    H(2, 3) = 1; // yaw

    VectorXd y = z - H * state_pred; // Innovation

    // Normalize yaw in the innovation (angle wrap to [-pi, pi])
    while (y(2) > M_PI) y(2) -= 2 * M_PI;
    while (y(2) < -M_PI) y(2) += 2 * M_PI;

    MatrixXd S = H * P_pred * H.transpose() + R; // Innovation covariance
    MatrixXd K = P_pred * H.transpose() * S.inverse(); // Kalman gain

    state_pred += K * y; // Update state estimate
    P_pred -= K * H * P_pred; // Update covariance
}

int main() {
    state_ = VectorXd(6);
    state_ << 10, 0, 0.5, 0, 0, 0; // Initial state: x, y, velocity, yaw, yaw_rate
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);
    P_ = MatrixXd::Identity(6,6);
    Q_ = MatrixXd::Zero(6,6);
    Q_(2, 2) = 0.01; // velocity noise
    Q_(4, 4) = 0.01; // yaw rate noise
    Q_(5, 5) = 0.01; // acceleration noise

    cout << "Initial state: " << state_.transpose() << endl;

    for (int step = 1; step <= 100; ++step) {
        // Simulate a path with random variations
        double actual_x = 10 + cos(0.1 * step) * step + (rand() % 10 - 5); // Random path deviations
        double actual_y = sin(0.1 * step) * step + (rand() % 10 - 5); // Random path deviations
        double actual_yaw = 0.1 * step + (rand() % 10 - 5) * 0.01;  // Random yaw deviations

        // Simulate varying accelerations
        double actual_acceleration = 0.05 * sin(0.05 * step) + (rand() % 3 - 1) * 0.01; // Random acceleration

        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, dt, actual_acceleration);

        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

        // Simulate a noisy GPS measurement
        VectorXd gps = VectorXd::Zero(3); // GPS simulated measurement
        gps(0) = actual_x + gps_noise(generator);    // GPS X with noise
        gps(1) = actual_y + gps_noise(generator);    // GPS Y with noise
        gps(2) = actual_yaw + gps_noise(generator) * 0.01; // GPS Yaw with noise

        // Simulate noisy IMU (Accelerometer and Gyroscope)
        double imu_ax = (state_pred(2) - prev_v) / dt + imu_noise(generator); // Linear acceleration with noise
        double imu_yaw_rate = (state_pred(3) - prev_yaw) / dt + imu_noise(generator); // Yaw rate with noise
        prev_v = state_pred(2);
        prev_yaw = state_pred(3);

        // Update state with the measurement
        updateStateWithMeasurement(state_pred, P_pred, gps);
 

         // Print the actual path along with the predicted state
        cout << "Step " << step << ": ";
        cout << "Actual Path (X: " << actual_x << ", Y: " << actual_y << ", Yaw: " << actual_yaw << ") ";
        cout << "Predicted State (X: " << state_pred(0) << ", Y: " << state_pred(1) << ", Yaw: " << state_pred(3) << ")" << endl;
    }
    return 0;
}

