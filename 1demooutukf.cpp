#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>

using namespace Eigen;
using namespace std;

// Constants
const int STATE_DIM = 8; //[x, y, vel, yaw, yaw_rate, accn, acc_bias, yaw_rate_bias]
const int MEASUREMENT_DIM = 3; //[x, y, yaw]
double dt = 0.01; // time step

// Tuning parameters for UKF
double alpha = 0.05; // Spread
double beta = 2;     // Gaussian distribution assumption
double kappa = 0;    // Secondary scaling parameter
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

// Noise generators
random_device rd;
mt19937 gen(rd());
normal_distribution<> gps_noise(0, 0.05);  // Reduced GPS noise
normal_distribution<> imu_noise(0, 0.01); // IMU noise

// State, covariance, and noise matrices
VectorXd state_ = VectorXd::Zero(STATE_DIM);
MatrixXd P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1; // Better initial covariance
MatrixXd Q_ = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01; // Process noise
MatrixXd R_ = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.05; // Measurement noise

// Normalize angles to [-pi, pi]
double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// Adjust noise matrices dynamically
void adjustNoiseMatrices(const VectorXd& state) {
    double speed = state(2); // Velocity
    double noise_factor = 1.0 + speed / 10.0; // Adjust noise based on speed
    Q_.diagonal().array() = 0.01 * noise_factor;
    R_.diagonal().array() = 0.05 * noise_factor;
}

// Generate sigma points
void generateSigmaPoints(const VectorXd& state, const MatrixXd& P, MatrixXd& sigma_points) {
    sigma_points.col(0) = state;
    MatrixXd A = P.llt().matrixL();
    A *= sqrt(lambda + STATE_DIM);

    for (int i = 0; i < STATE_DIM; ++i) {
        sigma_points.col(i + 1) = state + A.col(i);
        sigma_points.col(i + 1 + STATE_DIM) = state - A.col(i);
    }
}

// Predict sigma points with control inputs
void predictSigmaPoints(MatrixXd& sigma_points, double dt, double throttle, double steering_angle) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i);
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);
        double acc = sigma_points(5, i);
        double acc_bias = sigma_points(6, i);
        double yaw_rate_bias = sigma_points(7, i);

        double adjusted_acc = acc + acc_bias + throttle;
        double adjusted_yaw_rate = yaw_rate + yaw_rate_bias + steering_angle;

        // Predict position
        if (fabs(adjusted_yaw_rate) > 1e-5) {
            sigma_points(0, i) += v / adjusted_yaw_rate * (sin(yaw + adjusted_yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += v / adjusted_yaw_rate * (-cos(yaw + adjusted_yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += v * cos(yaw) * dt;
            sigma_points(1, i) += v * sin(yaw) * dt;
        }

        // Predict other states
        sigma_points(2, i) += adjusted_acc * dt;
        sigma_points(3, i) = normalizeAngle(yaw + adjusted_yaw_rate * dt);
        sigma_points(5, i) = adjusted_acc;
        sigma_points(6, i) += 0.0001; // Simulate slow bias drift
        sigma_points(7, i) += 0.0001; // Simulate slow bias drift
    }
}

// Predict mean and covariance
VectorXd predictMeanAndCovariance(MatrixXd& sigma_points, MatrixXd& P_pred) {
    VectorXd weights = VectorXd(2 * STATE_DIM + 1);
    weights(0) = lambda / (lambda + STATE_DIM);
    for (int i = 1; i < weights.size(); ++i) {
        weights(i) = 1 / (2 * (lambda + STATE_DIM));
    }

    VectorXd x_pred = VectorXd::Zero(STATE_DIM);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        x_pred += weights(i) * sigma_points.col(i);
    }

    P_pred = MatrixXd::Zero(STATE_DIM, STATE_DIM);
    for (int i = 0; i < sigma_points.cols(); ++i) {
        VectorXd diff = sigma_points.col(i) - x_pred;
        diff(3) = normalizeAngle(diff(3)); // Normalize yaw
        P_pred += weights(i) * diff * diff.transpose();
    }
    P_pred += Q_; // Add process noise
    return x_pred;
}

// Update state with measurement
void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z) {
    MatrixXd H = MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    H(0, 0) = 1;
    H(1, 1) = 1;
    H(2, 3) = 1;

    VectorXd y = z - H * state_pred;
    y(2) = normalizeAngle(y(2)); // Normalize yaw

    MatrixXd S = H * P_pred * H.transpose() + R_;
    MatrixXd K = P_pred * H.transpose() * S.inverse();

    state_pred += K * y;
    P_pred -= K * H * P_pred;
}

// Main function
int main() {
    state_ << 0, 0, 1, 0, 0, 0, 0, 0; // Initial state
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);

    for (int step = 1; step <= 10; ++step) {
        double actual_x = 10 + cos(0.1 * step) * step;
        double actual_y = sin(0.1 * step) * step;
        double actual_yaw = 0.1 * step;
        double throttle = 0.2;
        double steering_angle = 0.02;

        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, dt, throttle, steering_angle);

        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

        VectorXd gps = VectorXd::Zero(3);
        gps(0) = actual_x + gps_noise(gen);
        gps(1) = actual_y + gps_noise(gen);
        gps(2) = actual_yaw + imu_noise(gen);

        adjustNoiseMatrices(state_pred); // Adjust noise dynamically
        updateStateWithMeasurement(state_pred, P_pred, gps);

        state_ = state_pred;
        P_ = P_pred;

        cout << "Step " << step << ": Actual [x, y, yaw] = [" << actual_x << ", " << actual_y << ", " << actual_yaw << "]" << endl;
        cout << "Estimated [x, y, yaw] = [" << state_pred(0) << ", " << state_pred(1) << ", " << state_pred(3) << "]" << endl;
    }

    return 0;
}

