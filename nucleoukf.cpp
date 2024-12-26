#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <iomanip>
#include <sstream> // For parsing serial data

using namespace Eigen;
using namespace std;

const int STATE_DIM = 5; // [x, y, vel, yaw, yaw_rate]
const int MEASUREMENT_DIM = 3; // [x, y, yaw]
double dt = 0.1; // time step in seconds
double prev_v = 0.0;
double prev_yaw = 0.0;
double alpha = 0.001;
double beta = 2;
double kappa = 0;
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

// Normalizing angles to [-pi, pi]
double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// Initial state and covariance
VectorXd state_ = VectorXd::Zero(STATE_DIM);
MatrixXd P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);

// Process and measurement noise
MatrixXd Q = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01; // Process noise covariance
MatrixXd R = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.1; // Measurement noise covariance

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

void predictSigmaPoints(MatrixXd& sigma_points, double dt, double imu_ax, double imu_yaw_rate, double left_velocity, double right_velocity) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i); // velocity
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);

        // Update velocity using left and right motor velocities
        double linear_velocity = (left_velocity + right_velocity) / 2.0;
        yaw_rate = (right_velocity - left_velocity) / 0.5; // Assuming a differential drive robot with 0.5m axle width

        if (fabs(yaw_rate) > 1e-5) {
            sigma_points(0, i) += linear_velocity / yaw_rate * (sin(yaw + yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += linear_velocity / yaw_rate * (-cos(yaw + yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += linear_velocity * cos(yaw) * dt;
            sigma_points(1, i) += linear_velocity * sin(yaw) * dt;
        }

        sigma_points(2, i) = linear_velocity + imu_ax * dt;
        sigma_points(3, i) = normalizeAngle(yaw + yaw_rate * dt);
        sigma_points(4, i) = yaw_rate; // Update yaw_rate based on motor velocities
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
        diff(3) = normalizeAngle(diff(3)); // Normalize yaw
        P_pred += weights(i) * diff * diff.transpose();
    }
    P_pred += Q; // Add process noise

    return x_pred;
}

void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z) {
    MatrixXd H = MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    H(0, 0) = 1; // x
    H(1, 1) = 1; // y
    H(2, 3) = 1; // yaw

    VectorXd y = z - H * state_pred; // Innovation
    y(2) = normalizeAngle(y(2)); // Normalize yaw in innovation

    MatrixXd S = H * P_pred * H.transpose() + R; // Innovation covariance
    MatrixXd K = P_pred * H.transpose() * S.inverse(); // Kalman gain

    state_pred += K * y; // Update state estimate
    P_pred -= K * H * P_pred; // Update covariance
}

int main() {
    state_ << 10, 0, 0.5, 0, 0; // Initial state: x, y, velocity, yaw, yaw_rate
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);

    cout << fixed << setprecision(4);
    cout << "Initial actual position: x = 10, y = 0" << endl;
    cout << "Initial UKF state: x = " << state_(0) << ", y = " << state_(1) << endl;

    while (true) {
        string line;
        getline(cin, line); // Read CSV line from serial input

        if (line.empty()) continue;

        stringstream ss(line);
        vector<double> values;
        string value;

        while (getline(ss, value, ',')) {
            values.push_back(stod(value));
        }

        if (values.size() < 13) { // Ensure enough data points are present
            cerr << "Invalid data received!" << endl;
            continue;
        }

        // Parse input data
        double gps_x = values[0];
        double gps_y = values[1];
        double gps_yaw = values[2];
        double imu_ax = values[3];
        double imu_yaw_rate = values[9]; // Optional, may not be used
        double speedLeft_mps = values[11]; // Left motor speed
        double speedRight_mps = values[12]; // Right motor speed

        // Simulate measurement
        VectorXd gps(3);
        gps << gps_x, gps_y, gps_yaw;

        // Dynamic process noise update
        Q(2, 2) = std::max(0.01, fabs(imu_ax) * 0.1);
        Q(3, 3) = std::max(0.01, fabs(imu_yaw_rate) * 0.1);
        Q(4, 4) = std::max(0.01, fabs(imu_yaw_rate) * 0.1);

        // Generate and predict sigma points
        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, dt, imu_ax, imu_yaw_rate, speedLeft_mps, speedRight_mps);

        // Predict mean and covariance
        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

        // Update with measurement
        updateStateWithMeasurement(state_pred, P_pred, gps);

        // Print results
        cout << "Estimated position: x = " << state_pred(0)
             << ", y = " << state_pred(1)
             << ", yaw = " << state_pred(3) << endl;

        state_ = state_pred;
        P_ = P_pred;
    }

    return 0;
}

