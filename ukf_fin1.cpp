#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <cmath>

using namespace Eigen;
using namespace std;

const int STATE_DIM = 5; // [x, y, velocity, yaw, yaw_rate]
const int MEASUREMENT_DIM = 2; // [x, y] (no yaw anymore)
double dt = 0.1; // Time step in seconds
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

// Declare process noise covariance (Q) and measurement noise covariance (R) matrices
MatrixXd Q = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.1; // Initial Q matrix (process noise covariance)
MatrixXd R = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.1; // Initial R matrix (measurement noise covariance)

// Process noise covariance (Q) initialization
void initializeNoiseMatrices() {
    Q(0, 0) = 0.1; // x position uncertainty
    Q(1, 1) = 0.1; // y position uncertainty
    Q(2, 2) = 0.05; // velocity uncertainty
    Q(3, 3) = 0.01; // yaw uncertainty
    Q(4, 4) = 0.01; // yaw rate uncertainty

    R(0, 0) = 0.02; // x GPS measurement uncertainty (0.02 meters)
    R(1, 1) = 0.02; // y GPS measurement uncertainty (0.02 meters)
}

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

void predictSigmaPoints(MatrixXd& sigma_points, double dt, double imu_linear_acc, double imu_angular_vel, double linear_vel, double angular_accel) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i); // velocity
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);

        // Update velocity using the provided linear acceleration
        double linear_velocity = v + imu_linear_acc * dt;

        // Update angular velocity using the provided angular acceleration
        double angular_velocity = yaw_rate + angular_accel * dt;

        // Update position using the provided yaw rate and linear velocity
        if (fabs(angular_velocity) > 1e-5) {
            sigma_points(0, i) += linear_velocity / angular_velocity * (sin(yaw + angular_velocity * dt) - sin(yaw));
            sigma_points(1, i) += linear_velocity / angular_velocity * (-cos(yaw + angular_velocity * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += linear_velocity * cos(yaw) * dt;
            sigma_points(1, i) += linear_velocity * sin(yaw) * dt;
        }

        sigma_points(2, i) = linear_velocity;  // Update velocity
        sigma_points(3, i) = normalizeAngle(yaw + angular_velocity * dt); // Update yaw
        sigma_points(4, i) = angular_velocity; // Update yaw rate (angular velocity)
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

    VectorXd y = z - H * state_pred; // Innovation

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

    // Open the CSV file
    ifstream file("simulation_data.csv");
    if (!file.is_open()) {
        cerr << "Could not open the file!" << endl;
        return 1;
    }

    string line;

    getline(file, line); // Skip header

    // Initialize noise matrices
    initializeNoiseMatrices();

    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> values;
        string value;

        while (getline(ss, value, ',')) {
            values.push_back(stod(value));
        }

        if (values.size() < 6) { // Ensure enough data points are present
            cerr << "Invalid data received!" << endl;
            continue;
        }

        // Parse input data
        double gps_x = values[0];
        double gps_y = values[1];
        double imu_linear_acc = values[2]; // IMU linear acceleration
        double imu_angular_vel = values[3]; // IMU angular velocity
        double linear_vel = values[4]; // Linear velocity
        double angular_accel = values[5]; // Angular acceleration

        // Print the current line from the CSV for comparison
        cout << "CSV data: gps_x = " << gps_x << ", gps_y = " << gps_y
             << ", imu_linear_acc = " << imu_linear_acc << ", imu_angular_vel = " << imu_angular_vel
             << ", linear_vel = " << linear_vel << ", angular_accel = " << angular_accel << endl;

        // Simulate measurement (only using gps_x and gps_y now)
        VectorXd gps(2); // Only x, y now
        gps << gps_x, gps_y;

        // Dynamic process noise update
        Q(2, 2) = std::max(0.01, fabs(imu_linear_acc) * 0.1);
        Q(3, 3) = std::max(0.01, fabs(imu_angular_vel) * 0.1);
        Q(4, 4) = std::max(0.01, fabs(imu_angular_vel) * 0.1);

        // Generate and predict sigma points
        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, dt, imu_linear_acc, imu_angular_vel, linear_vel, angular_accel); // Use all 6 inputs here

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

    file.close();
    return 0;
}

