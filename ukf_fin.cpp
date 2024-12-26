#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <thread> // For multithreading (to simulate GPS/IMU update timing)

using namespace Eigen;
using namespace std;

const int STATE_DIM = 5; // [x, y, vel, yaw, yaw_rate]
const int MEASUREMENT_DIM = 3; // [x, y, yaw]
double dt = 0.1; // time step in seconds
double alpha = 0.001;
double beta = 2;
double kappa = 0;
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

// Initial state and covariance
VectorXd state_ = VectorXd::Zero(STATE_DIM);
MatrixXd P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);

// Process and measurement noise
MatrixXd Q = MatrixXd::Identity(STATE_DIM, STATE_DIM) * 0.01; // process noise covariance
MatrixXd R = MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 0.1; // measurement noise covariance

// Function to normalize angles
double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

// Function to read CSV data
void readCSVData(ifstream &file, double &gps_x, double &gps_y, double &gps_yaw, double &imu_ax, double &imu_yaw_rate) {
    string line;
    if (getline(file, line)) { // Read one line at a time
        stringstream ss(line);
        string value;

        // Read the GPS values (gps_x, gps_y, gps_yaw)
        getline(ss, value, ','); gps_x = stod(value);
        getline(ss, value, ','); gps_y = stod(value);
        getline(ss, value, ','); gps_yaw = stod(value);

        // Read the IMU values (imu_ax, imu_yaw_rate)
        getline(ss, value, ','); imu_ax = stod(value);
        getline(ss, value, ','); imu_yaw_rate = stod(value);

        // Print the raw CSV data
        cout << "CSV Data: GPS(x: " << gps_x << ", y: " << gps_y << ", yaw: " << gps_yaw
             << "), IMU(ax: " << imu_ax << ", yaw_rate: " << imu_yaw_rate << ")" << endl;
    }
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

void predictSigmaPoints(MatrixXd& sigma_points, double dt, double imu_ax, double imu_yaw_rate) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i);
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);

        v += imu_ax * dt;
        if (fabs(yaw_rate) > 1e-5) {
            sigma_points(0, i) += v / yaw_rate * (sin(yaw + yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += v / yaw_rate * (-cos(yaw + yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += v * cos(yaw) * dt;
            sigma_points(1, i) += v * sin(yaw) * dt;
        }
        sigma_points(2, i) = v;
        sigma_points(3, i) = normalizeAngle(yaw + imu_yaw_rate * dt);
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
    P_pred += Q;

    return x_pred;
}

void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z) {
    MatrixXd H = MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    H(0, 0) = 1; // x
    H(1, 1) = 1; // y
    H(2, 3) = 1; // yaw

    VectorXd y = z - H * state_pred; // Innovation
    y(2) = normalizeAngle(y(2)); // Normalize yaw in innovation

    MatrixXd S = H * P_pred * H.transpose() + R;
    MatrixXd K = P_pred * H.transpose() * S.inverse();

    state_pred += K * y;
    P_pred -= K * H * P_pred;

    state_ = state_pred;
}

int main() {
    state_ << 10, 0, 0.5, 0, 0; // Initial state: x, y, velocity, yaw, yaw_rate
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);
    P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);

    cout << "Initial actual position: x = 10, y = 0" << endl;
    cout << "Initial UKF state: x = " << state_(0) << ", y = " << state_(1) << endl;

    ifstream csvFile("simulation_data.csv");
    std::string line;
    getline(csvFile, line); // Skip header

    int step = 1;
    while (!csvFile.eof()) {
        double gps_x, gps_y, gps_yaw, imu_ax, imu_yaw_rate;

        readCSVData(csvFile, gps_x, gps_y, gps_yaw, imu_ax, imu_yaw_rate);

        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, 0.1, imu_ax, imu_yaw_rate);

        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

        VectorXd gps = VectorXd(MEASUREMENT_DIM);
        gps << gps_x, gps_y, gps_yaw;

        updateStateWithMeasurement(state_pred, P_pred, gps);

        cout << "Step " << step++ << ": Predicted state: x = " << state_(0)
             << ", y = " << state_(1) << ", yaw = " << state_(3) << endl;

        this_thread::sleep_for(chrono::milliseconds(100));
    }

    csvFile.close();
    return 0;
}

