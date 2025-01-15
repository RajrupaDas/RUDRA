#include "UKF.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>
#include <cmath>

using namespace std;

// Global Constants
double dt = 0.1;
double alpha = 0.001;
double beta = 2;
double kappa = 0;
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

// Function Definitions
double normalizeAngle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

bool readCSVData(ifstream &file, double &gps_x, double &gps_y, double &gps_yaw,
                 double &imu_ax, double &imu_yaw_rate, double &additional_ax, double &additional_yaw_rate) {
    string line;
    if (getline(file, line)) {
        stringstream ss(line);
        string value;

        getline(ss, value, ','); gps_x = stod(value);
        getline(ss, value, ','); gps_y = stod(value);
        getline(ss, value, ','); gps_yaw = stod(value);
        getline(ss, value, ','); imu_ax = stod(value);
        getline(ss, value, ','); imu_yaw_rate = stod(value);
        getline(ss, value, ','); additional_ax = stod(value);
        getline(ss, value, ','); additional_yaw_rate = stod(value);

        return true;
    }
    return false;
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

void predictSigmaPoints(MatrixXd& sigma_points, double dt, double imu_ax, double imu_yaw_rate,
                        double additional_ax, double additional_yaw_rate) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i);
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);

        double combined_ax = 0.5 * (imu_ax + additional_ax);
        double combined_yaw_rate = 0.5 * (imu_yaw_rate + additional_yaw_rate);

        v += imu_ax * dt;
        if (fabs(yaw_rate) > 1e-5) {
            sigma_points(0, i) += v / combined_yaw_rate * (sin(yaw + combined_yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += v / combined_yaw_rate * (-cos(yaw + combined_yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += v * cos(yaw) * dt;
            sigma_points(1, i) += v * sin(yaw) * dt;
        }
        sigma_points(2, i) = v;
        sigma_points(3, i) = normalizeAngle(yaw + combined_yaw_rate * dt);
        sigma_points(4, i) = combined_yaw_rate;
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
    P_pred += MatrixXd::Identity(STATE_DIM, STATE_DIM) * 1e-3; // Q noise

    return x_pred;
}

void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z) {
    MatrixXd H = MatrixXd::Zero(MEASUREMENT_DIM, STATE_DIM);
    H(0, 0) = 1;
    H(1, 1) = 1;
    H(2, 3) = 1;

    VectorXd y = z - H * state_pred;
    y(2) = normalizeAngle(y(2));

    MatrixXd S = H * P_pred * H.transpose() + MatrixXd::Identity(MEASUREMENT_DIM, MEASUREMENT_DIM) * 1e-2; // R noise
    MatrixXd K = P_pred * H.transpose() * S.inverse();

    state_pred += K * y;
    P_pred -= K * H * P_pred;
}

Vector3d locate() {
    static VectorXd state_ = VectorXd::Zero(STATE_DIM);
    static MatrixXd P_ = MatrixXd::Identity(STATE_DIM, STATE_DIM);
    static ifstream csvFile("simulation_data.csv");
    static bool initialized = false;

    if (!initialized) {
        string temp;
        getline(csvFile, temp);
        state_ << 10, 0, 0.5, 0, 0;
        initialized = true;
    }

    double gps_x, gps_y, gps_yaw, imu_ax, imu_yaw_rate, additional_ax, additional_yaw_rate;
    if (!readCSVData(csvFile, gps_x, gps_y, gps_yaw, imu_ax, imu_yaw_rate, additional_ax, additional_yaw_rate)) {
        throw runtime_error("No more data available in the CSV file.");
    }

    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);
    generateSigmaPoints(state_, P_, sigma_points);
    predictSigmaPoints(sigma_points, dt, imu_ax, imu_yaw_rate, additional_ax, additional_yaw_rate);

    MatrixXd P_pred;
    VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

    VectorXd gps = VectorXd(MEASUREMENT_DIM);
    gps << gps_x, gps_y, gps_yaw;

    updateStateWithMeasurement(state_pred, P_pred, gps);

    state_ = state_pred;
    P_ = P_pred;

    return state_.head(3);
}

