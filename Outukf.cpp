#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace Eigen;
using namespace std;

const int STATE_DIM = 5; //[x, y, vel, yaw, yaw_rate]
const int MEASUREMENT_DIM = 3; //[x, y, yaw]
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

void predictSigmaPoints(MatrixXd& sigma_points, double dt) {
    for (int i = 0; i < sigma_points.cols(); ++i) {
        double px = sigma_points(0, i);
        double py = sigma_points(1, i);
        double v = sigma_points(2, i); // velocity
        double yaw = sigma_points(3, i);
        double yaw_rate = sigma_points(4, i);

        // Predict position and yaw
        if (fabs(yaw_rate) > 1e-5) {
            sigma_points(0, i) += v / yaw_rate * (sin(yaw + yaw_rate * dt) - sin(yaw));
            sigma_points(1, i) += v / yaw_rate * (-cos(yaw + yaw_rate * dt) + cos(yaw));
        } else {
            sigma_points(0, i) += v * cos(yaw) * dt;
            sigma_points(1, i) += v * sin(yaw) * dt;
        }

        sigma_points(3, i) += yaw_rate * dt; // Update yaw
        // Velocities and yaw_rate stay constant in this model
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
    state_ << 10, 0, 0.5, 0, 0; // Initial state: x, y, velocity, yaw, yaw_rate
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);

    cout << "Initial actual position: x = 10, y = 0" << endl;
    cout << "Initial UKF state: x = " << state_(0) << ", y = " << state_(1) << endl;

    for (int step = 1; step <= 10; ++step) {
        // Simulate actual points
        double actual_x = 10 + 0.5 * step;
        double actual_y = 0.2 * step;
        double actual_yaw = 0.05 * step;

        generateSigmaPoints(state_, P_, sigma_points);
        predictSigmaPoints(sigma_points, dt);

        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

        // Simulate a measurement with noise
        VectorXd z = VectorXd::Zero(MEASUREMENT_DIM);
        z << actual_x + 0.1 * ((rand() % 100) / 100.0),
             actual_y + 0.1 * ((rand() % 100) / 100.0),
             actual_yaw + 0.01 * ((rand() % 100) / 100.0);

        // Perform measurement update
        updateStateWithMeasurement(state_pred, P_pred, z);

        cout << "Step " << step << ":" << endl;
        cout << "Actual position: x = " << actual_x << ", y = " << actual_y << ", yaw = " << actual_yaw << endl;
        cout << "UKF estimated position: x = " << state_pred(0)
             << ", y = " << state_pred(1)
             << ", yaw = " << state_pred(3) << endl;

        // Update state and covariance for next iteration
        state_ = state_pred;
        P_ = P_pred;
    }

    return 0;
}

