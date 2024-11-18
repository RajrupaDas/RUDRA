#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

using namespace Eigen;
using namespace std;

const int STATE_DIM = 6; //[x, y, vel x, vel y, yaw, yaw rate]
const int MEASUREMENT_DIM = 2; //[x, y]
double dt = 0.1; //time step in secs

double alpha = 0.001;
double beta = 2;
double kappa = 0;
double lambda = alpha * alpha * (STATE_DIM + kappa) - STATE_DIM;

//INIT STATE MAT AND COV MAT
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
        double vel_x = sigma_points(2, i);
        double vel_y = sigma_points(3, i);

        //const vel model for prediction
        sigma_points(0, i) += vel_x * dt;
        sigma_points(1, i) += vel_y * dt;
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

    return x_pred;
}

// Measurement update step
void updateStateWithMeasurement(VectorXd& state, MatrixXd& P, const VectorXd& z, const MatrixXd& sigma_points) {
    // Define weights
    VectorXd weights = VectorXd(2 * STATE_DIM + 1);
    weights(0) = lambda / (lambda + STATE_DIM);
    for (int i = 1; i < 2 * STATE_DIM + 1; ++i) {
        weights(i) = 1 / (2 * (lambda + STATE_DIM));
    }

    // Predict measurement mean
    MatrixXd Z_sigma = sigma_points.topRows(MEASUREMENT_DIM);
    VectorXd z_pred = VectorXd::Zero(MEASUREMENT_DIM);
    for (int i = 0; i < Z_sigma.cols(); ++i) {
        z_pred += weights(i) * Z_sigma.col(i);
    }

    // Calculate measurement covariance and cross-correlation matrix
    MatrixXd S = MatrixXd::Zero(MEASUREMENT_DIM, MEASUREMENT_DIM);
    MatrixXd Tc = MatrixXd::Zero(STATE_DIM, MEASUREMENT_DIM);
    for (int i = 0; i < Z_sigma.cols(); ++i) {
        VectorXd z_diff = Z_sigma.col(i) - z_pred;
        S += weights(i) * z_diff * z_diff.transpose();

        VectorXd x_diff = sigma_points.col(i) - state;
        Tc += weights(i) * x_diff * z_diff.transpose();
    }

    // Add measurement noise covariance
    S += R;

    // Calculate Kalman gain
    MatrixXd K = Tc * S.inverse();

    // Update state and covariance
    VectorXd z_diff = z - z_pred;
    state += K * z_diff;
    P -= K * S * K.transpose();
}

int main() {
    // Example measurement input
    VectorXd z = VectorXd(MEASUREMENT_DIM);
    z << 5.5, 3.2; // Replace with actual measurements

    // Generate sigma points
    MatrixXd sigma_points(STATE_DIM, 2 * STATE_DIM + 1);
    generateSigmaPoints(state_, P_, sigma_points);

    // Predict sigma points
    predictSigmaPoints(sigma_points, dt);

    // Predict mean and covariance
    MatrixXd P_pred;
    VectorXd x_pred = predictMeanAndCovariance(sigma_points, P_pred);

    // Update state with measurement
    updateStateWithMeasurement(x_pred, P_pred, z, sigma_points);

    // Print updated state
    cout << "Updated state:\n" << x_pred << endl;
    cout << "Updated covariance:\n" << P_pred << endl;

    return 0;
}

