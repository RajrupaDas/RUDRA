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

//PREDICTION
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

//MAIN FUNCN
int main() {
    state_ << 10, 0, 0.5, 0.5, 0, 0;
    MatrixXd sigma_points = MatrixXd(STATE_DIM, 2 * STATE_DIM + 1);

    //PRINT VAL
    cout << "Initial actual position: x = 10, y = 0" << endl;
    cout << "Initial UKF state: x = " << state_(0) << ", y = " << state_(1) << endl;

    for (int step = 1; step <= 10; ++step) {
        //SIM OF ACTUAL PTS
        double actual_x = 10 + 0.001 * step * step;
        double actual_y = 0.0001 * step * step;

        generateSigmaPoints(state_, P_, sigma_points);

        predictSigmaPoints(sigma_points, dt);

        MatrixXd P_pred;
        VectorXd state_pred = predictMeanAndCovariance(sigma_points, P_pred);

	//PRINT MORE VAL
        cout << "Step " << step << ":" << endl;
        cout << "Actual position: x = " << actual_x << ", y = " << actual_y << endl;
        cout << "UKF estimated position: x = " << state_pred(0) << ", y = " << state_pred(1) << endl;

        //UPDATE STATE FOR NEXT ITERATION
        state_ = state_pred;
        P_ = P_pred;
    }

    return 0;
}

