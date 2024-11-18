#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

using Matrix6f = Eigen::Matrix<double, 6, 1>;
using Matrix6by6f = Eigen::Matrix<double, 6, 6>;
using Matrix4f = Eigen::Matrix<double, 4, 1>;
using Matrix4by4f = Eigen::Matrix<double, 4, 4>;
using Matrix6by4f = Eigen::Matrix<double, 6, 4>;

class UKF {
public:
    const double m_dt = 0.01;  //time step
    Matrix6f m_state;          //state vector
    Matrix6by6f m_covariance;  //cov mat

    //for sigma point
    double alpha = 1e-3;
    double beta = 2;
    double kappa = 0;

    //weights
    double lambda;
    std::vector<double> weights_mean;
    std::vector<double> weights_cov;

    UKF() {
        //init state and cov
        m_state.setZero();
        m_covariance.setIdentity();

        //lambda and weights are calculated
        int n = m_state.size();
        lambda = alpha * alpha * (n + kappa) - n;
        weights_mean.resize(2 * n + 1);
        weights_cov.resize(2 * n + 1);
        
        weights_mean[0] = lambda / (n + lambda);
        weights_cov[0] = weights_mean[0] + (1 - alpha * alpha + beta);
        for (int i = 1; i < 2 * n + 1; ++i) {
            weights_mean[i] = weights_cov[i] = 1.0 / (2 * (n + lambda));
        }
    }

    //sigma point gen
    void generateSigmaPoints(const Matrix6f& x, const Matrix6by6f& P, std::vector<Matrix6f>& sigma_points) {
        int n = x.size();
        sigma_points.resize(2 * n + 1);
        sigma_points[0] = x;
        Matrix6by6f A = ((n + lambda) * P).llt().matrixL();
        
        for (int i = 0; i < n; ++i) {
            sigma_points[i + 1] = x + A.col(i);
            sigma_points[i + 1 + n] = x - A.col(i);
        }
    }

    //propagate sigma points through nonlinear process function
    void propagateSigmaPoints(const std::vector<Matrix6f>& sigma_points, const Matrix6f& u, std::vector<Matrix6f>& sigma_points_predicted) {
        sigma_points_predicted.resize(sigma_points.size());
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            sigma_points_predicted[i] = f(sigma_points[i], u);
        }
    }

    //nonlinear process function (eg. for vehicle dynamics)
    Matrix6f f(const Matrix6f& x, const Matrix6f& u) {
        Matrix6f x_next;
        //need to replace with model (eg. from git)
        x_next = x + m_dt * u;
        return x_next;
    }

    //calculate predicted state and cov
    void calculateMeanAndCovariance(const std::vector<Matrix6f>& sigma_points, Matrix6f& x_predicted, Matrix6by6f& P_predicted) {
        x_predicted.setZero();
        P_predicted.setZero();

	for (size_t i= 0; i < sigma_points.size(); ++i) {
            x_predicted += weights_mean[i] * sigma_points[i];
        }

        for (size_t i= 0; i < sigma_points.size(); ++i) {
            Matrix6f diff = sigma_points[i] - x_predicted;
            P_predicted += weights_cov[i] * (diff * diff.transpose());
        }
    }

    //transform sigma points to measurement space
    void transformToMeasurementSpace(const std::vector<Matrix6f>& sigma_points, std::vector<Matrix4f>& sigma_points_z) {
        sigma_points_z.resize(sigma_points.size());
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            sigma_points_z[i] = h(sigma_points[i]);
        }
    }

    //nonlinear measurement model (eg.)
    Matrix4f h(const Matrix6f& x) {
        Matrix4f z;
        //need to be replaced with measurement model(eg.)
        z << x[0], x[1], x[2], x[3];
        return z;
    }

    //update
    void update(const Matrix4f& z) {
        //transform sigma pts to measurement space
        std::vector<Matrix4f> sigma_points_z;
        transformToMeasurementSpace(sigma_points, sigma_points_z);

        //predicted measurement mean calculated
        Matrix4f z_pred;
        z_pred.setZero();
        for (size_t i = 0; i < sigma_points_z.size(); ++i) {
            z_pred += weights_mean[i] * sigma_points_z[i];
        }

        //innov cov s calculated
        Matrix4by4f S;
        S.setZero();
        for (size_t i = 0; i < sigma_points_z.size(); ++i) {
            Matrix4f diff = sigma_points_z[i] - z_pred;
            S += weights_cov[i] * (diff * diff.transpose());
        }

        //cross cov calculated
        Matrix6by4f Tc;
        Tc.setZero();
        for (size_t i = 0; i < sigma_points.size(); ++i) {
            Matrix6f x_diff = sigma_points[i] - m_state;
            Matrix4f z_diff = sigma_points_z[i] - z_pred;
            Tc += weights_cov[i] y(x_diff * z_diff.transpose());
        }

        //kalman gain
        Matrix6by4f K = Tc * S.inverse();

        //update state and cov
        m_state = m_state + K * (z - z_pred);
        m_covariance = m_covariance - K * S * K.transpose();
     }
};
