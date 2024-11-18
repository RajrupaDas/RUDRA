#ifndef UKF_HPP
#define UKF_HPP

#include <Eigen/Dense>
#include <iostream>

class UKF {
public:
    UKF(double dt, int state_dim, int meas_dim, double lambda)
        : dt(dt), state_dim(state_dim), meas_dim(meas_dim), lambda(lambda) {
        Xsig_pred = Eigen::MatrixXd(state_dim, 2 * state_dim + 1);
        P = Eigen::MatrixXd(state_dim, state_dim);
        X = Eigen::VectorXd(state_dim);
        Zsig = Eigen::MatrixXd(meas_dim, 2 * state_dim + 1);
        Z = Eigen::VectorXd(meas_dim);
        R = Eigen::MatrixXd(meas_dim, meas_dim);
        Q = Eigen::MatrixXd(state_dim, state_dim);  //process noise cov

        //initialize state and cov
        X.setZero();
        P.setIdentity();
        P *= 0.1;  //less init uncertainty
        R.setIdentity();
        R *= 0.1;  //measu noise
        Q.setIdentity(); //init process noise
        Q *= 0.01;  //small process noise
        Q(9, 9) = 0.001; //larger process noise for gyro bias
        Q(10, 10) = 0.001; 
        Q(11, 11) = 0.001;
    }

 
    void Prediction();
  
    void Update(const Eigen::VectorXd& z);

    // get estimated state and covariance
    Eigen::VectorXd getState() { return X; }
    Eigen::MatrixXd getCovariance() { return P; }

private:
    double dt;  //time step
    int state_dim, meas_dim;
    double lambda;

    Eigen::MatrixXd Xsig_pred;  // Predicted sigma points
    Eigen::VectorXd X;          //state vector [x, y, z, vx, vy, vz, yaw, pitch, roll, yawrate, pitchrate, rollrate]
    Eigen::MatrixXd P;          //state covariance 
    Eigen::MatrixXd Zsig;       //predicted measurement sigma points
    Eigen::VectorXd Z;          //predicted measurement
    Eigen::MatrixXd R;          //measurement noise covariance
    Eigen::MatrixXd Q;          //process noise covariance 

    //augmented sigma points with gyro bias 
    Eigen::MatrixXd AugmentedSigmaPoints();

    //predict sigma points 3D, yaw, pitch, roll
    void PredictSigmaPoints(const Eigen::MatrixXd& Xsig_aug);

    //predict measurement
    void PredictMeasurement();

    Eigen::MatrixXd ComputeCrossCorrelation();
};

#endif  

