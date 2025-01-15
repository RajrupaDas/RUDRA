#ifndef UKF_LOCALIZATION_H
#define UKF_LOCALIZATION_H

#include <Eigen/Dense>
#include <fstream>

using namespace Eigen;

// Constants
const int STATE_DIM = 5; // [x, y, vel, yaw, yaw_rate]
const int MEASUREMENT_DIM = 3; // [x, y, yaw]
extern double dt;
extern double alpha;
extern double beta;
extern double kappa;
extern double lambda;

// Function Declarations
double normalizeAngle(double angle);
bool readCSVData(std::ifstream &file, double &gps_x, double &gps_y, double &gps_yaw,
                 double &imu_ax, double &imu_yaw_rate, double &additional_ax, double &additional_yaw_rate);
void generateSigmaPoints(const VectorXd& state, const MatrixXd& P, MatrixXd& sigma_points);
void predictSigmaPoints(MatrixXd& sigma_points, double dt, double imu_ax, double imu_yaw_rate,
                        double additional_ax, double additional_yaw_rate);
VectorXd predictMeanAndCovariance(MatrixXd& sigma_points, MatrixXd& P_pred);
void updateStateWithMeasurement(VectorXd& state_pred, MatrixXd& P_pred, const VectorXd& z);
Vector3d locate();

#endif // UKF_LOCALIZATION_H

