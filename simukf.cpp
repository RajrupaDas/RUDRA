#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <random>
#include "H_ukf.hpp"
#include <numeric>
#include <math.h>

int main() {
    int simulation_time = 40;
    float dt = 1;
    int steps = simulation_time / dt;
    std::vector<float> tspan(steps);
    for (int i = 0; i < steps; i++) tspan[i] = i * dt;

    int gain = 3;
    std::vector<float> Uvx(steps-1);
    std::vector<float> Uvy(steps-1);
    std::vector<float> Uvtheta(steps-1);

    for (int i = 1; i < steps; i++) {
        Uvx[i] = gain * cos(tspan[i]);
        Uvy[i] = gain * -sin(tspan[i]);
        Uvtheta[i] = gain * pow(1 / cos(tspan[i]), 2);
    }

    //init state
    Matrix6f initial_state{0, 1, 1, 0, 0, 1}; //adjustable
    Eigen::Matrix<float, 6, Eigen::Dynamic> true_states(6, steps);
    true_states.col(0) = initial_state;
    
    UKF ukf; 
    
    //init ukf state and cov
    ukf.m_state = initial_state;
    ukf.m_covariance.setIdentity(); //id mat for init cov

    Eigen::Matrix<float, 6, Eigen::Dynamic> estimates(6, steps); //store estimates
    estimates.col(0) = initial_state;

    // Define measurement matrix H (assuming it's a simple identity matrix or custom)
    Eigen::Matrix<float, 4, 6> H;
    H.setIdentity();  //adjustable

    //noise gen
    Eigen::LLT<Eigen::MatrixXf> lltOfR(ukf.m_const_R);  // Assuming m_const_R is your noise covariance matrix
    Eigen::MatrixXf L = lltOfR.matrixL(); // Lower triangular matrix from Cholesky decomposition

    // Gen random noises
    std::default_random_engine generator(2024);
    std::normal_distribution<float> distribution(0.0, 1.0);  //Std normal distributions

    Eigen::Matrix<float, 4, Eigen::Dynamic> randn(4, steps);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < steps; ++j) {
            randn(i, j) = distribution(generator);
        }
    }

    //mat for noisy measurements
    Eigen::MatrixXf Measurements(4, steps);  //4 elements in measurement vector
    for (int i = 0; i < steps; ++i) {
        //noisy measurements for true state at each step
        Measurements.col(i) = H * true_states.col(i) + L * randn.col(i);
    }

    //sim loop from ekf_sim
    for (int i = 1; i < steps; i++) {
        Matrix6f u{0, Uvx[i], 0, Uvy[i], 0, Uvtheta[i]};
        true_states.col(i) = ukf.f(true_states.col(i-1), u);  //prop true state through process model         
        
        ukf.predict(u);

        Eigen::Matrix4f z = Measurements.col(i);  //gen noisy measurements used here
						  
        ukf.update(z);
        
        estimates.col(i) = ukf.m_state;
    }

    //output saved in files
    std::ofstream estimatef("estimates.txt"), truef("trues.txt");
    for (int i = 1; i < steps; i++) {
        estimatef << estimates(0, i) << " " << estimates(2, i) << '\n';
        truef << true_states(0, i) << " " << true_states(2, i) << '\n';
    }

    return 0;
}

