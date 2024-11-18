#include <Eigen/Dense>
#include <vector>
using namespace Eigen;
using namespace std;
#define STATE_DIM 6

//INITIALIZATION
Eigen::Matrix<double, STATE_DIM, 1> m_state;        
Eigen::Matrix<double, STATE_DIM, STATE_DIM> m_covariance;
vector<double> Wm, Wc;
Eigen::Matrix<double, STATE_DIM, STATE_DIM> Q;
Eigen::Matrix<double, 2, 2> R;

void initialize(){
   m_state.setZero();
   m_covariance.setIdentity(); 
   m_covariance *= 0.1;
   Q.setIdentity();
   Q*=0.01;
   R.setIdentity();
   R*=0.01;// ADJUST BASED ON SENSOR
   double lambda=3-STATE_DIM;
}
//GENERATION OF SIGMA PTS
void generateSigmaPoints(const Matrix<double, STATE_DIM, 1>& state,
                          const Matrix<double, STATE_DIM, STATE_DIM>& P,
                          vector<Matrix<double, STATE_DIM, 1>>& sigma_points) {
    sigma_points.clear();
    Eigen::Matrix<double, STATE_DIM, 1> mean_state = state;
    Eigen::Matrix<double, STATE_DIM, STATE_DIM> A = P.llt().matrixL(); //cholesky decomposition (p +ve semi definite)

    sigma_points.push_back(mean_state); //first sigma point is mean

//gen of sigma points
    for (int i = 0; i < STATE_DIM; ++i) {
	double scaling_factor=sqrt(STATE_DIM+lambda);
        sigma_points.push_back(mean_state +scaling_factor* A.col(i));
        sigma_points.push_back(mean_state -scaling_factor* A.col(i));
    }
}
//PROCESS FUNCTION
//simple kinematics
Eigen::Matrix<double, STATE_DIM, 1> processModel(const Matrix<double, STATE_DIM, 1>& state, double delta_t) {
	Eigen::Matrix<double, STATE_DIM, 1> predicted_state = state;

    //updating pos based on vel 
    double v=state(3); //forward velocity
    double yaw= state(2);//yaw angle (heading)
    
    //updating x,y with vel and yaw angle
    predicted_state(0)+=v*cos(yaw)*delta_t;
    predicted_state(1)+=v*sin(yaw)*delta_t;

    //update yaw with yaw rate
    predicted_state(2)+=state(4)*delta_t;
    
    //vel const for now
    predicted_state(3)=state(3);

    //yaw rate const for now
    predicted_state(4)=state(4);

    return predicted_state;
}
//PROPAGATION OF SIGMA PTS THROUGH PROCESS MODEL
void predictSigmaPoints(const vector<Eigen::Matrix<double, STATE_DIM, 1>>& sigma_points, 
                        vector<Eigen::Matrix<double, STATE_DIM, 1>>& predicted_sigma_points,
                        double delta_t) {
    int n = sigma_points.size();

    for (int i = 0; i < n; ++i) {
        predicted_sigma_points.push_back(processModel(sigma_points[i], delta_t));
    }
}

//WEIGHTS INITIALIZED
void initializeWeights(int state_dim, double lambda) {
    int num_sigma_points = 2 *state_dim + 1;
    Wm.resize(num_sigma_points);
    Wc.resize(num_sigma_points);

    //calculate weights
    Wm[0] = lambda/(state_dim +lambda);
    Wc[0] =lambda/(state_dim+lambda)+(1-std::pow(0.1,2+2));//ADJUST FOR NOISE AND SPREAD
    // alpha=0.01(for spread)  beta=2  k=0
    for (int i = 1; i < num_sigma_points; ++i) {
        Wm[i] = 1 / (2 *(state_dim +lambda));
        Wc[i] = Wm[i]; //copy wm to wc
    }
}

//PREDICTED MEAN AND COVARIANCE
// Compute the predicted mean and covariance
void computePredictedMeanAndCovariance(const vector<Eigen::Matrix<double, STATE_DIM, 1>>& predicted_sigma_points,
                                        Eigen::Matrix<double, STATE_DIM, 1>& predicted_state,
                                        Eigen::Matrix<double, STATE_DIM, STATE_DIM>& predicted_P, const vector<double>& Wm, const vector<double>& Wc) {
    int n = predicted_sigma_points.size();
    predicted_state.setZero();

    //predicted state (mean)
    for (int i = 0; i < n; ++i) {
        predicted_state += Wm[i] *predicted_sigma_points[i];
    }

    predicted_P.setZero();

    //predicted covarience 
    for (int i = 0; i < n; ++i) {
	    Eigen::Matrix<double, STATE_DIM, 1> diff =predicted_sigma_points[i] -predicted_state;
        predicted_P += Wc[i] *(diff *diff.transpose());
    }
    
    //process noise covarience
    predicted_P += Q;
}
//MAPPING STATE TO MEASUREMENT MODEL
Eigen::Matrix<double, 2, 1> measurementModel(const Matrix<double, 6, 1>& state) {
	Eigen::Matrix<double, 2, 1> measurement;
    
    //2d measurements
    measurement(0) =state(0);//x
    measurement(1) =state(1);//y
    
    return measurement;
}
//MEASUREMENT RESIDUAL (INNOVATION)
Eigen::Matrix<double, 2, 1> computeMeasurementResidual(const Eigen::Matrix<double, 2, 1>& actual_measurement,
                                                const Eigen::Matrix<double, 2, 1>& predicted_measurement) {
	Eigen::Matrix<double, 2, 1> measurement;
    return actual_measurement-predicted_measurement;
}
//K=P.H^T.(H.P.H^T+R)^-1
//P=predictive cov
//H=measurement matrix
//R=measurement noise cov

//KALMAN GAIN
Eigen::Matrix<double, 6, 2> computeKalmanGain(const Eigen::Matrix<double, 6, 6>& predicted_P,
                                      const Eigen::Matrix<double, 2, 2>& R) {
    //mapping from state to measurement
    Eigen::Matrix<double, 2, 6> H =Eigen::Matrix<double, 2, 6>::Zero();
    H(0, 0) = 1.0f;  //x
    H(1, 1) = 1.0f;  //y

    //innovation cov S=H*P* H^T +R
    Eigen::Matrix<double, 2, 2> S=H*predicted_P*H.transpose()+R;

    // Kalman Gain K=P* H^T *S^(-1)
    Eigen::Matrix<double, 6, 2> K=predicted_P*H.transpose()*S.inverse();

    return K;
}

//UPDATE STATE ESTIMATE
void updateStateEstimate(Eigen::Matrix<double, 6, 1>& state,
                         const Eigen::Matrix<double, 6, 2>& K,
                         const Eigen::Matrix<double, 2, 1>& residual) {
    state+ =K*residual;
}

//UPDATE COVARIENCE
void updateCovariance(Eigen::Matrix<double, 6, 6>& P,
                      const Eigen::Matrix<double, 6, 2>& K,
                      const Eigen::Matrix<double, 2, 2>& R) {
    Eigen::MatrixXd H=computeMeasurementMatrix(predicted_state);
    P=(Eigen::Matrix<double, 6, 6>::Identity()-K*H)*P;
}

//MAIN FUNCN
int main() {
    m_state.setZero();
    m_covariance.setIdentity();
    m_covariance *= 0.1;  //init cov
    double lambda = 3 - STATE_DIM;  //turning parameter
    double delta_t = 0.1;  //time step ADJUST FOR TURNING PRECISION
    
    vector<Matrix<double, STATE_DIM, 1>> sigma_points;
    vector<Matrix<double, STATE_DIM, 1>> predicted_sigma_points;
    Matrix<double, STATE_DIM, 1> predicted_state;
    Matrix<double, STATE_DIM, STATE_DIM> predicted_P;
    vector<double> Wm, Wc;

    initializeWeights(STATE_DIM, lambda, WM, Wc);
    
        for (int t = 0; t < 1000; ++t) {         
        generateSigmaPoints(m_state, m_covariance, sigma_points);
        
        predictSigmaPoints(sigma_points, predicted_sigma_points, delta_t);
        
        computePredictedMeanAndCovariance(predicted_sigma_points, predicted_state, predicted_P, Wm, Wc);
        
        Matrix<double, 2, 1> actual_measurement;  // Should come from sensor input
        Matrix<double, 2, 1> predicted_measurement = measurementModel(predicted_state);
        
        Matrix<double, 2, 1> residual = computeMeasurementResidual(actual_measurement, predicted_measurement);
        
        Matrix<double, 6, 2> K = computeKalmanGain(predicted_P, R);
        
        updateStateEstimate(predicted_state, K, residual);
        
        updateCovariance(predicted_P, K, R);
        
        m_state = predicted_state;
        m_covariance = predicted_P;
        
        if (t % 100 == 0) {
            std::cout << "Time step: " << t << " State: " << m_state.transpose() << std::endl;
        }
    }
    return 0;
}


