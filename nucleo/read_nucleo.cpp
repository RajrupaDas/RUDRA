#include <iostream>
#include <sstream>
//#include <vector>
#include <libserial/SerialPort.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace LibSerial;
using namespace std;

#define SERIAL_PORT "/dev/pts/3"
#define BAUD_RATE 9600

int main() {
    SerialPort serial;

    try {
        // open serial port
        serial.Open(SERIAL_PORT);
        if (!serial.IsOpen()) {
           cerr << "Error: Serial port failed to open!" << endl;
           return -1;
        }

        serial.SetBaudRate(BaudRate::BAUD_9600);
        serial.SetCharacterSize(CharacterSize::CHAR_SIZE_8);
        serial.SetStopBits(StopBits::STOP_BITS_1);
        serial.SetParity(Parity::PARITY_NONE);

        cout << "Listening on " << SERIAL_PORT << " at " << BAUD_RATE << " baud..." << endl;

        while (true) {
            // read line
            std::string data;
            serial.ReadLine(data, '\n', 500);  // 500ms rest

            // parse data
            std::stringstream ss(data);
            double gps_x, gps_y;
            double imu_accel_x, imu_accel_y, imu_accel_z;
            double imu_gyro_x, imu_gyro_y, imu_gyro_z;
            double control_acceleration, control_yaw_rate;

            ss >> gps_x >> gps_y
               >> imu_accel_x >> imu_accel_y >> imu_accel_z
               >> imu_gyro_x >> imu_gyro_y >> imu_gyro_z
               >> control_acceleration >> control_yaw_rate;

            // store in eigen vector
            VectorXd sensor_data(10);
            sensor_data << gps_x, gps_y,
                           imu_accel_x, imu_accel_y, imu_accel_z,
                           imu_gyro_x, imu_gyro_y, imu_gyro_z,
                           control_acceleration, control_yaw_rate;

            // print
            cout << "GPS: (" << sensor_data(0) << ", " << sensor_data(1) << ")"
                 << " | Accel: (" << sensor_data(2) << ", " << sensor_data(3) << ", " << sensor_data(4) << ")"
                 << " | Gyro: (" << sensor_data(5) << ", " << sensor_data(6) << ", " << sensor_data(7) << ")"
                 << " | Control: (" << sensor_data(8) << ", " << sensor_data(9) << ")"
                 << endl;
        }

    } catch (const std::exception &e) {
        cerr << "Serial Error: " << e.what() << endl;
        return -1;
    }

    serial.Close();
    return 0;
}
