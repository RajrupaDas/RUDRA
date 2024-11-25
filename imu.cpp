#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <cstring>

int setupSerialPort(const char *portname, int baudRate) {
    int fd = open(portname, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        std::cerr << "Unable to open port " << portname << std::endl;
        return -1;
    }

    struct termios options;
    tcgetattr(fd, &options);
    cfsetispeed(&options, baudRate);
    cfsetospeed(&options, baudRate);

    options.c_cflag &= ~PARENB;  // No parity
    options.c_cflag &= ~CSTOPB;  // 1 stop bit
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;      // 8 bits

    options.c_cflag &= ~CRTSCTS; //no hardware flow controls
    options.c_cflag |= CREAD | CLOCAL; //enable receiver... ignore modem control lines

    tcsetattr(fd, TCSANOW, &options); //apply config.

    return fd;
}

void readIMUData(int serialPort) {
    char buf[256];
    std::string imu_data;
    
    while (true) {
        int bytesRead = read(serialPort, buf, sizeof(buf));
        if (bytesRead > 0) {
            buf[bytesRead] = '\0';  //terminate
            imu_data = std::string(buf); //buffer to string

            std::cout << imu_data << std::endl;

            //can parse data here
            // Example: Parse and extract ax, ay, az
            if (imu_data.find("Filtered") != std::string::npos) {
                std::cout << "Received IMU Data: " << imu_data << std::endl;
            }
        }
    }
}

int main() {
    const char *portname = "/dev/ttyUSB0";//double check
    int baudRate = B115200; //double check

    //set up serial port
    int serialPort = setupSerialPort(portname, baudRate);
    if (serialPort == -1) {
        return 1;
    }

    
    readIMUData(serialPort);

    close(serialPort); 
    return 0;
}

