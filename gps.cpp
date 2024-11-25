#include <iostream>
#include <fstream>
#include <vector>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#define UBX_HEADER_1 0xB5
#define UBX_HEADER_2 0x62
#define NAV_PVT_ID   0x07

// Function to set up the serial port
int setupSerialPort(const char *portname, int baudRate) {
    int fd = open(portname, O_RDWR | O_NOCTTY);
    if (fd == -1) {
        std::cerr << "Unable to open port " << portname << std::endl;
        return -1;
    }

    // Configure serial port
    struct termios options;
    tcgetattr(fd, &options);

    cfsetispeed(&options, baudRate); //input 
    cfsetospeed(&options, baudRate); //output

    options.c_cflag &= ~PARENB;  // No parity
    options.c_cflag &= ~CSTOPB;  // 1 stop bit
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;      // 8 bits

    options.c_cflag &= ~CRTSCTS; //no hardware flow control
    options.c_cflag |= CREAD | CLOCAL; //enable receiver ignore modem control lines

    tcsetattr(fd, TCSANOW, &options); //apply config

    return fd;
}

void readGPSData(int serialPort) {
    uint8_t buffer[256];
    size_t index = 0;

    while (true) {
        int byte = read(serialPort, &buffer[index], 1);
        if (byte == -1) {
            std::cerr << "Error reading from serial port" << std::endl;
            break;
        }

        if (index == 0 && buffer[index] == UBX_HEADER_1) {
            // Header 1 found... wait for header 2
            index++;
        } else if (index == 1 && buffer[index] == UBX_HEADER_2) {
            // Header 2 found ready to read the message
            index++;
        } else if (index >= 2) {
            //in message body
            size_t msgLength = buffer[2] + (buffer[3] << 8); // length at byte 2, 3
            if (index >= 6 + msgLength) {
                //full message received... parse
                if (buffer[4] == 0x01 && buffer[5] == NAV_PVT_ID) {
                    long lat = (long)buffer[6] | (long)buffer[7] << 8 | (long)buffer[8] << 16 | (long)buffer[9] << 24;
                    long lon = (long)buffer[10] | (long)buffer[11] << 8 | (long)buffer[12] << 16 | (long)buffer[13] << 24;

                    std::cout << "Latitude: " << lat << std::endl;
                    std::cout << "Longitude: " << lon << std::endl;
                }
                index = 0; //reset buffer after processing msg
            }
        }

        //next byte
        index = (index + 1) % sizeof(buffer);
    }
}

int main() {
    const char *portname = "/dev/ttyUSB0"; //double check
    int baudRate = B230400;

    int serialPort = setupSerialPort(portname, baudRate);
    if (serialPort == -1) {
        return 1;
    }

    std::cout << "Reading GPS data..." << std::endl;

    readGPSData(serialPort);

    close(serialPort);
    return 0;
}
