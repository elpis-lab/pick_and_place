#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "dynamixel_sdk.h"

#define ADDR_OPERATING_MODE     11
#define ADDR_TORQUE_ENABLE      512
#define ADDR_GOAL_CURRENT       550
#define ADDR_GOAL_POSITION      564
#define ADDR_PRESENT_POSITION   580
#define ADDR_PRESENT_CURRENT    574

#define PROTOCOL_VERSION        2.0
#define GRIPPER_ID              1
#define BAUDRATE                57600
#define DEVICENAME              "/dev/ttyUSB0"

dynamixel::PortHandler *portHandler;
dynamixel::PacketHandler *packetHandler;

extern "C" {
    bool openPort() {
        portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
        packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
        
        if (portHandler->openPort()) {
            printf("Succeeded to open the port!\n");
            if (portHandler->setBaudRate(BAUDRATE)) {
                printf("Succeeded to change the baudrate!\n");
                return true;
            }
        }
        return false;
    }

    void closePort() {
        portHandler->closePort();
    }

    void enableTorque(bool enable) {
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, enable ? 1 : 0);
    }

    void setOperatingMode(int mode) {
        packetHandler->write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_OPERATING_MODE, mode);
    }

    void setGoalPosition(int position) {
        packetHandler->write4ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, position);
    }

    void setGoalCurrent(int current) {
        packetHandler->write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_CURRENT, current);
    }

    int readPosition() {
        uint32_t position;
        packetHandler->read4ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_POSITION, &position);
        return (int)position;
    }

    int readCurrent() {
        uint16_t current;
        packetHandler->read2ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_CURRENT, &current);
        return (int)current;
    }
}