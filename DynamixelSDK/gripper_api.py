import os
import sys
import time
from dynamixel_sdk import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Gripper Configuration
ADDR_OPERATING_MODE     = 11
ADDR_TORQUE_ENABLE      = 512
ADDR_GOAL_PWM           = 548
ADDR_GOAL_CURRENT       = 550
ADDR_GOAL_VELOCITY      = 552
ADDR_GOAL_POSITION      = 564
ADDR_MOVING             = 570
ADDR_PRESENT_CURRENT    = 574
ADDR_PRESENT_VELOCITY   = 576
ADDR_PRESENT_POSITION   = 580

MIN_POSITION            = 0
MAX_POSITION            = 1150
MIN_VELOCITY            = 0
MAX_VELOCITY            = 2970
MIN_CURRENT             = 0
MAX_CURRENT             = 1984
MIN_PWM                 = 0
MAX_PWM                 = 2009

PROTOCOL_VERSION        = 2.0
BAUDRATE                = 57600
DEVICENAME              = '/dev/ttyUSB0'
GRIPPER_ID              = 1

# Control Mode
MODE_CURRENT_BASED_POSITION = 5

# FastAPI app
app = FastAPI()

# Pydantic model for torque control request
class TorqueControlRequest(BaseModel):
    max_torque: int
    speed: int = 100

# Global variables
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

def initialize_gripper():
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        return False

    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        return False

    # Enable Dynamixel Torque
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_TORQUE_ENABLE, 1)
    if dxl_comm_result != COMM_SUCCESS:
        print("In Success")
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        return False
    elif dxl_error != 0:
        print("In Error")
        print("%s" % packetHandler.getRxPacketError(dxl_error))
        return False

    # Set operating mode to current-based position control
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, GRIPPER_ID, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        print("Failed to set operating mode")
        return False

    print("Gripper initialized successfully")
    return True

@app.post("/open_gripper")
async def open_gripper():
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, MIN_POSITION)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise HTTPException(status_code=500, detail="Failed to open gripper")
    return {"message": "Gripper opened"}

@app.post("/close_gripper")
async def close_gripper():
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, MAX_POSITION)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise HTTPException(status_code=500, detail="Failed to close gripper")
    return {"message": "Gripper closed"}

@app.post("/close_gripper_with_torque")
async def close_gripper_with_torque_control(request: TorqueControlRequest):
    # Set the maximum torque (current limit)
    max_torque = max(MIN_CURRENT, min(request.max_torque, MAX_CURRENT))
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_CURRENT, max_torque)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise HTTPException(status_code=500, detail="Failed to set maximum torque")

    # Set the closing speed
    speed = max(MIN_VELOCITY, min(request.speed, MAX_VELOCITY))
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_VELOCITY, speed)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise HTTPException(status_code=500, detail="Failed to set closing speed")

    # Start closing the gripper
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, GRIPPER_ID, ADDR_GOAL_POSITION, MAX_POSITION)
    if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
        raise HTTPException(status_code=500, detail="Failed to start closing gripper")

    # Wait for the gripper to stop moving or reach the torque limit
    while True:
        moving, result, error = packetHandler.read1ByteTxRx(portHandler, GRIPPER_ID, ADDR_MOVING)
        if result != COMM_SUCCESS or error != 0:
            raise HTTPException(status_code=500, detail="Failed to read moving status")
        
        if moving == 0:
            break
        
        time.sleep(0.1)

    final_position, result, error = packetHandler.read4ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_POSITION)
    if result != COMM_SUCCESS or error != 0:
        raise HTTPException(status_code=500, detail="Failed to read final position")

    return {"message": "Gripper closed with torque control", "final_position": final_position}

@app.get("/get_position")
async def get_position():
    position, result, error = packetHandler.read4ByteTxRx(portHandler, GRIPPER_ID, ADDR_PRESENT_POSITION)
    if result != COMM_SUCCESS or error != 0:
        raise HTTPException(status_code=500, detail="Failed to read position")
    return {"position": position}

if __name__ == "__main__":
    if not initialize_gripper():
        print("Failed to initialize gripper")
        sys.exit(1)
    
    uvicorn.run(app, host="0.0.0.0", port=8005)