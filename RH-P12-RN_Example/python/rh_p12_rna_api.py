import ctypes
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the wrapper directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'wrapper'))

# Load the shared library
lib = ctypes.CDLL('../wrapper/librh_p12_rna.so')

# Define function prototypes
lib.openPort.restype = ctypes.c_bool
lib.closePort.restype = None
lib.enableTorque.argtypes = [ctypes.c_bool]
lib.setOperatingMode.argtypes = [ctypes.c_int]
lib.setGoalPosition.argtypes = [ctypes.c_int]
lib.setGoalCurrent.argtypes = [ctypes.c_int]
lib.readPosition.restype = ctypes.c_int
lib.readCurrent.restype = ctypes.c_int

# Constants
MIN_POSITION = 0
MAX_POSITION = 1150

app = FastAPI()

class GripperState:
    def __init__(self):
        self.is_initialized = False
        self.current_mode = 5  # Default to Position Control Mode

gripper_state = GripperState()

@app.get("/")
async def welcome():
    return {"message": "Welcome to RH-P12-RN(A) Gripper Control API"}

@app.post("/open")
async def open_gripper():
    if not gripper_state.is_initialized:
        if not initialize_gripper():
            raise HTTPException(status_code=500, detail="Failed to initialize gripper")
    
    if gripper_state.current_mode == 5:  # Position Control Mode
        lib.setGoalPosition(MIN_POSITION)
    else:  # Current Control Mode
        lib.setGoalCurrent(-1000)  # Use a negative current to open
    return {"message": "Gripper opening"}

@app.post("/close")
async def close_gripper():
    if not gripper_state.is_initialized:
        if not initialize_gripper():
            raise HTTPException(status_code=500, detail="Failed to initialize gripper")
    
    if gripper_state.current_mode == 5:  # Position Control Mode
        lib.setGoalPosition(MAX_POSITION)
    else:  # Current Control Mode
        lib.setGoalCurrent(1000)  # Use a positive current to close
    return {"message": "Gripper closing"}

@app.get("/get_position")
async def get_position():
    if not gripper_state.is_initialized:
        if not initialize_gripper():
            raise HTTPException(status_code=500, detail="Failed to initialize gripper")
    
    position = lib.readPosition()
    current = lib.readCurrent()
    return {"position": position, "current": current}

def initialize_gripper():
    if not lib.openPort():
        return False
    
    lib.setOperatingMode(gripper_state.current_mode)
    lib.enableTorque(True)
    gripper_state.is_initialized = True
    return True

@app.on_event("shutdown")
async def shutdown_event():
    if gripper_state.is_initialized:
        lib.enableTorque(False)
        lib.closePort()
    print("Shutting down and cleaning up...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)