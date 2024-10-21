import os
import sys
import time
import ctypes

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
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 512
ADDR_GOAL_CURRENT = 550
ADDR_GOAL_POSITION = 564
ADDR_PRESENT_POSITION = 580
ADDR_PRESENT_CURRENT = 574

PROTOCOL_VERSION = 2.0
GRIPPER_ID = 1
BAUDRATE = 57600
DEVICENAME = "/dev/ttyUSB0"

MIN_POSITION = 0
MAX_POSITION = 1150
MIN_CURRENT = -1984
MAX_CURRENT = 1984

MODE_CURRENT = 0
MODE_POSITION = 5

class GripperState:
    def __init__(self):
        self.current_mode = MODE_POSITION
        self.is_torque_on = False
        self.goal_position = 740
        self.goal_current = 0

gripper_state = GripperState()

def print_menu():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("RH-P12-RN(A) Example")
    print(f"Mode: {'Current' if gripper_state.current_mode == MODE_CURRENT else 'Position'} control")
    print(f"Torque: {'On' if gripper_state.is_torque_on else 'Off'}")
    print(f"Goal Position: {gripper_state.goal_position}")
    print(f"Goal Current: {gripper_state.goal_current}")
    print("\nCommands:")
    print("p: Switch to Position control mode")
    print("c: Switch to Current control mode")
    print("t: Toggle torque")
    print("o: Open gripper")
    print("l: Close gripper")
    print("g: Go to goal position (Position mode only)")
    print("[: Decrease value")
    print("]: Increase value")
    print("q: Quit")

def initialize_gripper():
    if not lib.openPort():
        print("Failed to open port")
        return False
    
    lib.setOperatingMode(gripper_state.current_mode)
    lib.enableTorque(True)
    gripper_state.is_torque_on = True
    return True

def set_operating_mode(mode):
    if gripper_state.is_torque_on:
        lib.enableTorque(False)
    lib.setOperatingMode(mode)
    lib.enableTorque(True)
    gripper_state.current_mode = mode
    gripper_state.is_torque_on = True

def open_gripper():
    if gripper_state.current_mode == MODE_POSITION:
        lib.setGoalPosition(MIN_POSITION)
    else:
        lib.setGoalCurrent(MIN_CURRENT)

def close_gripper():
    if gripper_state.current_mode == MODE_POSITION:
        lib.setGoalPosition(MAX_POSITION)
    else:
        lib.setGoalCurrent(MAX_CURRENT)

def main():
    if not initialize_gripper():
        return

    while True:
        print_menu()
        cmd = input("Enter command: ").lower()

        if cmd == 'q':
            break
        elif cmd == 'p':
            set_operating_mode(MODE_POSITION)
        elif cmd == 'c':
            set_operating_mode(MODE_CURRENT)
        elif cmd == 't':
            gripper_state.is_torque_on = not gripper_state.is_torque_on
            lib.enableTorque(gripper_state.is_torque_on)
        elif cmd == 'o':
            open_gripper()
        elif cmd == 'l':
            close_gripper()
        elif cmd == 'g' and gripper_state.current_mode == MODE_POSITION:
            lib.setGoalPosition(gripper_state.goal_position)
        elif cmd == '[':
            if gripper_state.current_mode == MODE_POSITION:
                gripper_state.goal_position = max(MIN_POSITION, gripper_state.goal_position - 10)
            else:
                gripper_state.goal_current = max(MIN_CURRENT, gripper_state.goal_current - 10)
        elif cmd == ']':
            if gripper_state.current_mode == MODE_POSITION:
                gripper_state.goal_position = min(MAX_POSITION, gripper_state.goal_position + 10)
            else:
                gripper_state.goal_current = min(MAX_CURRENT, gripper_state.goal_current + 10)

        time.sleep(0.1)

    lib.enableTorque(False)
    lib.closePort()
    print("Program terminated.")

if __name__ == "__main__":
    main()