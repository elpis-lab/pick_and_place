import pyrealsense2 as rs

def list_realsense_cameras():
    """
    Lists all available RealSense cameras with their serial numbers, names, and additional info
    Returns a list of dictionaries containing camera information
    """
    cameras = []
    
    # Create a context object to manage RealSense devices
    ctx = rs.context()
    
    # Get list of connected devices
    devices = ctx.query_devices()
    
    # Iterate through all detected devices
    for i in range(len(devices)):
        device = devices[i]
        
        try:
            # Get device info
            info = {
                'name': device.get_info(rs.camera_info.name),
                'serial_number': device.get_info(rs.camera_info.serial_number),
                'firmware_version': device.get_info(rs.camera_info.firmware_version),
                'product_id': device.get_info(rs.camera_info.product_id),
                'product_line': device.get_info(rs.camera_info.product_line)
            }
            cameras.append(info)
            
        except RuntimeError as e:
            print(f"Error reading device {i}: {e}")
    
    return cameras

# Example usage
if __name__ == "__main__":
    try:
        cameras = list_realsense_cameras()
        
        if not cameras:
            print("No RealSense cameras found")
        else:
            print(f"Found {len(cameras)} RealSense camera(s):")
            for i, camera in enumerate(cameras, 1):
                print(f"\nCamera {i}:")
                print(f"  Name: {camera['name']}")
                print(f"  Serial Number: {camera['serial_number']}")
                print(f"  Product Line: {camera['product_line']}")
                print(f"  Firmware Version: {camera['firmware_version']}")
                print(f"  Product ID: {camera['product_id']}")
                
    except Exception as e:
        print(f"Error: {e}")