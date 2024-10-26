from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Vector3

def setup_interactive_markers(self):
    """Initialize interactive marker server"""
    self.marker_server = InteractiveMarkerServer(
        self,
        'pick_point_markers'
    )
    self.current_marker = None

def create_pick_point_marker(self, target_pose):
    """Create an interactive marker at the target pick point"""
    # Create the interactive marker
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = "base_link"
    int_marker.name = "pick_point"
    int_marker.description = "Pick Point Target"
    int_marker.pose = target_pose
    int_marker.scale = 0.2

    # Create a marker for visualization
    marker = Marker()
    marker.type = Marker.SPHERE
    marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    # Create marker control
    marker_control = InteractiveMarkerControl()
    marker_control.always_visible = True
    marker_control.markers.append(marker)
    int_marker.controls.append(marker_control)

    # Add 6-DOF controls
    control_names = ["rotate_x", "rotate_y", "rotate_z", "move_x", "move_y", "move_z"]
    control_axes = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (False, False, True)
    ]
    control_types = [
        InteractiveMarkerControl.ROTATE_AXIS,
        InteractiveMarkerControl.ROTATE_AXIS,
        InteractiveMarkerControl.ROTATE_AXIS,
        InteractiveMarkerControl.MOVE_AXIS,
        InteractiveMarkerControl.MOVE_AXIS,
        InteractiveMarkerControl.MOVE_AXIS
    ]

    for name, axis, control_type in zip(control_names, control_axes, control_types):
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation.w = 1.0
        control.orientation.x = axis[0]
        control.orientation.y = axis[1]
        control.orientation.z = axis[2]
        control.interaction_mode = control_type
        int_marker.controls.append(control)

    # Add the interactive marker to the server
    self.marker_server.insert(int_marker, self.marker_feedback_callback)
    self.marker_server.applyChanges()
    self.current_marker = int_marker

def marker_feedback_callback(self, feedback):
    """Handle interactive marker feedback"""
    if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        # Update the target pose based on marker movement
        self.get_logger().info(f"Marker moved to position: {feedback.pose.position}")
        return feedback.pose

def update_marker_pose(self, new_pose):
    """Update the position of the interactive marker"""
    if self.current_marker:
        self.current_marker.pose = new_pose
        self.marker_server.insert(self.current_marker)
        self.marker_server.applyChanges()

def remove_marker(self):
    """Remove the interactive marker"""
    if self.current_marker:
        self.marker_server.erase(self.current_marker.name)
        self.marker_server.applyChanges()
        self.current_marker = None