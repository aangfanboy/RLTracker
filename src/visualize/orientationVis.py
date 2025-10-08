import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time
import os
import sys
import numpy as np

from numpy.typing import NDArray

# Add the project root to Python path when running this file directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


import utils.math as math_utils

floatMatrix = NDArray[np.float64]

class OrientationVisualizer:
    def __init__(self, model_path: str, quaternion: floatMatrix):
        self.model_path: str = model_path
        self.quaternion: floatMatrix = quaternion

        # Don't create the visualizer here - do it in the thread
        self.mesh = None
        self.vis = None

        self.running: bool = True
        self.mainThread = None

    def add_axes(self, axes_length: float = 1.0) -> None:
        """
        Add axes to the visualizer for reference.
        @param axes_length: Length of the axes.

        The x, y, z axis will be rendered as red, green, and blue arrows respectively.
        """
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_length, origin=[0, 0, 0])
        self.vis.add_geometry(axes)

    def lock_camera(self) -> None:
        """
        Lock the camera to coordinate frame. No zoom, no rotation. Look at 45 degrees of the axes.
        This is useful to prevent the camera from moving during visualization.
        """
        view_control = self.vis.get_view_control()
        
        # Set the camera to look at the origin
        view_control.set_lookat([0, 0, 0])
        
        # Set Z as up direction (standard for Open3D where Z is up)
        view_control.set_up([0, 0, -1])
        
        # Position camera at a good viewing angle
        # Looking from front-right-above to get a good 3D perspective of the plane
        # The values are normalized direction vectors
        view_control.set_front([0.4, 0.6, -0.4])  # Slightly more from the side for better plane visibility
        
        # Set a reasonable zoom level - adjust this if the plane appears too small/large
        view_control.set_zoom(1)
        
        # Optional: Lock the view to prevent user interaction
        # Uncomment the next line if you want to completely lock the camera
        # view_control.lock_view_control(True)


    def ac_coordinate_frame_to_o3d_coordinate_frame(self, rotation_matrix: floatMatrix) -> None:
        """
        Convert the coordinate frame from AC to Open3D.
        90 deg around o3d-z, -90 deg around o3d-x
        """
        R_ac_to_o3d = R.from_euler('yzx', [90, 0, -90], degrees=True)
        return rotation_matrix @ R_ac_to_o3d.as_matrix()

    def update_quaternion(self, new_quaternion: floatMatrix) -> None:
        """
        Update the quaternion for the next frame.
        @param new_quaternion: New quaternion as a floatMatrix of shape (4,1).
        """
        self.quaternion = new_quaternion

    def update_orientation(self, rotation_matrix: floatMatrix) -> None:
        """
        Update the orientation of the missile mesh based on the given quaternion.
        @param quaternion: Quaternion as a floatMatrix of shape (4,1).
        """
        transform: floatMatrix = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        self.mesh.transform(transform)

    def reset_orientation(self, rotation_matrix: floatMatrix) -> None:
        """
        Reset the orientation of the missile mesh to the initial quaternion.
        @param rotation_matrix: Rotation matrix as a floatMatrix of shape (3,3).
        """
        transform: floatMatrix = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_matrix
        self.mesh.transform(np.linalg.inv(transform))

    def run(self) -> None:
        """
        Start the visualizer and continuously update the orientation.
        """
        # Initialize the visualizer in this thread
        self.mesh = o3d.io.read_triangle_mesh(self.model_path)
        self.mesh.compute_vertex_normals()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Missile Orientation', width=800, height=600)
        self.vis.add_geometry(self.mesh)

        self.add_axes(axes_length=500.0)  # Add axes for reference
        self.lock_camera()  # Lock the camera to the current view

        while self.running:
            # Here you would typically get the latest quaternion from your data source
            # For demonstration, we will use the initial quaternion
            rotation_matrix = math_utils.calculateAttitudeMatrixFromQuaternion(self.quaternion)
            rotation_matrix = self.ac_coordinate_frame_to_o3d_coordinate_frame(rotation_matrix)

            self.update_orientation(rotation_matrix)

            self.vis.update_geometry(self.mesh)
            self.vis.poll_events()
            self.vis.update_renderer()

            # Reset transformation for next frame
            self.reset_orientation(rotation_matrix)
            time.sleep(0.02)
        
        # Clean up when stopping
        self.vis.destroy_window()

    def stop(self) -> None:
        """
        Stop the visualizer.
        """
        self.running = False
        if self.mainThread and self.mainThread.is_alive():
            self.mainThread.join()

        self.vis.destroy_window()
        self.mainThread = None

    def start_thread(self) -> None:
        """
        Start the visualizer in a separate thread.
        """
        self.mainThread = threading.Thread(target=self.run)
        self.mainThread.start()


if __name__ == "__main__":
    quaternion = np.array([[0], [0], [0], [1]], dtype=np.float64)  # Example quaternion
    visualizer = OrientationVisualizer("C:\\Users\\burak\\Desktop\\Missile-Guidance-Project\\visualize\\plane.obj", quaternion)
    visualizer.start_thread()
    
    time.sleep(10)  # Let it run for a while
    visualizer.stop()
    print("Visualizer stopped.")