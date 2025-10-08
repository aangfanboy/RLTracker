import numpy as np
import threading
import time
from numpy.typing import NDArray
import queue
import open3d as o3d

floatMatrix = NDArray[np.float64]

"""
This will only show positions of the target and missile in 3D space using Open3D.
It does not visualize orientations or velocities.
"""

class PointMapVisualizer:
    def __init__(self, points: floatMatrix):
        self.points = points
        self.vis = None
        self.running = False
        self.thread = None
        self.update_queue = queue.Queue()
        
        # Create geometries for missile and target
        self.missile_sphere = None
        self.target_sphere = None
        self.coordinate_frame = None
        
    def start(self):
        """Start the visualization in a separate thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_visualization, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the visualization"""
        self.running = False
        if self.vis is not None:
            self.vis.close()
        if self.thread and self.thread.is_alive():
            self.thread.join()

    def _run_visualization(self):
        """Run the visualization loop in the thread"""
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="Missile and Target Positions", width=800, height=600)
            
            # Create initial geometries
            self._create_geometries()
            
            # Add geometries to visualizer
            self.vis.add_geometry(self.missile_sphere)
            self.vis.add_geometry(self.target_sphere)
            self.vis.add_geometry(self.coordinate_frame)
            
            # Set up camera
            self._setup_camera()
            
            # Main visualization loop
            while self.running:
                # Process updates
                self._process_updates()
                
                # Update visualization
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
        except Exception as e:
            print(f"Visualization error: {e}")
        finally:
            if self.vis is not None:
                self.vis.destroy_window()

    def _create_geometries(self):
        """Create the 3D geometries for missile and target"""
        # Create missile sphere (red)
        self.missile_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        self.missile_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        
        # Create target sphere (blue)
        self.target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        self.target_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
        
        # Create coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        
        # Initial positioning
        self._update_geometry_positions()

    def _update_geometry_positions(self):
        """Update the positions of the geometries"""
        if len(self.points) >= 2:
            # Update missile position
            missile_pos = self.points[0]
            self.missile_sphere.translate(missile_pos, relative=False)
            
            # Update target position
            target_pos = self.points[1]
            self.target_sphere.translate(target_pos, relative=False)

    def _setup_camera(self) -> None:
        """
        Lock the camera to coordinate frame. No zoom, no rotation. Look at 45 degrees of the axes.
        This is useful to prevent the camera from moving during visualization.
        """
        view_control = self.vis.get_view_control()
        
        # Set the camera to look at the origin
        view_control.set_lookat([0, 0, 0])
        
        # Set Z as up direction (standard for Open3D where Z is up)
        view_control.set_up([0, 0, 1])
        
        # Position camera at a good viewing angle
        # Looking from front-right-above to get a good 3D perspective of the plane
        # The values are normalized direction vectors
        view_control.set_front([0.4, 0.6, -0.4])  # Slightly more from the side for better plane visibility
        
        # Set a reasonable zoom level - adjust this if the plane appears too small/large
        view_control.set_zoom(4)
        
        # Optional: Lock the view to prevent user interaction
        # Uncomment the next line if you want to completely lock the camera
        # view_control.lock_view_control(True)

    def _process_updates(self):
        """Process queued updates"""
        if not self.update_queue.empty():
            try:
                # Get the latest update
                new_points = self.update_queue.get_nowait()
                self.points = new_points
                
                # Update geometry positions
                self._update_geometry_positions()
                
                # Update geometries in visualizer
                self.vis.update_geometry(self.missile_sphere)
                self.vis.update_geometry(self.target_sphere)
                
                # Clear any remaining updates to avoid lag
                while not self.update_queue.empty():
                    try:
                        self.update_queue.get_nowait()
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                pass

    def update_points(self, new_points):
        """Thread-safe method to update points"""
        if self.running:
            # Queue the update
            try:
                self.update_queue.put_nowait(new_points)
            except queue.Full:
                # If queue is full, clear it and add the new update
                while not self.update_queue.empty():
                    try:
                        self.update_queue.get_nowait()
                    except queue.Empty:
                        break
                self.update_queue.put_nowait(new_points)

    def show(self):
        """Show the visualization"""
        if not self.running:
            self.start()

    def close(self):
        """Close the visualization"""
        self.stop()

def visualize_3d_point_map(missile_position: floatMatrix, target_position: floatMatrix):
    """Create and start a 3D point map visualizer"""
    points = np.array([missile_position.flatten(), target_position.flatten()])
    visualizer = PointMapVisualizer(points)
    visualizer.start()
    return visualizer

if __name__ == "__main__":
    # Example usage
    missile_position = np.array([[0.0, 0.0, 0.0]])
    target_position = np.array([[10.0, 10.0, 10.0]])
    
    visualizer = visualize_3d_point_map(missile_position, target_position)
    
    # Simulate movement for demonstration
    try:
        for i in range(200):
            # Update missile position (moving towards target)
            progress = i / 200.0
            new_missile_pos = missile_position * (1 - progress) + target_position * progress
            new_missile_pos += np.random.normal(0, 0.1, (1, 3))  # Add some noise
            
            new_points = np.array([new_missile_pos.flatten(), target_position.flatten()])
            visualizer.update_points(new_points)
            
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopping visualization...")
    finally:
        visualizer.stop()