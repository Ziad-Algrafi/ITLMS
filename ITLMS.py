"""
ITLMS: An Intelligent Traffic Light Management System Using Edge Computing and Deep Learning

This module implements a computer vision-based traffic management system for the 
paper (An Intelligent Traffic Light Management System Using Edge Computing and Deep Learning) this version include:
1. Processes video feeds from multiple roads
2. Detects and classifies vehicles using YOLOv9
3. Dynamically controls traffic lights based on:
   - Emergency vehicle presence (ambulance, firefighter, police)
   - Traffic density
4. Provides a graphical user interface for monitoring

Classes:
    GUI: Manages the graphical user interface
    VideoProcessor: Handles video processing and vehicle detection
    TrafficLightSystem: Controls traffic light logic

Functions:
    on_closing: Handles window close event
"""
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import time
import os

class GUI:
    """
    Manages the graphical user interface for the traffic management system.
    
    Attributes:
        root: Tkinter root window
        frames: Dictionary of LabelFrames for each road
        labels: Dictionary of Labels for video display
        traffic_lights: Dictionary of Labels for traffic light status
    """
    def __init__(self, root, video_paths_and_roads):
        """
        Initialize the GUI.
        
        Args:
            root: Tkinter root window
            video_paths_and_roads: Dictionary mapping road names to video paths
        """
        self.root = root
        self.root.title("Traffic Management System")
        self.frames = {}
        self.labels = {}
        self.traffic_lights = {}
        for road in video_paths_and_roads:
            self.frames[road] = tk.LabelFrame(root, text=f"Traffic on  {road}", padx=10, pady=10)
            self.frames[road].grid(row=0 if int(road[-1]) % 2 == 1 else 1, column=int(road[-1]) // 3, padx=10, pady=10)
            self.labels[road] = tk.Label(self.frames[road])
            self.labels[road].grid(row=0, column=0, padx=10, pady=10)
            self.traffic_lights[road] = tk.Label(self.frames[road], text="Traffic Light: Closed", fg="red")
            self.traffic_lights[road].grid(row=1, column=0, padx=10, pady=10)

    def update_frame(self, road, frame):
        """
        Update the video frame display for a specific road.
        
        Args:
            road: Road identifier (e.g., "Road1")
            frame: Image frame to display
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (500, 400))  
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.labels[road].configure(image=photo)
        self.labels[road].image = photo
        self.root.update()  

    def update_traffic_display(self, traffic_lights):
        """
        Update traffic light status displays.
        
        Args:
            traffic_lights: Dictionary mapping roads to light states
                (True = open/green, False = closed/red)
        """
        for road, state in traffic_lights.items():
            if state:
                self.traffic_lights[road].config(text="Traffic Light: Open", fg="green")
            else:
                self.traffic_lights[road].config(text="Traffic Light: Closed", fg="red")

class VideoProcessor:
    """
    Handles video processing and vehicle detection.
    
    Attributes:
        video_caps: Dictionary of VideoCapture objects per road
        models: Dictionary of YOLO models per road
        trackers: Path to tracker configuration
        road_data: Dictionary storing traffic data per road
    """
    def __init__(self, video_paths_and_roads):
        """
        Initialize the VideoProcessor.
        
        Args:
            video_paths_and_roads: Dictionary mapping road names to video paths
        """
        self.video_caps = {}
        self.models = {}
        for road, info in video_paths_and_roads.items():
            # Use os.path.join for cross-platform compatibility
            video_path = os.path.join("assets", os.path.basename(info["video_path"]))
            if not os.path.exists(video_path):
                print(f"Error: Video file not found for {road}: {video_path}")
                continue
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video for {road}: {video_path}")
                continue
            self.video_caps[road] = cap
            self.models[road] = YOLO(os.path.join('assets', 'best.pt'))
        self.trackers = 'bytetrack.yaml' 
        self.road_data = {}

    def process_video_frame(self, road, gui):
        """
        Process a single video frame for a road.
        
        Args:
            road: Road identifier
            gui: GUI object for updating displays
            
        Returns:
            bool: True if frame processed successfully, False otherwise
        """
        video_cap = self.video_caps[road]
        road_counts = {"ambulance": 0, "firefighter": 0, "police": 0, "traffic": 0, "car": 0}

        ret, frame = video_cap.read()
        if ret:
            results = self.models[road].track(frame, imgsz=640, persist=True, verbose=False, tracker=self.trackers)
            boxes = results[0].boxes.xywh.cpu()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                track_clss = results[0].boxes.cls.int().cpu().tolist()

                for track_id, track_cls, box in zip(track_ids, track_clss, boxes):
                    x, y, w, h = box
                    cls = track_cls

                    normalized_threshold = 7000
                    object_Area = w * h 
                                
                    if cls == 0:
                        road_counts["ambulance"] += 1
                    elif cls == 1:
                        road_counts["firefighter"] += 1
                    elif cls == 4:
                        road_counts["police"] += 1
                    elif cls == 3:
                        road_counts["car"] += 1
                    elif cls == 2:
                        if object_Area > normalized_threshold:
                            road_counts["traffic"] += 7 # Increment count by a smaller value for close objects
                        else:
                            road_counts["traffic"] += 13   # Increment count by a larger value for far objects
                
                    x_min = x - w / 2
                    y_min = y - h / 2
                    x_max = x + w / 2
                    y_max = y + h / 2

                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)
                    x_above = int(x_min + 5)
                    y_above = int(y_min - 5)
                    
                    cv2.putText(frame, str(track_id), (x_above, y_above), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Update road data with counts and track_ids length
                self.road_data[road] = [road_counts, len(track_ids)]
            
                road_counts_dict = self.road_data[road][0]

                # Calculate the sum of values in road_counts_dict
                total_count_road_counts = sum(road_counts_dict.values())
                      
                # Draw counts on the frame
                cv2.putText(frame, f"Road Count: {total_count_road_counts}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Ambulance: {self.road_data[road][0]['ambulance']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Firefighter: {self.road_data[road][0]['firefighter']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Police: {self.road_data[road][0]['police']}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Traffic jam: {self.road_data[road][0]['traffic']}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Cars: {self.road_data[road][0]['car']}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            gui.update_frame(road, frame)
       
            traffic_system.process_and_update_traffic()  
        return ret
    
    def get_road_data(self):
        """
        Get current road traffic data.
        
        Returns:
            dict: Road data containing vehicle counts and track IDs
        """
        return self.road_data
    
    def process_next_frame(self, road, gui):
        """
        Process the next video frame for a road and schedule the next frame.
        
        Args:
            road: Road identifier
            gui: GUI object for updating displays
        """
        if road not in self.video_caps:
            return
            
        video_cap = self.video_caps[road]
        if not video_cap.isOpened():
            return
            
        ret = self.process_video_frame(road, gui)
        if not ret:
            video_cap.release()
            del self.video_caps[road]
            print(f"No more frames for {road}")
            return
            
        # Schedule next frame processing
        gui.root.after(10, lambda: self.process_next_frame(road, gui))
        
    def start_processing(self, gui):
        """
        Start processing all video feeds asynchronously.
        
        Args:
            gui: GUI object for updating displays
        """
        for road in list(self.video_caps.keys()):
            self.process_next_frame(road, gui)
              
class TrafficLightSystem:
    """
    Controls traffic light logic based on vehicle detection.
    
    Attributes:
        video_processor: Reference to VideoProcessor instance
        road_data: Current traffic data
        traffic_lights: Dictionary of current light states per road
        last_open_data: Records when roads were last opened
        gui: Reference to GUI instance
        last_open_time_em: Timestamp of last emergency vehicle detection
    """
    def __init__(self, video_processor):
        """
        Initialize the TrafficLightSystem.
        
        Args:
            video_processor: VideoProcessor instance
        """
        self.video_processor = video_processor
        self.road_data = None
        self.traffic_lights = {road: False for road in video_processor.video_caps}  # Initialize with False for all roads
        self.last_open_data = {}   
        self.gui = None
        self.last_open_time_em = 0

    def set_gui(self, gui):
        """
        Set the GUI reference for status updates.
        
        Args:
            gui: GUI instance
        """
        self.gui = gui

    def lock_traffic(self, road):
        """
        Lock traffic light (set to red) for a road.
        
        Args:
            road: Road identifier
        """
        if not self.traffic_lights[road]:  # Check if already locked
            print(f"Traffic light for {road} is already locked", "\n")
            return
        print("Before locking Traffic:", self.traffic_lights) 
        print(f"Locking traffic light for {road}")
        self.traffic_lights[road] = False
        print("After locking Traffic:", self.traffic_lights)  

    def unlock_traffic(self, road):
        """
        Unlock traffic light (set to green) for a road.
        
        Args:
            road: Road identifier
        """
        if self.traffic_lights[road]:  # Check if already unlocked
            print(f"Traffic light for {road} is already unlocked", "\n")
            return
        print("Before unlocking Traffic:", self.traffic_lights)  
        print(f"Unlocking traffic light for {road}")
        self.traffic_lights[road] = True
        print("After unlocking Traffic:", self.traffic_lights, "\n")  

    def update_traffic_light(self):
        """
        Update traffic light states based on:
        1. Emergency vehicle presence (highest priority)
        2. Traffic density (when no emergencies)
        """
        if self.road_data is None:
            return 

        road_counts = {road: data[0] for road, data in self.road_data.items()}
   
        has_emergency_vehicle = any(road_counts[r][vehicle] > 0 for r in road_counts for vehicle in ["ambulance", "firefighter", "police"])
      
        if has_emergency_vehicle:
            # Priority order: ambulance > firefighter > police
            for road, counts in road_counts.items():
                if counts["ambulance"] > 0:
                    self.update_traffic_with_lock(road, has_emergency_vehicle)
                    break
                elif counts["firefighter"] > 0:
                    self.update_traffic_with_lock(road, has_emergency_vehicle)

                    break
                elif counts["police"] > 0:
                    self.update_traffic_with_lock(road, has_emergency_vehicle)
                    break

        else:
    
            total_count_road_counts = {road: sum(road_counts[road].values()) for road in road_counts}

            max_count_road = max(total_count_road_counts, key=total_count_road_counts.get)

            for road in self.road_data.keys():
                    if road == max_count_road:
                       self.update_traffic_with_lock(road)

    def update_traffic_with_lock(self, road, has_emergency_vehicle=False):
        """
        Update traffic light with locking mechanism.
        
        Args:
            road: Road identifier
            has_emergency_vehicle: Whether emergency vehicle is present
        """

        try:
            if has_emergency_vehicle and self.traffic_lights[road] == False:

                for prev_road in self.last_open_data.keys():
                        if self.last_open_data[prev_road]["open_road"] != road:
                            self.lock_traffic(prev_road)

                self.unlock_traffic(road)
                self.last_open_data[road] = {"time": time.time(), "open_road": road}
                self.last_open_time_em = time.time()

                if self.gui:
                    self.gui.update_traffic_display(self.traffic_lights)

            else:
                if (road in self.traffic_lights and
                    self.traffic_lights[road] == False and
                    time.time() - self.last_open_data.get(road, {}).get("time", 0) >= 10 and 
                    has_emergency_vehicle == False and 
                    time.time() - self.last_open_time_em >= 10):

                    for prev_road in self.last_open_data.keys():
                        if self.last_open_data[prev_road]["open_road"] != road:
                            self.lock_traffic(prev_road)

                    self.unlock_traffic(road)
                    self.last_open_data[road] = {"time": time.time(), "open_road": road}
                    self.last_open_time_em = 0

                    if self.gui:
                        self.gui.update_traffic_display(self.traffic_lights)         

        except KeyError:
         
                        self.last_open_data[road] = {"time": time.time(), "open_road": road}

    def process_and_update_traffic(self):
        """
        Process road data and update traffic light states.
        Also updates the GUI display.
        """
        self.road_data = self.video_processor.get_road_data()
        self.update_traffic_light()

    def check_and_open_overdue_road(self):
        """
        Check for roads closed longer than 2 minutes and
        open the one with the most cars.
        """
        current_time = time.time()
        for road in self.last_open_data.keys():
            if current_time - self.last_open_data[road]["time"] >= 120:  # 2 minutes
                eligible_roads = [r for r in self.last_open_data if time.time() - self.last_open_data[r]["time"] >= 10 and self.road_data[r][0]["car"] > 0]
                if eligible_roads:
                    new_open_road = max(eligible_roads, key=lambda r: self.road_data[r][0]["car"])
                    for prev_road in self.last_open_data.keys():
                        if self.last_open_data[prev_road]["open_road"] != new_open_road:
                            self.lock_traffic(prev_road)
                    # Unlock the new road after locking the previous one
                    self.unlock_traffic(new_open_road)
                    self.last_open_data[new_open_road] = {"time": current_time, "open_road": new_open_road}
             
                
                
                    if self.gui:
                        self.gui.update_traffic_display(self.traffic_lights)
                break  


if __name__ == "__main__":

    def on_closing():
        """
        Handle application shutdown when window is closed.
        """
        root.destroy()
    
    video_paths_and_roads = {
        "Road1": {"video_path": "Crowded1.mp4", "road": 1},
        "Road2": {"video_path": "Crowded2.mp4", "road": 2},
        "Road3": {"video_path": "ambulance-with-traffic.mp4", "road": 3},
        "Road4": {"video_path": "Road.mp4", "road": 4},
    }


    root = tk.Tk()
    gui = GUI(root, video_paths_and_roads)
    processor = VideoProcessor(video_paths_and_roads)
    traffic_system = TrafficLightSystem(processor)
    traffic_system.set_gui(gui)
    root.protocol("WM_DELETE_WINDOW", on_closing)  

    # Start video processing
    processor.start_processing(gui)
    
    # Function to update traffic system periodically
    def update_traffic_system():
        traffic_system.process_and_update_traffic()
        traffic_system.check_and_open_overdue_road()
        root.after(100, update_traffic_system)  # Update every 100ms
        
    # Start traffic system updates
    update_traffic_system()

    root.mainloop()
