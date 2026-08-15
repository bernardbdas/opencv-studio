import os
import time
import random
from collections import deque

class MetricsService:
    def __init__(self):
        # Store max 50 points
        self.max_points = 50
        self.metrics = {
            "YOLO Studio": deque(maxlen=self.max_points),
            "Mediapipe Lab": deque(maxlen=self.max_points),
            "U-Net Segmentation": deque(maxlen=self.max_points),
            "3D Depth Estimation": deque(maxlen=self.max_points),
        }
        self._initialize_mock_data()
        
    def _initialize_mock_data(self):
        # Pre-populate with base historical curves
        now = time.time()
        for i in range(self.max_points):
            t = now - (self.max_points - i) * 2  # 2s intervals
            self.metrics["YOLO Studio"].append((t, random.uniform(18, 28)))
            self.metrics["Mediapipe Lab"].append((t, random.uniform(32, 48)))
            self.metrics["U-Net Segmentation"].append((t, random.uniform(8, 16)))
            self.metrics["3D Depth Estimation"].append((t, random.uniform(85, 115)))
            
    def record(self, key: str, latency_ms: float):
        if key in self.metrics:
            self.metrics[key].append((time.time(), latency_ms))
            
    def get_chart_html(self) -> str:
        import xy
        
        # Build line marks for each service
        marks = []
        colors = {
            "YOLO Studio": "#06b6d4",        # Cyan
            "Mediapipe Lab": "#10b981",     # Emerald
            "U-Net Segmentation": "#8b5cf6", # Violet
            "3D Depth Estimation": "#f43f5e" # Rose
        }
        
        for name, data in self.metrics.items():
            if not data:
                continue
            x_vals = [pt[0] for pt in data]
            y_vals = [pt[1] for pt in data]
            
            # Convert timestamps to relative seconds
            start_t = x_vals[0]
            rel_x = [round(x - start_t, 1) for x in x_vals]
            
            marks.append(
                xy.line(
                    rel_x, y_vals,
                    color=colors[name],
                    name=name
                )
            )
            
        chart = xy.line_chart(
            *marks,
            xy.x_axis(label="Time elapsed (seconds)"),
            xy.y_axis(label="Latency (ms)"),
            xy.legend(),
            title="Real-time Performance Inference Latency"
        )
        return chart.to_html()

metrics_service = MetricsService()
