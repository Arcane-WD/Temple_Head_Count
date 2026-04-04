import cv2
import numpy as np

class ZoneFilter:
    def __init__(self, exclusion_polygons: list):
        """
        exclusion_polygons: list of lists of (x, y) tuples.
        Example: [[(10, 10), (100, 10), (100, 100), (10, 100)]]
        """
        # Convert to numpy arrays of shape (N, 1, 2) and type int32 for cv2
        self.polygons = []
        for poly in exclusion_polygons:
            if poly and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                self.polygons.append(pts)

    def is_excluded(self, x, y) -> bool:
        """
        Check if point (x, y) falls inside any exclusion polygon.
        """
        if not self.polygons:
            return False

        point = (float(x), float(y))
        for poly in self.polygons:
            # pointPolygonTest returns +1 if inside, 0 if on edge, -1 if outside
            if cv2.pointPolygonTest(poly, point, False) >= 0:
                return True
                
        return False
        
    def draw_zones(self, frame):
        """
        Draw the exclusion zones on a frame for debugging/visualization.
        """
        for poly in self.polygons:
            cv2.polylines(frame, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
            # Add subtle transparent overlay
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], color=(0, 0, 255))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
        return frame
