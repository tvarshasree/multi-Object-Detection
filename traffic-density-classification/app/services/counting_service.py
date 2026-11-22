"""
TrafficVision - Object Counting Service
Calculates per-class detection counts and aggregates total/vehicle/pedestrian metrics.
"""

from typing import List
from app.schemas import DetectionItem, ClassCounts


class CountingService:
    @staticmethod
    def count_objects(detections: List[DetectionItem]) -> ClassCounts:
        """
        Counts instances of each road-user class from the list of detections.
        """
        counts = {"person": 0, "car": 0, "bus": 0, "motorcycle": 0}

        for det in detections:
            cname = det.class_name.lower()
            if cname in counts:
                counts[cname] += 1

        return ClassCounts(
            person=counts["person"],
            car=counts["car"],
            bus=counts["bus"],
            motorcycle=counts["motorcycle"]
        )

    @staticmethod
    def compute_aggregates(counts: ClassCounts):
        """
        Computes vehicle count (cars + buses + motorcycles),
        pedestrian count (persons), and total objects.
        """
        vehicle_count = counts.car + counts.bus + counts.motorcycle
        pedestrian_count = counts.person
        total_objects = vehicle_count + pedestrian_count
        return vehicle_count, pedestrian_count, total_objects
