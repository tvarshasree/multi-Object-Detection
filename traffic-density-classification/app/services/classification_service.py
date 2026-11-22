"""
TrafficVision - Traffic Density Classification Service
Implements three-tier density classification (Low / Medium / High)
with configurable thresholds and human-readable explanations.
"""

from app.schemas import DensityClassification, ClassCounts
from app.config import settings


class ClassificationService:
    @staticmethod
    def classify_traffic_density(
        vehicle_count: int,
        pedestrian_count: int,
        total_objects: int,
        low_thresh: int = None,
        med_thresh: int = None
    ) -> DensityClassification:
        """
        Classifies traffic density into Low, Medium, or High based on vehicle count.
        Pedestrian volume is explicitly kept separate to prevent false alerts.
        """
        t1 = low_thresh if low_thresh is not None else settings.LOW_THRESHOLD
        t2 = med_thresh if med_thresh is not None else settings.MEDIUM_THRESHOLD

        if vehicle_count <= t1:
            level = "Low"
            color = "#10b981"  # Green
            explanation = (
                f"Low traffic density observed ({vehicle_count} vehicle{'s' if vehicle_count != 1 else ''} detected, "
                f"threshold <= {t1}). Road condition is free-flowing. "
                f"{pedestrian_count} pedestrian{'s' if pedestrian_count != 1 else ''} tracked separately."
            )
        elif vehicle_count <= t2:
            level = "Medium"
            color = "#f59e0b"  # Amber
            explanation = (
                f"Moderate traffic density observed ({vehicle_count} vehicles detected, "
                f"threshold between {t1} and {t2}). Road occupancy is moderate with steady flow. "
                f"{pedestrian_count} pedestrian{'s' if pedestrian_count != 1 else ''} tracked separately."
            )
        else:
            level = "High"
            color = "#ef4444"  # Red
            explanation = (
                f"High traffic congestion detected ({vehicle_count} vehicles detected, "
                f"exceeding high threshold of {t2}). Heavy vehicular density indicates congestion. "
                f"{pedestrian_count} pedestrian{'s' if pedestrian_count != 1 else ''} tracked separately."
            )

        return DensityClassification(
            density_level=level,
            total_objects=total_objects,
            vehicle_count=vehicle_count,
            pedestrian_count=pedestrian_count,
            low_threshold=t1,
            medium_threshold=t2,
            explanation=explanation,
            badge_color=color
        )
