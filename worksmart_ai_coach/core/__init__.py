"""
Core WorkSmart AI Coach components
"""

from .ai_coach import AICoach
from .telemetry_system import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer

__all__ = [
    "AICoach",
    "WorkSmartTelemetryCollector",
    "WorkSmartTelemetryAnalyzer",
]
