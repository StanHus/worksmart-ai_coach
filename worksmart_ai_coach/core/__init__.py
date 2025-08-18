"""
Core WorkSmart AI Coach components
"""

from .coach import AICoach
from .telemetry import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer

__all__ = [
    "AICoach",
    "WorkSmartTelemetryCollector",
    "WorkSmartTelemetryAnalyzer",
]