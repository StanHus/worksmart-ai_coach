"""
WorkSmart AI Coach - AI-powered productivity coaching for WorkSmart tracker
"""

__version__ = "1.0.0"
__author__ = "WorkSmart AI Team"
__email__ = "support@worksmart.ai"

from .core.coach import AICoach
from .core.telemetry import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer

__all__ = [
    "AICoach",
    "WorkSmartTelemetryCollector", 
    "WorkSmartTelemetryAnalyzer",
]