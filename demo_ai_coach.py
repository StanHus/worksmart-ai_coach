#!/usr/bin/env python3
"""
AI Coach Demo - Shows the actual working functionality with real data
"""
import sys
import asyncio
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "worksmart_ai_coach" / "core"))
from ai_coach import AICoach

def main():
    coach = AICoach(data_dir=str(Path.home() / ".worksmart-ai-coach"), test_mode=True)
    
    print("🚀 AI Coach Demo - 1 minute intervals")
    print("Uses YOUR real WorkSmart data and current app")
    print("Press Ctrl+C to stop\n")
    
    coach.start_test_mode(interval_minutes=1)

if __name__ == "__main__":
    main()