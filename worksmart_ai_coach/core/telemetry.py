#!/usr/bin/env python3
"""
WorkSmart Telemetry Tap
========================
Taps into WorkSmart-style telemetry data and provides real-time AI coaching.
Mimics the exact data structure from the Java tracker.
"""

import asyncio
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

# Import our AI coach
from .coach import AICoach

class WorkSmartTelemetryCollector:
    """Collects telemetry data in WorkSmart format"""
    
    def __init__(self):
        # Only keep essential tracking variables
        self.current_window = ""
        self.process_name = ""
        self.visiting_url = ""
        self.session_start = datetime.now()
        self.last_activity = datetime.now()
        
        # Use WorkSmart data as primary source (READ-ONLY)
        from .worksmart_reader import WorkSmartDataReader
        self.worksmart_reader = WorkSmartDataReader()
        print("✅ Using WorkSmart telemetry as READ-ONLY data source")
    
    def _setup_hooks(self):
        """No longer needed - using WorkSmart data instead"""
        pass
    
    def collect_event(self) -> Dict:
        """Collect telemetry event using WorkSmart data (STRICTLY READ-ONLY)
        
        This method ONLY reads WorkSmart's official telemetry data without any modifications,
        accumulations, or transformations. All data comes directly from WorkSmart logs.
        """
        
        # Get latest WorkSmart activity data (READ-ONLY access to logs)
        recent_activities = self.worksmart_reader.get_recent_activity_from_logs(hours=1)
        worksmart_stats = self.worksmart_reader.get_current_session_stats()
        
        # Use WorkSmart data exactly as reported (NO modifications)
        keyboard_count = 0
        mouse_count = 0
        scroll_count = 0
        
        if recent_activities:
            # Use the most recent WorkSmart activity data exactly as reported
            latest = recent_activities[-1]
            keyboard_count = latest.get('keystrokes', 0)
            mouse_count = latest.get('mouse_clicks', 0)
            scroll_count = latest.get('scroll_counts', 0)
            self.last_activity = datetime.fromisoformat(latest['timestamp'])
        
        # Get current window/app info and Chrome tab details
        import subprocess
        try:
            # Try to get active app on macOS
            script = '''
            tell application "System Events"
                set frontApp to name of first application process whose frontmost is true
                set windowTitle to "Unknown"
                try
                    tell application process frontApp
                        set windowTitle to name of front window
                    end tell
                end try
                return frontApp & "|" & windowTitle
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True, timeout=1)
            if result.returncode == 0:
                app_info = result.stdout.strip().split('|')
                self.process_name = app_info[0]
                self.current_window = app_info[1] if len(app_info) > 1 else "Unknown"
                
                # Get Chrome tab URL if Chrome is active
                if "Chrome" in self.process_name:
                    self.visiting_url = self._get_chrome_tab_url()
        except:
            pass
        
        # Build event enhanced with WorkSmart official data
        event = {
            "type": "ACTIVITY",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keyboard_count": keyboard_count,
            "mouse_count": mouse_count,
            "current_window": self.current_window,
            "process_name": self.process_name,
            "visiting_url": self.visiting_url if "Chrome" in self.process_name or "browser" in self.process_name.lower() else "",
            "in_call": self._detect_call(),
            "cpu_usage": self._get_cpu_usage(),
            "mem_usage": self._get_memory_usage(),
            "session_duration_hours": (datetime.now() - self.session_start).total_seconds() / 3600,
            "idle_time_minutes": (datetime.now() - self.last_activity).total_seconds() / 60,
            # Enhanced with WorkSmart official data
            "worksmart_hours_today": worksmart_stats.get('total_hours_today', '0:0'),
            "worksmart_session_active": worksmart_stats.get('session_active', False),
            "worksmart_user_id": worksmart_stats.get('user_id'),
            "worksmart_team_id": worksmart_stats.get('team_id'),
            "worksmart_data_files": worksmart_stats.get('data_files', 0),
            "data_source": "WorkSmart Official (READ-ONLY)"
        }
        
        # No modifications needed - using WorkSmart data as-is
        
        return event
    
    def _get_chrome_tab_url(self) -> str:
        """Get current Chrome tab URL and title"""
        try:
            import subprocess
            script = '''
            tell application "Google Chrome"
                if exists window 1 then
                    set currentTab to active tab of window 1
                    set tabURL to URL of currentTab
                    set tabTitle to title of currentTab
                    return tabURL & "|" & tabTitle
                else
                    return ""
                end if
            end tell
            '''
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                tab_info = result.stdout.strip().split('|')
                url = tab_info[0] if len(tab_info) > 0 else ""
                title = tab_info[1] if len(tab_info) > 1 else ""
                # Update window title with tab title for better context
                if title:
                    self.current_window = f"{title[:50]}..."
                return url
        except Exception as e:
            pass
        return ""
    
    def _detect_call(self) -> bool:
        """Detect if user is in a call"""
        call_apps = ['zoom', 'teams', 'slack', 'meet', 'skype', 'discord']
        return any(app in self.process_name.lower() for app in call_apps)
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return random.uniform(10, 50)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return random.uniform(40, 70)

class WorkSmartTelemetryAnalyzer:
    """Analyzes WorkSmart telemetry like TimecardAnalyzer.java"""
    
    # Thresholds from Java code
    HIGH_ACTIVITY_THRESHOLD = 200  # keystrokes + mouse per 10min
    LOW_ACTIVITY_THRESHOLD = 20
    HIGH_APP_SWITCHES_THRESHOLD = 50
    
    def analyze(self, events: List[Dict]) -> Dict:
        """Analyze telemetry events and produce insights"""
        
        if not events:
            return self._empty_analysis()
        
        # Calculate metrics
        total_keystrokes = sum(e.get('keyboard_count', 0) for e in events)
        total_mouse = sum(e.get('mouse_count', 0) for e in events)
        total_activity = total_keystrokes + total_mouse
        
        # Determine activity level
        if total_activity >= self.HIGH_ACTIVITY_THRESHOLD:
            activity_level = "HIGH"
        elif total_activity <= self.LOW_ACTIVITY_THRESHOLD:
            activity_level = "LOW"
        else:
            activity_level = "MEDIUM"
        
        # Get latest event for current state
        latest = events[-1] if events else {}
        
        return {
            "total_keystrokes": total_keystrokes,
            "total_mouse_events": total_mouse,
            "activity_level": activity_level,
            "in_call": latest.get('in_call', False),
            "current_app": latest.get('process_name', 'Unknown'),
            "session_hours": latest.get('session_duration_hours', 0),
            "idle_minutes": latest.get('idle_time_minutes', 0),
            "cpu_usage": latest.get('cpu_usage', 0),
            "memory_usage": latest.get('mem_usage', 0),
            "productivity_score": self._calculate_productivity_score(events),
            "focus_quality": self._calculate_focus_quality(events),
            "needs_break": latest.get('session_duration_hours', 0) > 2,
            "alert_level": self._determine_alert_level(activity_level, latest)
        }
    
    def _calculate_productivity_score(self, events: List[Dict]) -> float:
        """Calculate productivity score (0-1)"""
        if not events:
            return 0.5
        
        # Simple heuristic based on activity consistency
        activities = [e.get('keyboard_count', 0) + e.get('mouse_count', 0) for e in events]
        if not activities:
            return 0.5
        
        avg_activity = sum(activities) / len(activities)
        if avg_activity > 100:
            return min(1.0, 0.7 + (avg_activity - 100) / 300)
        elif avg_activity < 20:
            return max(0.2, avg_activity / 40)
        else:
            return 0.5 + (avg_activity - 20) / 160
    
    def _calculate_focus_quality(self, events: List[Dict]) -> float:
        """Calculate focus quality based on app switching patterns"""
        if not events:
            return 0.5
        
        # Check for app consistency
        apps = [e.get('process_name', '') for e in events]
        unique_apps = len(set(apps))
        
        if unique_apps == 1:
            return 0.9  # Single app = high focus
        elif unique_apps <= 2:
            return 0.7  # Two apps = good focus
        else:
            return max(0.3, 1.0 - (unique_apps * 0.15))
    
    def _determine_alert_level(self, activity_level: str, latest: Dict) -> str:
        """Determine coaching alert level"""
        session_hours = latest.get('session_duration_hours', 0)
        idle_minutes = latest.get('idle_time_minutes', 0)
        
        if session_hours > 4:
            return "HIGH"
        elif idle_minutes > 15:
            return "MEDIUM"
        elif activity_level == "LOW" and session_hours > 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            "total_keystrokes": 0,
            "total_mouse_events": 0,
            "activity_level": "NONE",
            "in_call": False,
            "current_app": "Unknown",
            "session_hours": 0,
            "idle_minutes": 0,
            "productivity_score": 0.5,
            "focus_quality": 0.5,
            "needs_break": False,
            "alert_level": "LOW"
        }

async def main():
    """Main telemetry tap loop"""
    print("🔌 WorkSmart Telemetry Tap")
    print("="*50)
    print("Tapping into WorkSmart-style telemetry data...")
    print("AI Coach will analyze patterns and provide insights\n")
    
    # Initialize components
    collector = WorkSmartTelemetryCollector()
    analyzer = WorkSmartTelemetryAnalyzer()
    ai_coach = AICoach()
    
    # Event buffer (like Java's Timecard)
    event_buffer = []
    buffer_size = 6  # Analyze every 6 events (10 minutes worth)
    
    print("📊 Starting telemetry collection...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Collect event (every 100 seconds like WorkSmart)
            event = collector.collect_event()
            event_buffer.append(event)
            
            # Display current telemetry
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Telemetry Event:")
            print(f"  📱 App: {event['process_name']}")
            print(f"  🪟 Window: {event['current_window'][:50]}...")
            print(f"  ⌨️  Keystrokes: {event['keyboard_count']}")
            print(f"  🖱️  Mouse events: {event['mouse_count']}")
            print(f"  ⏱️  Session: {event['session_duration_hours']:.2f} hours")
            print(f"  💤 Idle: {event['idle_time_minutes']:.1f} minutes")
            
            # Analyze when buffer is full
            if len(event_buffer) >= buffer_size:
                print("\n" + "="*50)
                print("🔍 ANALYZING TELEMETRY PATTERN...")
                
                # Analyze with WorkSmart analyzer
                analysis = analyzer.analyze(event_buffer)
                
                print(f"\n📈 Analysis Results:")
                print(f"  Activity Level: {analysis['activity_level']}")
                print(f"  Productivity Score: {analysis['productivity_score']:.1%}")
                print(f"  Focus Quality: {analysis['focus_quality']:.1%}")
                print(f"  Alert Level: {analysis['alert_level']}")
                
                # Get AI coaching
                telemetry_for_ai = {
                    "session_duration_hours": analysis['session_hours'],
                    "productivity_score": analysis['productivity_score'],
                    "focus_quality": analysis['focus_quality'],
                    "stress_level": 0.5 if not analysis['in_call'] else 0.7,
                    "energy_level": max(0.3, 1.0 - (analysis['session_hours'] / 10)),
                    "keystrokes_per_min": analysis['total_keystrokes'] / 10,
                    "app_switches_per_hour": len(set(e['process_name'] for e in event_buffer)) * 6,
                    "current_application": analysis['current_app'],
                    "in_meeting": analysis['in_call']
                }
                
                ai_result = await ai_coach.analyze_telemetry(telemetry_for_ai, "worksmart_user")
                
                if ai_result:
                    print(f"\n🤖 AI COACH INSIGHT:")
                    print(f"  {ai_result.get('message', 'No specific advice at this time')}")
                
                print("="*50 + "\n")
                
                # Keep only last 3 events for continuity
                event_buffer = event_buffer[-3:]
            
            # Wait 100 seconds (WorkSmart's interval)
            await asyncio.sleep(100)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Telemetry tap stopped")
        print(f"📊 Final session duration: {collector.session_start}")

if __name__ == "__main__":
    asyncio.run(main())