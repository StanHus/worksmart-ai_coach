#!/usr/bin/env python3
"""
WorkSmart Data Reader
====================
READ-ONLY analysis of WorkSmart telemetry data and logs.
This module safely reads and parses WorkSmart data without modification.
"""

import os
import json
import base64
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class WorkSmartDataReader:
    """Safe READ-ONLY reader for WorkSmart telemetry data"""
    
    def __init__(self, crossover_files_path: str = None):
        if crossover_files_path is None:
            crossover_files_path = os.getenv('CROSSOVER_FILES_PATH', os.path.expanduser('~/crossoverFiles'))
        self.base_path = Path(crossover_files_path)
        self.logs_path = self.base_path / "logs"
        self.data_path = self.base_path / "DataCapture"
        self.config_path = self.base_path / "config"
        
    def get_recent_activity_from_logs(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Extract recent activity data from WorkSmart logs"""
        activities = []
        
        try:
            log_file = self.logs_path / "deskapp.log"
            if not log_file.exists():
                return activities
                
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Look for ActivityJob entries
            for line in reversed(lines[-1000:]):  # Check last 1000 lines
                if "ActivityJob" in line and "Counted" in line:
                    # Parse: "Counted 4 key press , 1 mouse clicks and 0 scroll counts until 2025-08-18 17:18:06"
                    match = re.search(r'Counted (\d+) key press , (\d+) mouse clicks and (\d+) scroll counts until ([\d-]+ [\d:]+)', line)
                    if match:
                        timestamp_str = match.group(4)
                        try:
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            activities.append({
                                'timestamp': timestamp.isoformat(),
                                'keystrokes': int(match.group(1)),
                                'mouse_clicks': int(match.group(2)), 
                                'scroll_counts': int(match.group(3)),
                                'source': 'WorkSmart ActivityJob'
                            })
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"⚠️ Could not read WorkSmart activity logs: {e}")
            
        return activities[-20:]  # Return last 20 activities
    
    def get_global_data_from_logs(self) -> Dict[str, Any]:
        """Extract global productivity data from logs"""
        global_data = {}
        
        try:
            log_file = self.logs_path / "deskapp.log"
            if not log_file.exists():
                return global_data
                
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Look for GlobalData entries
            for line in reversed(lines[-500:]):  # Check recent entries
                if "GlobalData" in line and "hours today:" in line:
                    # Parse: "User: 37326, team: 5292, timezone: +01:00, hours today: 7:40, hours this week: 7:40"
                    match = re.search(r'User: (\d+), team: (\d+), timezone: ([^,]+), hours today: ([\d:]+), hours this week: ([\d:]+)', line)
                    if match:
                        global_data = {
                            'user_id': int(match.group(1)),
                            'team_id': int(match.group(2)),
                            'timezone': match.group(3),
                            'hours_today': match.group(4),
                            'hours_this_week': match.group(5),
                            'last_update': datetime.now().isoformat(),
                            'source': 'WorkSmart GlobalData'
                        }
                        break
                        
        except Exception as e:
            print(f"⚠️ Could not read WorkSmart global data: {e}")
            
        return global_data
    
    def list_data_capture_sessions(self) -> List[Dict[str, Any]]:
        """List available data capture sessions"""
        sessions = []
        
        try:
            if not self.data_path.exists():
                return sessions
                
            for session_dir in self.data_path.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith('Data_'):
                    # Parse directory name: Data_08_18_25_17_10_00
                    session_info = {
                        'session_dir': str(session_dir),
                        'session_name': session_dir.name,
                        'files': []
                    }
                    
                    # List files in session
                    for file_path in session_dir.iterdir():
                        session_info['files'].append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'size': file_path.stat().st_size,
                            'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                    
                    sessions.append(session_info)
                    
        except Exception as e:
            print(f"⚠️ Could not list data capture sessions: {e}")
            
        return sessions
    
    def get_current_session_stats(self) -> Dict[str, Any]:
        """Get statistics for the current session"""
        stats = {
            'session_active': False,
            'data_files': 0,
            'latest_activity': None,
            'total_hours_today': '0:0'
        }
        
        # Get latest session
        sessions = self.list_data_capture_sessions()
        if sessions:
            latest_session = max(sessions, key=lambda x: x['session_name'])
            stats['session_active'] = True
            stats['data_files'] = len(latest_session['files'])
            stats['latest_session'] = latest_session['session_name']
        
        # Get hours from global data
        global_data = self.get_global_data_from_logs()
        if global_data:
            stats['total_hours_today'] = global_data.get('hours_today', '0:0')
            stats['user_id'] = global_data.get('user_id')
            stats['team_id'] = global_data.get('team_id')
        
        # Get recent activity
        activities = self.get_recent_activity_from_logs(hours=1)
        if activities:
            stats['latest_activity'] = activities[-1]
            stats['recent_keystrokes'] = sum(a['keystrokes'] for a in activities[-5:])
            stats['recent_mouse_clicks'] = sum(a['mouse_clicks'] for a in activities[-5:])
        
        return stats
    
    def analyze_productivity_patterns(self, hours: int = 8) -> Dict[str, Any]:
        """Analyze productivity patterns from WorkSmart data"""
        activities = self.get_recent_activity_from_logs(hours=hours)
        
        if not activities:
            return {'analysis': 'No activity data available'}
        
        total_keystrokes = sum(a['keystrokes'] for a in activities)
        total_clicks = sum(a['mouse_clicks'] for a in activities)
        total_scrolls = sum(a['scroll_counts'] for a in activities)
        
        analysis = {
            'time_period_hours': hours,
            'total_activities': len(activities),
            'total_keystrokes': total_keystrokes,
            'total_mouse_clicks': total_clicks,
            'total_scrolls': total_scrolls,
            'average_keystrokes_per_minute': round(total_keystrokes / (hours * 60), 2) if hours > 0 else 0,
            'activity_distribution': {
                'keyboard_heavy': total_keystrokes > total_clicks * 5,
                'mouse_heavy': total_clicks > total_keystrokes,
                'balanced': abs(total_keystrokes - total_clicks * 3) < total_keystrokes * 0.3
            },
            'latest_activity_time': activities[-1]['timestamp'] if activities else None
        }
        
        return analysis


def test_worksmart_reader():
    """Test function to validate WorkSmart reader"""
    print("🔍 Testing WorkSmart Data Reader...")
    
    reader = WorkSmartDataReader()
    
    # Test current session stats
    stats = reader.get_current_session_stats()
    print(f"📊 Current Session Stats:")
    print(f"   Active: {stats['session_active']}")
    print(f"   Hours today: {stats['total_hours_today']}")
    print(f"   Data files: {stats['data_files']}")
    
    # Test recent activity
    activities = reader.get_recent_activity_from_logs(hours=2)
    print(f"📈 Recent Activities: {len(activities)} entries")
    if activities:
        latest = activities[-1]
        print(f"   Latest: {latest['keystrokes']}🔤 {latest['mouse_clicks']}🖱️ at {latest['timestamp']}")
    
    # Test productivity analysis
    analysis = reader.analyze_productivity_patterns(hours=8)
    print(f"🎯 Productivity Analysis:")
    print(f"   Total keystrokes: {analysis.get('total_keystrokes', 0)}")
    print(f"   Total clicks: {analysis.get('total_mouse_clicks', 0)}")
    print(f"   Avg keystrokes/min: {analysis.get('average_keystrokes_per_minute', 0)}")


if __name__ == "__main__":
    test_worksmart_reader()