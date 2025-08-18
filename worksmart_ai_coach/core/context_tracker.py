#!/usr/bin/env python3
"""
Context History Tracker
=======================
Advanced context tracking for transition detection and momentum analysis.
Maintains real-time history of app switches, URL changes, and productivity states.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from dataclasses import dataclass, asdict

@dataclass
class ContextSnapshot:
    """Single context snapshot with all relevant data"""
    timestamp: str
    app_name: str
    window_title: str
    chrome_url: str = ""
    chrome_title: str = ""
    tab_count: int = 0
    productivity_score: float = 0.0
    focus_quality: float = 0.0
    activity_level: str = "MEDIUM"
    keyboard_activity: int = 0
    mouse_activity: int = 0
    session_duration_hours: float = 0.0

class ContextHistoryTracker:
    """Advanced context tracking for transition and momentum detection"""
    
    def __init__(self, max_history_minutes: int = 60):
        self.max_history_minutes = max_history_minutes
        self.context_history = deque(maxlen=360)  # 6 hours at 1-minute resolution
        self.transition_patterns = {}
        self.momentum_states = deque(maxlen=20)  # Last 20 states for momentum analysis
        
    def add_context(self, context_data: Dict[str, Any]) -> None:
        """Add new context snapshot to history"""
        snapshot = ContextSnapshot(
            timestamp=datetime.now().isoformat(),
            app_name=context_data.get('current_application', 'Unknown'),
            window_title=context_data.get('current_window', ''),
            chrome_url=context_data.get('chrome_context', {}).get('active_tab_url', ''),
            chrome_title=context_data.get('chrome_context', {}).get('active_tab_title', ''),
            tab_count=context_data.get('chrome_context', {}).get('total_tabs', 0),
            productivity_score=context_data.get('productivity_score', 0.0),
            focus_quality=context_data.get('focus_quality', 0.0),
            activity_level=context_data.get('activity_level', 'MEDIUM'),
            keyboard_activity=context_data.get('keyboard_count', 0),
            mouse_activity=context_data.get('mouse_count', 0),
            session_duration_hours=context_data.get('session_duration_hours', 0.0)
        )
        
        self.context_history.append(snapshot)
        self._update_momentum_tracking(snapshot)
        self._detect_transition_patterns()
    
    def _update_momentum_tracking(self, snapshot: ContextSnapshot) -> None:
        """Track momentum states for proactive detection"""
        # Classify current momentum
        if snapshot.productivity_score > 0.8 and snapshot.focus_quality > 0.8:
            momentum = "high_flow"
        elif snapshot.productivity_score > 0.6 and snapshot.focus_quality > 0.6:
            momentum = "building"
        elif snapshot.productivity_score < 0.4:
            momentum = "declining" 
        elif self._is_ai_tool_active(snapshot):
            momentum = "ai_engaged"
        elif self._is_meeting_active(snapshot):
            momentum = "meeting_focused"
        else:
            momentum = "neutral"
        
        self.momentum_states.append({
            'timestamp': snapshot.timestamp,
            'momentum': momentum,
            'context': {
                'app': snapshot.app_name,
                'url': snapshot.chrome_url,
                'tabs': snapshot.tab_count
            }
        })
    
    def _detect_transition_patterns(self) -> None:
        """Analyze context history for transition patterns"""
        if len(self.context_history) < 3:
            return
        
        # Analyze recent 3-snapshot patterns
        recent = list(self.context_history)[-3:]
        
        # App transition pattern
        app_pattern = " → ".join([s.app_name for s in recent])
        
        # URL transition pattern (for Chrome)
        url_pattern = " → ".join([self._categorize_url(s.chrome_url) for s in recent if s.chrome_url])
        
        # Productivity transition pattern
        prod_pattern = " → ".join([self._categorize_productivity(s.productivity_score) for s in recent])
        
        # Store patterns with timestamps
        pattern_key = datetime.now().strftime('%H:%M')
        self.transition_patterns[pattern_key] = {
            'app_pattern': app_pattern,
            'url_pattern': url_pattern,
            'productivity_pattern': prod_pattern,
            'tab_reduction': recent[0].tab_count - recent[-1].tab_count if len(recent) >= 2 else 0
        }
    
    def detect_post_meeting_momentum(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect transition from meeting to potentially productive state"""
        if len(self.context_history) < 5:
            return False, {}
        
        recent_5min = [s for s in self.context_history 
                      if (datetime.now() - datetime.fromisoformat(s.timestamp)).total_seconds() < 300]
        
        # Look for Safari meeting → other app transition
        meeting_to_work_transition = False
        transition_context = {}
        
        for i in range(len(recent_5min) - 1):
            current = recent_5min[i]
            next_snap = recent_5min[i + 1]
            
            # Meeting pattern: Safari with meeting keywords
            is_meeting = (current.app_name == 'Safari' and 
                         any(keyword in current.window_title.lower() 
                             for keyword in ['meet', 'zoom', 'call', 'conference']))
            
            # Transition to productive app
            next_is_productive = (next_snap.app_name in ['Terminal', 'ChatGPT'] or
                                'grok.com' in next_snap.chrome_url or
                                'chat.openai.com' in next_snap.chrome_url)
            
            if is_meeting and next_is_productive:
                meeting_to_work_transition = True
                transition_context = {
                    'meeting_duration_minutes': current.session_duration_hours * 60,
                    'transition_to': next_snap.app_name,
                    'transition_url': next_snap.chrome_url,
                    'meeting_quality_indicator': current.focus_quality
                }
                break
        
        return meeting_to_work_transition, transition_context
    
    def detect_focus_improvement_pattern(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect tab consolidation and focus improvement"""
        if len(self.context_history) < 4:
            return False, {}
        
        recent_10min = [s for s in self.context_history 
                       if (datetime.now() - datetime.fromisoformat(s.timestamp)).total_seconds() < 600]
        
        if len(recent_10min) < 3:
            return False, {}
        
        # Check for tab count reduction
        tab_counts = [s.tab_count for s in recent_10min if s.tab_count > 0]
        if len(tab_counts) < 3:
            return False, {}
        
        # Significant tab reduction (25%+ decrease)
        initial_tabs = tab_counts[0]
        final_tabs = tab_counts[-1]
        
        if initial_tabs > 8 and final_tabs < initial_tabs * 0.75:
            focus_improvement = True
            improvement_data = {
                'tab_reduction': initial_tabs - final_tabs,
                'improvement_percentage': ((initial_tabs - final_tabs) / initial_tabs) * 100,
                'duration_minutes': len(recent_10min),
                'productivity_trend': self._calculate_productivity_trend(recent_10min)
            }
            return focus_improvement, improvement_data
        
        return False, {}
    
    def detect_ai_recovery_sequence(self) -> Tuple[bool, Dict[str, Any]]:
        """Detect transition from distraction to AI tools"""
        if len(self.momentum_states) < 4:
            return False, {}
        
        recent_momentum = list(self.momentum_states)[-4:]
        
        # Look for declining → ai_engaged pattern
        momentum_sequence = [m['momentum'] for m in recent_momentum]
        
        if ('declining' in momentum_sequence and 
            momentum_sequence[-1] == 'ai_engaged'):
            
            recovery_data = {
                'recovery_sequence': momentum_sequence,
                'recovery_tool': recent_momentum[-1]['context']['app'],
                'recovery_url': recent_momentum[-1]['context']['url'],
                'recovery_duration_minutes': len(recent_momentum)
            }
            return True, recovery_data
        
        return False, {}
    
    def get_context_summary(self, minutes_back: int = 10) -> Dict[str, Any]:
        """Get context summary for specified time period"""
        cutoff = datetime.now() - timedelta(minutes=minutes_back)
        
        relevant_contexts = [s for s in self.context_history 
                           if datetime.fromisoformat(s.timestamp) > cutoff]
        
        if not relevant_contexts:
            return {}
        
        apps_used = list(set(s.app_name for s in relevant_contexts))
        urls_visited = list(set(s.chrome_url for s in relevant_contexts if s.chrome_url))
        
        return {
            'time_period_minutes': minutes_back,
            'snapshots_count': len(relevant_contexts),
            'apps_used': apps_used,
            'unique_urls': len(urls_visited),
            'url_categories': [self._categorize_url(url) for url in urls_visited],
            'avg_productivity': sum(s.productivity_score for s in relevant_contexts) / len(relevant_contexts),
            'avg_focus': sum(s.focus_quality for s in relevant_contexts) / len(relevant_contexts),
            'tab_count_trend': [s.tab_count for s in relevant_contexts if s.tab_count > 0],
            'latest_context': asdict(relevant_contexts[-1]) if relevant_contexts else None
        }
    
    def predict_next_optimal_action(self) -> Optional[Dict[str, Any]]:
        """Predict optimal next action based on context history"""
        if len(self.context_history) < 3:
            return None
        
        # Analyze recent patterns
        post_meeting_momentum, meeting_data = self.detect_post_meeting_momentum()
        focus_improvement, focus_data = self.detect_focus_improvement_pattern()
        ai_recovery, recovery_data = self.detect_ai_recovery_sequence()
        
        current_context = self.context_history[-1]
        
        # Prioritized predictions
        if post_meeting_momentum:
            return {
                'prediction_type': 'post_meeting_momentum',
                'confidence': 0.85,
                'suggested_action': 'Start AI-assisted creative work',
                'suggested_tools': ['ChatGPT', 'Grok'],
                'reasoning': 'Excellent meeting → AI productivity pattern detected',
                'context_data': meeting_data
            }
        
        if focus_improvement:
            return {
                'prediction_type': 'focus_building',
                'confidence': 0.75,
                'suggested_action': 'Continue focused work session',
                'suggested_tools': [current_context.app_name],
                'reasoning': f'Tab consolidation improving focus (+{focus_data["improvement_percentage"]:.1f}%)',
                'context_data': focus_data
            }
        
        if ai_recovery:
            return {
                'prediction_type': 'ai_recovery_momentum', 
                'confidence': 0.70,
                'suggested_action': 'Maintain AI tool engagement',
                'suggested_tools': [recovery_data['recovery_tool']],
                'reasoning': 'AI tool recovery sequence in progress',
                'context_data': recovery_data
            }
        
        # Fallback pattern-based prediction
        if current_context.productivity_score < 0.4 and current_context.tab_count > 10:
            return {
                'prediction_type': 'tab_overload_intervention',
                'confidence': 0.60,
                'suggested_action': 'Consolidate tabs and focus on single AI tool',
                'suggested_tools': ['Grok', 'ChatGPT'],
                'reasoning': 'Tab overload pattern detected',
                'context_data': {'tab_count': current_context.tab_count}
            }
        
        return None
    
    # Helper methods
    def _is_ai_tool_active(self, snapshot: ContextSnapshot) -> bool:
        """Check if AI tools are currently active"""
        ai_apps = ['ChatGPT', 'Terminal']
        ai_urls = ['grok.com', 'chat.openai.com', 'openai.com']
        
        return (snapshot.app_name in ai_apps or 
                any(url in snapshot.chrome_url for url in ai_urls))
    
    def _is_meeting_active(self, snapshot: ContextSnapshot) -> bool:
        """Check if in meeting context"""
        return (snapshot.app_name == 'Safari' and 
                any(keyword in snapshot.window_title.lower() 
                    for keyword in ['meet', 'zoom', 'call', 'conference']))
    
    def _categorize_url(self, url: str) -> str:
        """Categorize URL for pattern analysis"""
        if not url:
            return 'none'
        
        url_lower = url.lower()
        if any(ai in url_lower for ai in ['grok', 'openai', 'chatgpt']):
            return 'ai_tool'
        elif any(social in url_lower for social in ['linkedin', 'twitter', 'facebook']):
            return 'social'
        elif 'google.com/search' in url_lower:
            return 'search'
        elif any(meet in url_lower for meet in ['meet', 'zoom']):
            return 'meeting'
        else:
            return 'general'
    
    def _categorize_productivity(self, score: float) -> str:
        """Categorize productivity score for pattern analysis"""
        if score > 0.8:
            return 'high'
        elif score > 0.6:
            return 'medium'
        elif score > 0.4:
            return 'low'
        else:
            return 'very_low'
    
    def _calculate_productivity_trend(self, snapshots: List[ContextSnapshot]) -> str:
        """Calculate productivity trend over time"""
        if len(snapshots) < 2:
            return 'stable'
        
        scores = [s.productivity_score for s in snapshots]
        initial_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
        final_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        
        if final_avg > initial_avg + 0.1:
            return 'improving'
        elif final_avg < initial_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def save_context_history(self, filename: str) -> None:
        """Save context history to file"""
        history_data = {
            'saved_at': datetime.now().isoformat(),
            'context_history': [asdict(snapshot) for snapshot in self.context_history],
            'transition_patterns': self.transition_patterns,
            'momentum_states': list(self.momentum_states)
        }
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_context_history(self, filename: str) -> bool:
        """Load context history from file"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Restore context history
            for snapshot_dict in data.get('context_history', []):
                snapshot = ContextSnapshot(**snapshot_dict)
                self.context_history.append(snapshot)
            
            # Restore patterns and momentum
            self.transition_patterns = data.get('transition_patterns', {})
            self.momentum_states = deque(data.get('momentum_states', []), maxlen=20)
            
            return True
        except Exception as e:
            print(f"Error loading context history: {e}")
            return False