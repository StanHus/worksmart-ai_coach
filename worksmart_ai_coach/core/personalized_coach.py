#!/usr/bin/env python3
"""
Personalized AI Coach
====================
AI coaching engine with personalized algorithms based on user feedback patterns.
Uses actual productivity correlations instead of generic advice.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import asdict
from .coach import AICoach
from .context_tracker import ContextHistoryTracker
from .micro_interventions import MicroInterventionSystem
from .adaptive_learning import AdaptiveLearningSystem
from .worksmart_reader import WorkSmartDataReader

class PersonalizedAICoach(AICoach):
    """AI Coach with personalized algorithms based on learned patterns"""
    
    def __init__(self):
        super().__init__()
        self.personal_patterns = self._load_personal_patterns()
        self.intervention_history = []
        self.context_tracker = ContextHistoryTracker()
        self.micro_interventions = MicroInterventionSystem()
        self.adaptive_learning = AdaptiveLearningSystem()
        self.worksmart_reader = WorkSmartDataReader()
        
        # Track current intervention for follow-up
        self.current_intervention_id = None
        self.intervention_timestamp = None
        
        # Load existing context history if available
        context_history_file = "context_history.json"
        self.context_tracker.load_context_history(context_history_file)
        
    def _load_personal_patterns(self) -> Dict[str, Any]:
        """Load learned personal productivity patterns"""
        # Try to load consolidated learning data
        patterns_file = "consolidated_learning_data_20250818.json"
        
        if os.path.exists(patterns_file):
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                return self._extract_patterns_from_data(data)
            except Exception as e:
                print(f"⚠️ Could not load personal patterns: {e}")
        
        # Fallback to discovered patterns
        return self._get_discovered_patterns()
    
    def _extract_patterns_from_data(self, data: Dict) -> Dict[str, Any]:
        """Extract productivity patterns from consolidated feedback data"""
        feedback_entries = data.get('feedback_data', [])
        activity_entries = data.get('activity_data', [])
        
        patterns = {
            # High productivity correlations
            'ai_tools': ['grok.com', 'chat.openai.com', 'ChatGPT', 'Terminal'],
            'productive_apps': ['Terminal', 'ChatGPT', 'Safari'],  # when in meetings
            'optimal_tab_count': 9,
            'tab_overload_threshold': 10,
            
            # Low productivity correlations  
            'distraction_signals': ['linkedin.com', 'how to grow your substack'],
            'unproductive_patterns': ['general google searches', 'social media browsing'],
            
            # Intervention timing
            'unproductive_intervention_minutes': 5,
            'optimal_focus_duration_minutes': 17,  # average from patterns
            'meeting_to_ai_momentum_minutes': 3,
            
            # Personal productivity states
            'productivity_states': {
                'unproductive': ['being non productive', 'non productive', 'unproductive'],
                'moderate': ['moderately productive', 'moderate productivity'],
                'high': ['very productive', 'superb', 'fantastic', 'amazing', 'very strong'],
                'ai_flow': ['ai engagement', 'having ideas', 'prompting ai'],
                'meeting_excellence': ['engaged in meeting', 'superb meeting', 'fantastic meeting']
            }
        }
        
        # Analyze actual feedback patterns
        productive_contexts = []
        unproductive_contexts = []
        
        for entry in feedback_entries:
            feedback = entry.get('feedback_text', '').lower()
            activity = entry.get('activity_at_feedback', {})
            
            if any(term in feedback for term in ['productive', 'superb', 'fantastic', 'amazing', 'engaged']):
                productive_contexts.append(activity)
            elif any(term in feedback for term in ['non productive', 'unproductive', 'distracted']):
                unproductive_contexts.append(activity)
        
        patterns['learned_productive_contexts'] = productive_contexts[:5]  # Top 5
        patterns['learned_unproductive_contexts'] = unproductive_contexts[:5]
        
        return patterns
    
    def _get_discovered_patterns(self) -> Dict[str, Any]:
        """Fallback patterns discovered from analysis"""
        return {
            'ai_tools': ['grok.com', 'chat.openai.com', 'openai.com', 'ChatGPT', 'Terminal'],
            'productive_apps': ['Terminal', 'ChatGPT', 'Safari'],
            'optimal_tab_count': 9,
            'tab_overload_threshold': 10,
            'distraction_signals': ['linkedin.com', 'google.com/search'],
            'unproductive_intervention_minutes': 5,
            'optimal_focus_duration_minutes': 17,
            'meeting_to_ai_momentum_minutes': 3
        }
    
    def calculate_personalized_productivity_score(self, events: List[Dict], context: Dict) -> float:
        """Calculate productivity score using personal correlations"""
        base_score = self._calculate_base_activity_score(events)
        
        # Personal productivity multipliers
        ai_tools_active = self._detect_ai_tools(context)
        tab_count = self._get_tab_count(context)
        file_activity = context.get('file_activity', {}).get('file_activity_detected', False)
        meeting_context = self._detect_meeting_context(context)
        social_browsing = self._detect_social_browsing(context)
        
        print(f"🔍 Personal scoring: base={base_score:.2f}, ai_tools={ai_tools_active}, tabs={tab_count}")
        
        # Apply personal correlations
        if ai_tools_active and tab_count <= self.personal_patterns['optimal_tab_count']:
            score = min(1.0, base_score * 1.3)  # 30% boost for AI tools + focused tabs
            print(f"   ✅ AI tools + focused tabs boost: {score:.2f}")
            return score
        
        if meeting_context:
            score = min(1.0, base_score * 1.2)  # 20% boost for meetings
            print(f"   📞 Meeting context boost: {score:.2f}")
            return score
        
        if tab_count > self.personal_patterns['tab_overload_threshold']:
            score = max(0.3, base_score * 0.7)  # 30% penalty for tab overload
            print(f"   ⚠️ Tab overload penalty: {score:.2f}")
            return score
        
        if social_browsing:
            score = max(0.2, base_score * 0.6)  # 40% penalty for social media
            print(f"   📱 Social browsing penalty: {score:.2f}")
            return score
        
        if file_activity:
            score = min(1.0, base_score * 1.1)  # 10% boost for file activity
            print(f"   📁 File activity boost: {score:.2f}")
            return score
        
        print(f"   📊 Base score unchanged: {base_score:.2f}")
        return base_score
    
    def calculate_personalized_focus_quality(self, events: List[Dict], context: Dict) -> float:
        """Calculate focus quality using personal patterns"""
        unique_apps = len(set(e.get('process_name', '') for e in events))
        tab_count = self._get_tab_count(context)
        
        print(f"🎯 Personal focus: apps={unique_apps}, tabs={tab_count}")
        
        # Personal focus patterns
        if self._detect_ai_flow_state(context):
            print(f"   🤖 AI flow state detected: 0.95")
            return 0.95  # Maximum focus during AI engagement
        
        if self._detect_meeting_excellence(context):
            print(f"   📞 Meeting excellence detected: 0.90")
            return 0.9  # High focus during meetings
        
        # Tab-aware focus calculation
        if unique_apps <= 2 and tab_count <= self.personal_patterns['optimal_tab_count']:
            score = 0.85
            print(f"   ✅ Good focus with reasonable tabs: {score}")
            return score
        
        if tab_count > self.personal_patterns['tab_overload_threshold'] + 2:
            score = max(0.3, 0.8 - (tab_count * 0.03))
            print(f"   ⚠️ Tab overload focus penalty: {score}")
            return score
        
        # Default calculation with personal adjustments
        base_focus = max(0.3, 1.0 - (unique_apps * 0.12))
        print(f"   📊 Base focus calculation: {base_focus}")
        return base_focus
    
    def should_intervene_personalized(self, current_state: str, duration_minutes: float, context: Dict) -> Tuple[bool, str, str]:
        """Determine if intervention is needed using personal patterns + WorkSmart data"""
        
        # Get WorkSmart productivity metrics
        worksmart_stats = self.worksmart_reader.get_current_session_stats()
        recent_activities = self.worksmart_reader.get_recent_activity_from_logs(hours=1)
        
        # WorkSmart-enhanced interventions: Multiple intelligence layers
        if recent_activities:
            latest = recent_activities[-1]
            recent_keystrokes = latest.get('keystrokes', 0)
            recent_clicks = latest.get('mouse_clicks', 0)
            
            # Very low activity pattern (WorkSmart shows minimal engagement)
            if recent_keystrokes < 3 and recent_clicks < 5 and duration_minutes >= 3:
                return True, "gentle_engagement", "worksmart_low_activity_detected"
            
            # High activity but potentially distracted (lots of clicks, few keystrokes)
            if recent_clicks > 20 and recent_keystrokes < 10 and duration_minutes >= 4:
                return True, "focus_coaching", "worksmart_high_click_low_type_pattern"
        
        # WorkSmart hours-based intervention (encourage breaks)
        hours_today = worksmart_stats.get('total_hours_today', '0:0')
        if ':' in hours_today:
            try:
                hours, minutes = hours_today.split(':')
                total_minutes = int(hours) * 60 + int(minutes)
                
                # After 4+ hours continuous work, suggest strategic break
                if total_minutes > 240 and duration_minutes >= 7:
                    return True, "strategic_break", "worksmart_extended_work_session"
            except:
                pass
        
        # Personal pattern: 5-minute rule for unproductive states  
        if current_state == 'unproductive' and duration_minutes >= self.personal_patterns['unproductive_intervention_minutes']:
            return True, "immediate", "5_minute_unproductive_rule"
        
        # Detect tab overload early
        if self._detect_tab_overload(context) and duration_minutes >= 3:
            return True, "gentle_redirect", "tab_overload_detected"
        
        # Post-meeting momentum opportunity
        if self._detect_post_meeting_transition(context):
            return True, "momentum_suggestion", "post_meeting_momentum"
        
        # AI flow state protection (minimal interruptions)
        if self._detect_ai_flow_state(context):
            return False, "protect_flow", "ai_flow_protection"
        
        # Natural pondering phase (respect the process)
        if current_state == 'pondering':
            return False, "natural_process", "pondering_respect"
        
        # Social browsing intervention
        if self._detect_social_browsing(context) and duration_minutes >= 4:
            return True, "redirect", "social_browsing_detected"
        
        return False, "continue_monitoring", "no_intervention_needed"
    
    async def generate_personalized_coaching(self, context: Dict, analysis: Dict) -> Optional[Dict]:
        """Generate coaching message using personal patterns and context history"""
        
        productivity_score = analysis.get('productivity_score', 0.5)
        focus_quality = analysis.get('focus_quality', 0.5)
        current_app = analysis.get('current_app', 'Unknown')
        duration_minutes = (analysis.get('session_hours', 0) * 60)
        
        # Add context to history tracker
        context_with_analysis = {**context, **analysis}
        self.context_tracker.add_context(context_with_analysis)
        
        # Enhance with WorkSmart official data
        worksmart_stats = self.worksmart_reader.get_current_session_stats()
        worksmart_analysis = self.worksmart_reader.analyze_productivity_patterns(hours=2)
        
        # Merge WorkSmart data into analysis
        analysis['worksmart_hours_today'] = worksmart_stats.get('total_hours_today', '0:0')
        analysis['worksmart_session_active'] = worksmart_stats.get('session_active', False)
        analysis['worksmart_recent_keystrokes'] = worksmart_stats.get('recent_keystrokes', 0)
        analysis['worksmart_official_kpm'] = worksmart_analysis.get('average_keystrokes_per_minute', 0)
        
        # Update adaptive learning if we have a previous intervention to track
        if self.current_intervention_id and self.intervention_timestamp:
            minutes_elapsed = (datetime.now() - self.intervention_timestamp).total_seconds() / 60
            self.adaptive_learning.update_outcome(
                self.current_intervention_id, 
                productivity_score, 
                focus_quality,
                context,
                int(minutes_elapsed)
            )
        
        # Check for advanced patterns first
        post_meeting_momentum, meeting_data = self.context_tracker.detect_post_meeting_momentum()
        focus_improvement, focus_data = self.context_tracker.detect_focus_improvement_pattern()  
        ai_recovery, recovery_data = self.context_tracker.detect_ai_recovery_sequence()
        
        # Get predictive analysis
        prediction = self.context_tracker.predict_next_optimal_action()
        
        print(f"🤖 Advanced analysis: meeting_momentum={post_meeting_momentum}, focus_improvement={focus_improvement}, ai_recovery={ai_recovery}")
        if prediction:
            print(f"🔮 Prediction: {prediction['prediction_type']} (confidence: {prediction['confidence']:.2f})")
        
        # Priority 1: Handle advanced pattern-based interventions
        if post_meeting_momentum and meeting_data.get('meeting_quality_indicator', 0) > 0.7:
            intervention_data = {
                "message": f"Perfect timing! Your {meeting_data['meeting_duration_minutes']:.0f}-minute meeting showed excellent focus. This is the ideal moment for AI-assisted creative work. The momentum transfer window is now open.",
                "action": "momentum_capitalize",
                "priority": 1,
                "suggested_tools": ["ChatGPT", "Grok"],
                "reasoning": "Post-meeting excellence momentum detected",
                "context_data": meeting_data
            }
            
            # Track intervention for adaptive learning
            self.current_intervention_id = self.adaptive_learning.track_intervention(
                intervention_data, context, analysis
            )
            self.intervention_timestamp = datetime.now()
            
            return intervention_data
        
        if focus_improvement and focus_data.get('improvement_percentage', 0) > 30:
            return {
                "message": f"Excellent focus building! You've reduced tabs by {focus_data['tab_reduction']} ({focus_data['improvement_percentage']:.1f}% improvement). Your productivity trend is {focus_data['productivity_trend']}. Keep this focus session going.",
                "action": "focus_protection",
                "priority": 1,
                "suggested_tools": [current_app],
                "reasoning": "Significant focus improvement detected",
                "context_data": focus_data
            }
        
        if ai_recovery and len(recovery_data.get('recovery_sequence', [])) >= 3:
            return {
                "message": f"Strong AI recovery sequence detected! You've successfully transitioned through {' → '.join(recovery_data['recovery_sequence'])} using {recovery_data['recovery_tool']}. Maintain this positive momentum.",
                "action": "recovery_reinforcement", 
                "priority": 2,
                "suggested_tools": [recovery_data['recovery_tool']],
                "reasoning": "AI tool recovery pattern confirmed",
                "context_data": recovery_data
            }
        
        # Priority 2: Use predictive analysis for proactive coaching
        if prediction and prediction['confidence'] > 0.75:
            intervention_data = {
                "message": f"{prediction['suggested_action']}. Confidence: {prediction['confidence']:.0%}. {prediction['reasoning']}",
                "action": prediction['prediction_type'],
                "priority": 2 if prediction['confidence'] > 0.8 else 3,
                "suggested_tools": prediction.get('suggested_tools', []),
                "reasoning": f"Predictive analysis: {prediction['reasoning']}",
                "context_data": prediction.get('context_data', {})
            }
            
            # Check adaptive learning recommendations
            adaptation_rec = self.adaptive_learning.get_adaptation_recommendation(
                prediction['prediction_type'], context, analysis
            )
            
            if adaptation_rec:
                predicted_effectiveness = adaptation_rec['predicted_effectiveness']
                print(f"🧠 Adaptation recommendation: {predicted_effectiveness:.0%} predicted effectiveness")
                
                # Only proceed if predicted effectiveness is reasonable
                if predicted_effectiveness > 0.4:
                    # Track intervention
                    self.current_intervention_id = self.adaptive_learning.track_intervention(
                        intervention_data, context, analysis
                    )
                    self.intervention_timestamp = datetime.now()
                    
                    # Enhance message with learning insights
                    if adaptation_rec.get('recommendations'):
                        intervention_data['learning_notes'] = adaptation_rec['recommendations']
                    
                    return intervention_data
                else:
                    print(f"🚫 Skipping intervention due to low predicted effectiveness: {predicted_effectiveness:.0%}")
            else:
                # No learning data yet, proceed normally
                self.current_intervention_id = self.adaptive_learning.track_intervention(
                    intervention_data, context, analysis
                )
                self.intervention_timestamp = datetime.now()
                return intervention_data
        
        # Priority 2.5: Check for micro-interventions (2-3 minute gentle nudges)
        context_history = [asdict(snapshot) for snapshot in self.context_tracker.context_history]
        micro_intervention = self.micro_interventions.evaluate_micro_intervention(
            context, analysis, context_history
        )
        
        if micro_intervention:
            print(f"🔔 Micro-intervention: {micro_intervention['title']} (intensity: {micro_intervention['intensity']})")
            return {
                "message": micro_intervention['message'],
                "action": micro_intervention['type'],
                "priority": micro_intervention['notification_priority'],
                "reasoning": micro_intervention['reasoning'],
                "intervention_type": "micro",
                "intensity": micro_intervention['intensity'],
                "should_notify": micro_intervention['should_notify']
            }
        
        # Priority 3: Fall back to standard personalized logic
        current_state = self._classify_current_state(analysis)
        should_intervene, intervention_type, reason = self.should_intervene_personalized(
            current_state, duration_minutes, context
        )
        
        print(f"🤖 Standard coaching: state={current_state}, intervene={should_intervene}, reason={reason}")
        
        if not should_intervene:
            # Save context history before returning
            today = datetime.now().strftime('%Y-%m-%d')
            self.context_tracker.save_context_history("context_history.json")
            return None
        
        # Generate standard personalized message
        message_data = self._generate_personalized_message(current_state, context, analysis, reason)
        
        # Enhanced logging with context history
        context_summary = self.context_tracker.get_context_summary(minutes_back=10)
        
        self.intervention_history.append({
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'intervention_type': intervention_type,
            'message': message_data.get('message', ''),
            'context': {
                'current_app': current_app,
                'productivity_score': productivity_score,
                'focus_quality': focus_quality,
                'tab_count': self._get_tab_count(context)
            },
            'context_summary': context_summary,
            'advanced_patterns': {
                'post_meeting_momentum': post_meeting_momentum,
                'focus_improvement': focus_improvement,
                'ai_recovery': ai_recovery
            }
        })
        
        # Save context history and adaptive learning data
        today = datetime.now().strftime('%Y-%m-%d')
        self.context_tracker.save_context_history("context_history.json")
        self.adaptive_learning.save_learning_data()
        
        return message_data
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning and effectiveness summary"""
        learning_summary = self.adaptive_learning.get_learning_summary()
        context_summary = self.context_tracker.get_context_summary(minutes_back=30)
        micro_effectiveness = self.micro_interventions.get_intervention_effectiveness()
        
        return {
            'adaptive_learning': learning_summary,
            'context_tracking': {
                'snapshots_tracked': len(self.context_tracker.context_history),
                'recent_context': context_summary
            },
            'micro_interventions': micro_effectiveness,
            'personal_patterns_loaded': len(self.personal_patterns),
            'intervention_history': len(self.intervention_history)
        }
    
    def cleanup_and_save(self) -> None:
        """Cleanup and save all learning data"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Save all systems
        self.context_tracker.save_context_history("context_history.json")
        self.adaptive_learning.save_learning_data()
        
        # Print learning summary
        summary = self.get_learning_summary()
        print("🎓 LEARNING SUMMARY:")
        print(f"   Adaptive learning: {summary['adaptive_learning'].get('status', 'inactive')}")
        
        if 'total_outcomes' in summary['adaptive_learning']:
            print(f"   Tracked outcomes: {summary['adaptive_learning']['total_outcomes']}")
            print(f"   Avg effectiveness: {summary['adaptive_learning']['avg_effectiveness']:.2f}")
        
        print(f"   Context snapshots: {summary['context_tracking']['snapshots_tracked']}")
        print(f"   Micro-interventions: {summary['micro_interventions']['total_interventions']}")
        print(f"   Personal patterns: {summary['personal_patterns_loaded']}")
        
        print("💾 All learning data saved successfully")
    
    def _generate_personalized_message(self, state: str, context: Dict, analysis: Dict, reason: str) -> Dict:
        """Generate context-aware personalized coaching message"""
        
        tab_count = self._get_tab_count(context)
        current_app = analysis.get('current_app', 'Unknown')
        
        if reason == "5_minute_unproductive_rule":
            if tab_count > self.personal_patterns['tab_overload_threshold']:
                return {
                    "message": f"Noticed {tab_count} Chrome tabs open with general browsing for 5+ minutes. Try focusing on Grok or ChatGPT for the next 15 minutes? These tools consistently help you get back into productive flow.",
                    "action": "ai_tool_suggestion",
                    "priority": 2,
                    "suggested_tools": ["Grok", "ChatGPT"],
                    "reasoning": "Personal pattern: AI tools pull you out of unproductive states + tab consolidation needed"
                }
            else:
                return {
                    "message": "You've been unproductive for 5 minutes. Based on your patterns, switching to Grok or ChatGPT usually gets you back into productive flow within 2-3 minutes.",
                    "action": "ai_tool_focus",
                    "priority": 2,
                    "suggested_tools": ["Grok", "ChatGPT"],
                    "reasoning": "Personal 5-minute rule + AI tool recovery pattern"
                }
        
        elif reason == "tab_overload_detected":
            return {
                "message": f"Detected {tab_count} Chrome tabs open. Your optimal focus happens with ≤9 tabs. Consider consolidating or switching to a single AI tool for better concentration.",
                "action": "tab_consolidation",
                "priority": 2,
                "reasoning": "Personal optimal tab count pattern"
            }
        
        elif reason == "post_meeting_momentum":
            return {
                "message": "Great meeting energy detected! Based on your patterns, this is the perfect time for AI-assisted creative work. Your best productive sessions happen right after excellent meetings.",
                "action": "momentum_capitalize",
                "priority": 1,
                "suggested_tools": ["ChatGPT", "Grok"],
                "reasoning": "Personal post-meeting momentum transfer pattern"
            }
        
        elif reason == "social_browsing_detected":
            return {
                "message": "Social media browsing detected. Your productive recovery pattern shows switching to AI tools works best. Try Grok or ChatGPT for your next task?",
                "action": "productive_redirect",
                "priority": 2,
                "suggested_tools": ["Grok", "ChatGPT"],
                "reasoning": "Personal social media → AI tool recovery pattern"
            }
        
        elif reason == "worksmart_low_activity_detected":
            worksmart_stats = self.worksmart_reader.get_current_session_stats()
            hours_today = worksmart_stats.get('total_hours_today', '0:0')
            return {
                "message": f"WorkSmart shows very low activity (< 3 keystrokes, < 5 clicks). You've worked {hours_today} today. Based on your patterns, engaging with AI tools like Grok or ChatGPT helps re-energize your productivity.",
                "action": "gentle_engagement",
                "priority": 2,
                "suggested_tools": ["Grok", "ChatGPT"],
                "reasoning": "WorkSmart official telemetry + personal AI engagement pattern",
                "data_source": "WorkSmart Official"
            }
        
        elif reason == "worksmart_high_click_low_type_pattern":
            return {
                "message": f"WorkSmart detects high mouse activity (20+ clicks) but low typing (< 10 keystrokes). This often indicates browsing/distraction. Your patterns show AI tools help refocus productive work.",
                "action": "focus_coaching",
                "priority": 2,
                "suggested_tools": ["Grok", "ChatGPT"],
                "reasoning": "WorkSmart activity pattern analysis + personal focus recovery",
                "data_source": "WorkSmart Official"
            }
        
        elif reason == "worksmart_extended_work_session":
            worksmart_stats = self.worksmart_reader.get_current_session_stats()
            hours_today = worksmart_stats.get('total_hours_today', '0:0')
            return {
                "message": f"Impressive! WorkSmart shows {hours_today} of work today. After 4+ hours, your effectiveness patterns show strategic 10-15 minute breaks actually boost overall productivity. Consider a brief AI-assisted planning break?",
                "action": "strategic_break",
                "priority": 1,
                "suggested_tools": ["ChatGPT - Planning", "Brief walk"],
                "reasoning": "WorkSmart extended session + personal sustainability patterns",
                "data_source": "WorkSmart Official"
            }
        
        # Fallback to generic but still personalized
        return {
            "message": f"Based on your patterns, consider focusing on {current_app if current_app in self.personal_patterns['productive_apps'] else 'an AI tool'} for optimal productivity.",
            "action": "general_focus",
            "priority": 3,
            "reasoning": "Fallback personalized message"
        }
    
    def _classify_current_state(self, analysis: Dict) -> str:
        """Classify current productivity state based on analysis"""
        productivity = analysis.get('productivity_score', 0.5)
        focus = analysis.get('focus_quality', 0.5)
        activity_level = analysis.get('activity_level', 'MEDIUM')
        
        if productivity > 0.8 and focus > 0.8:
            return 'high_productive'
        elif productivity > 0.6 and focus > 0.6:
            return 'moderate'
        elif productivity < 0.4 or activity_level == 'LOW':
            return 'unproductive'
        else:
            return 'moderate'
    
    # Detection helper methods
    def _detect_ai_tools(self, context: Dict) -> bool:
        """Detect if AI tools are active"""
        current_app = context.get('current_application', '')
        chrome_context = context.get('chrome_context', {})
        
        # Check current app
        if any(tool in current_app for tool in self.personal_patterns['ai_tools']):
            return True
        
        # Check Chrome tabs
        active_url = chrome_context.get('active_tab_url', '')
        if any(tool in active_url for tool in self.personal_patterns['ai_tools']):
            return True
        
        return False
    
    def _detect_meeting_context(self, context: Dict) -> bool:
        """Detect if in meeting context"""
        current_app = context.get('current_application', '')
        window_title = context.get('current_window', '').lower()
        
        return (current_app == 'Safari' and 
                any(term in window_title for term in ['meet', 'zoom', 'call', 'conference']))
    
    def _detect_social_browsing(self, context: Dict) -> bool:
        """Detect social media browsing"""
        chrome_context = context.get('chrome_context', {})
        active_url = chrome_context.get('active_tab_url', '')
        
        return any(signal in active_url for signal in self.personal_patterns['distraction_signals'])
    
    def _detect_tab_overload(self, context: Dict) -> bool:
        """Detect Chrome tab overload"""
        return self._get_tab_count(context) > self.personal_patterns['tab_overload_threshold']
    
    def _detect_ai_flow_state(self, context: Dict) -> bool:
        """Detect AI flow state"""
        return (self._detect_ai_tools(context) and 
                self._get_tab_count(context) <= self.personal_patterns['optimal_tab_count'])
    
    def _detect_meeting_excellence(self, context: Dict) -> bool:
        """Detect meeting excellence state"""
        return (self._detect_meeting_context(context) and
                context.get('window_count', 0) <= 2)
    
    def _detect_post_meeting_transition(self, context: Dict) -> bool:
        """Detect transition from meeting to other work"""
        # Simplified detection - would need recent context history
        return False  # TODO: Implement with context history
    
    def _get_tab_count(self, context: Dict) -> int:
        """Get Chrome tab count from context"""
        chrome_context = context.get('chrome_context', {})
        return chrome_context.get('total_tabs', 0)
    
    def _calculate_base_activity_score(self, events: List[Dict]) -> float:
        """Calculate base activity score from events"""
        if not events:
            return 0.5
        
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