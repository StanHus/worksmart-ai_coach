#!/usr/bin/env python3
"""
Micro-Intervention System
========================
Subtle productivity nudges that work at 2-3 minute intervals vs full interventions.
Provides gentle guidance without disrupting flow states.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class InterventionType(Enum):
    MICRO_NUDGE = "micro_nudge"         # 2-3 minute gentle hints
    GENTLE_REDIRECT = "gentle_redirect"  # 3-5 minute soft suggestions
    STANDARD = "standard"               # 5+ minute full interventions
    FLOW_PROTECTION = "flow_protection"  # Protective non-intervention


class InterventionIntensity(Enum):
    SILENT = 0          # No notification, just internal tracking
    SUBTLE = 1          # Very brief, minimal notification
    GENTLE = 2          # Standard notification
    ASSERTIVE = 3       # Priority notification with sound


class MicroInterventionSystem:
    """Advanced micro-intervention system for subtle productivity optimization"""

    def __init__(self):
        self.intervention_history = []
        self.last_intervention_time = None
        self.intervention_cooldown_minutes = 3  # Minimum time between interventions
        self.flow_state_protection = True

    def evaluate_micro_intervention(self, context: Dict, analysis: Dict,
                                    context_history: List[Dict]) -> Optional[Dict]:
        """Evaluate if micro-intervention is needed"""

        current_time = datetime.now()

        # Respect cooldown period
        if (self.last_intervention_time and
            (current_time - self.last_intervention_time).total_seconds() <
                self.intervention_cooldown_minutes * 60):
            return None

        productivity_score = analysis.get('productivity_score', 0.5)
        focus_quality = analysis.get('focus_quality', 0.5)
        session_minutes = (analysis.get('session_hours', 0) * 60)

        # Micro-intervention triggers (2-3 minute intervals)

        # 1. Early tab overload detection
        tab_count = self._get_tab_count(context)
        if tab_count > 12 and session_minutes >= 2:
            return self._create_micro_intervention(
                InterventionType.MICRO_NUDGE,
                InterventionIntensity.SUBTLE,
                "Tab consolidation hint",
                f"Noticed {tab_count} tabs open. Consider focusing on 1-2 key tabs for better concentration.",
                {"suggested_action": "tab_consolidation", "tab_count": tab_count}
            )

        # 2. Productivity drift detection (early warning)
        if (productivity_score < 0.6 and session_minutes >= 2.5 and
                not self._is_in_flow_state(context, analysis)):

            return self._create_micro_intervention(
                InterventionType.MICRO_NUDGE,
                InterventionIntensity.SUBTLE,
                "Gentle focus reminder",
                "Mild productivity drift detected. Consider switching to a focused task or AI tool.",
                {"suggested_tools": ["ChatGPT", "Grok"]}
            )

        # 3. Context switch frequency warning
        app_switches = self._count_recent_app_switches(
            context_history, minutes=3)
        if app_switches > 4:
            return self._create_micro_intervention(
                InterventionType.MICRO_NUDGE,
                InterventionIntensity.SILENT,
                "Context switching alert",
                f"Detected {app_switches} app switches in 3 minutes. Consider settling into one primary task.",
                {"app_switches": app_switches}
            )

        # 4. Pre-distraction pattern detection
        if self._detect_pre_distraction_pattern(context, context_history):
            return self._create_micro_intervention(
                InterventionType.GENTLE_REDIRECT,
                InterventionIntensity.SUBTLE,
                "Distraction prevention",
                "Pattern suggests potential distraction ahead. Stay focused on your current productive flow.",
                {"pattern_type": "pre_distraction"}
            )

        # 5. Momentum building opportunity
        if (productivity_score > 0.7 and focus_quality > 0.7 and session_minutes < 5):
            return self._create_micro_intervention(
                InterventionType.MICRO_NUDGE,
                InterventionIntensity.SILENT,
                "Momentum building",
                "Great start! You're building productive momentum. Keep this energy going.",
                {"momentum_type": "early_positive"}
            )

        # 6. Energy preservation for long sessions
        if session_minutes > 45 and productivity_score > 0.8:
            return self._create_micro_intervention(
                InterventionType.MICRO_NUDGE,
                InterventionIntensity.GENTLE,
                "Energy management",
                "Excellent sustained productivity! Consider taking a 30-second break to maintain this level.",
                {"session_minutes": session_minutes}
            )

        return None

    def _create_micro_intervention(self, intervention_type: InterventionType,
                                   intensity: InterventionIntensity, title: str,
                                   message: str, metadata: Dict) -> Dict:
        """Create micro-intervention response"""

        intervention = {
            "type": intervention_type.value,
            "intensity": intensity.value,
            "title": title,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
            "should_notify": intensity.value > InterventionIntensity.SILENT.value,
            # Cap at priority 3
            "notification_priority": min(3, intensity.value),
            "reasoning": f"Micro-intervention: {title}"
        }

        # Update tracking
        self.intervention_history.append(intervention)
        self.last_intervention_time = datetime.now()

        return intervention

    def _get_tab_count(self, context: Dict) -> int:
        """Get Chrome tab count from context"""
        chrome_context = context.get('chrome_context', {})
        return chrome_context.get('total_tabs', 0)

    def _is_in_flow_state(self, context: Dict, analysis: Dict) -> bool:
        """Detect if user is in flow state (protect from interruptions)"""
        productivity = analysis.get('productivity_score', 0.5)
        focus = analysis.get('focus_quality', 0.5)

        # AI tool engagement with high productivity
        ai_tools = ['grok.com', 'chat.openai.com', 'ChatGPT', 'Terminal']
        current_app = context.get('current_application', '')
        chrome_url = context.get(
            'chrome_context', {}).get('active_tab_url', '')

        ai_tool_active = (current_app in ai_tools or
                          any(tool in chrome_url for tool in ai_tools))

        return (productivity > 0.8 and focus > 0.8 and ai_tool_active)

    def _count_recent_app_switches(self, context_history: List[Dict],
                                   minutes: int = 3) -> int:
        """Count app switches in recent history"""
        if not context_history or len(context_history) < 2:
            return 0

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_contexts = [c for c in context_history[-10:]
                           if datetime.fromisoformat(c.get('timestamp', datetime.now().isoformat())) > cutoff_time]

        if len(recent_contexts) < 2:
            return 0

        app_switches = 0
        for i in range(1, len(recent_contexts)):
            if recent_contexts[i].get('app_name') != recent_contexts[i-1].get('app_name'):
                app_switches += 1

        return app_switches

    def _detect_pre_distraction_pattern(self, context: Dict,
                                        context_history: List[Dict]) -> bool:
        """Detect patterns that typically lead to distraction"""

        # Pattern 1: Multiple quick URL changes in Chrome
        chrome_url = context.get(
            'chrome_context', {}).get('active_tab_url', '')
        if 'google.com/search' in chrome_url:
            recent_urls = [c.get('chrome_url', '')
                           for c in context_history[-5:]]
            google_searches = sum(
                1 for url in recent_urls if 'google.com/search' in url)
            if google_searches >= 2:
                return True

        # Pattern 2: Social media browsing initiation
        social_indicators = ['linkedin.com',
                             'twitter.com', 'facebook.com', 'reddit.com']
        if any(social in chrome_url for social in social_indicators):
            return True

        # Pattern 3: Tab count rapid increase
        current_tabs = self._get_tab_count(context)
        if current_tabs > 8 and context_history:
            previous_tabs = context_history[-1].get(
                'tab_count', 0) if context_history else 0
            if current_tabs > previous_tabs + 2:  # Rapid tab opening
                return True

        return False

    def get_intervention_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of recent micro-interventions"""
        if not self.intervention_history:
            return {"total_interventions": 0}

        recent_interventions = [i for i in self.intervention_history
                                if (datetime.now() - datetime.fromisoformat(i['timestamp'])).total_seconds() < 3600]

        intervention_types = {}
        for intervention in recent_interventions:
            intervention_type = intervention['type']
            if intervention_type not in intervention_types:
                intervention_types[intervention_type] = 0
            intervention_types[intervention_type] += 1

        return {
            "total_interventions": len(recent_interventions),
            "intervention_breakdown": intervention_types,
            "avg_intensity": sum(i['intensity'] for i in recent_interventions) / len(recent_interventions) if recent_interventions else 0,
            "most_common_type": max(intervention_types.items(), key=lambda x: x[1])[0] if intervention_types else None
        }

    def adjust_sensitivity(self, feedback: str) -> None:
        """Adjust micro-intervention sensitivity based on user feedback"""
        if "too many" in feedback.lower() or "annoying" in feedback.lower():
            self.intervention_cooldown_minutes += 1
            print(
                f"🔧 Increased intervention cooldown to {self.intervention_cooldown_minutes} minutes")

        elif "more" in feedback.lower() or "helpful" in feedback.lower():
            self.intervention_cooldown_minutes = max(
                2, self.intervention_cooldown_minutes - 0.5)
            print(
                f"🔧 Decreased intervention cooldown to {self.intervention_cooldown_minutes} minutes")

        elif "flow" in feedback.lower() and "protect" in feedback.lower():
            self.flow_state_protection = True
            print("🔧 Enhanced flow state protection enabled")
