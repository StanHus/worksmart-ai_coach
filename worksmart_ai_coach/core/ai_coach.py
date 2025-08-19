#!/usr/bin/env python3
"""
Ultimate AI Coach - Complete Consolidated Coaching System
========================================================

The ultimate consolidation of all AI coaching functionality into a single comprehensive system:

🧠 ALL COACHING SYSTEMS IN ONE FILE:
- Base AI Coach with Anthropic Claude integration
- Personalized AI Coach with persona detection and learning
- Enhanced ML AI Coach with full machine learning capabilities
- Pattern Learning, User Modeling, Predictive Analytics
- Feedback Collection and Continuous Learning
- Burnout Prediction and Optimal Timing
- Context Tracking and Adaptive Learning
- Micro-Interventions and Nudge DNA
- WorkSmart Integration and Telemetry Analysis

🚀 FEATURES INCLUDED:
✅ Machine Learning - Learns from user interactions with fallbacks
✅ Pattern Discovery - Finds effectiveness patterns in telemetry data  
✅ Burnout Prediction - Predicts and prevents user burnout
✅ Personalization - Adapts to individual user preferences and baselines
✅ Context Sensitivity - Multi-dimensional context analysis
✅ Continuous Learning - Improves over time with implicit/explicit feedback
✅ Predictive Intelligence - Anticipates user needs and optimal timing
✅ Adaptive Strategies - Changes approach based on measured effectiveness
✅ Persona Detection - Developer/Analyst/Manager specific coaching
✅ Anthropic API Integration - Advanced AI-generated coaching advice
✅ WorkSmart Integration - Uses official WorkSmart telemetry data
✅ Comprehensive Fallbacks - Works with or without ML dependencies

This single file replaces: coach.py, personalized_coach.py, enhanced_ai_coach.py, 
ai_coaching_system.py, and all ML component files.
"""

import json
import os
import asyncio
import logging
import pickle
import numpy as np
import time
import subprocess
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

# Optional ML dependencies with comprehensive fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Anthropic API integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# ============================================================================
# FALLBACK CLASSES FOR WHEN ML LIBRARIES ARE NOT AVAILABLE
# ============================================================================


class SimpleDataFrame:
    """Simple DataFrame-like class for basic data operations when pandas unavailable"""

    def __init__(self, data):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            self.data = data
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row.get(key) for row in self.data]
        else:
            return self.data[key]

    def mean(self):
        if not self.data:
            return {}
        result = {}
        for key in self.data[0].keys():
            values = [row[key] for row in self.data if isinstance(
                row.get(key), (int, float))]
            if values:
                result[key] = sum(values) / len(values)
        return result

# Simple ML model fallbacks


class SimpleFallbackClassifier:
    """Simple fallback classifier when sklearn unavailable"""

    def __init__(self, n_estimators=50, random_state=42):
        self.most_common_class = None

    def fit(self, X, y):
        if len(y) > 0:
            class_counts = {}
            for label in y:
                class_counts[label] = class_counts.get(label, 0) + 1
            self.most_common_class = max(
                class_counts.items(), key=lambda x: x[1])[0]

    def predict(self, X):
        return [self.most_common_class] * len(X) if self.most_common_class else ['unknown'] * len(X)

    def predict_proba(self, X):
        return [[0.5, 0.5]] * len(X)


class SimpleFallbackRegressor:
    """Simple fallback regressor when sklearn unavailable"""

    def __init__(self, n_estimators=50, random_state=42):
        self.mean_value = 0.5

    def fit(self, X, y):
        if len(y) > 0:
            self.mean_value = sum(y) / len(y)

    def predict(self, X):
        return [self.mean_value] * len(X)


class SimpleFallbackScaler:
    """Simple fallback scaler when sklearn unavailable"""

    def __init__(self):
        self.means = {}
        self.stds = {}

    def fit(self, X):
        if len(X) > 0 and len(X[0]) > 0:
            n_features = len(X[0])
            for i in range(n_features):
                values = [row[i] for row in X if len(row) > i]
                if values:
                    self.means[i] = sum(values) / len(values)
                    if len(values) > 1:
                        variance = sum(
                            (x - self.means[i]) ** 2 for x in values) / (len(values) - 1)
                        self.stds[i] = variance ** 0.5
                    else:
                        self.stds[i] = 1.0
        return self

    def transform(self, X):
        if not self.means:
            return X
        transformed = []
        for row in X:
            new_row = []
            for i, val in enumerate(row):
                if i in self.means and self.stds.get(i, 1.0) > 0:
                    new_row.append((val - self.means[i]) / self.stds[i])
                else:
                    new_row.append(val)
            transformed.append(new_row)
        return transformed

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class IsolationForestFallback:
    """Simple anomaly detection fallback"""

    def __init__(self, contamination=0.1, random_state=42):
        self.threshold = 0.1

    def fit(self, X):
        return self

    def predict(self, X):
        return [1] * len(X)  # Normal by default


# Import or use fallback classes based on availability
if SKLEARN_AVAILABLE:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
else:
    RandomForestClassifier = SimpleFallbackClassifier
    GradientBoostingRegressor = SimpleFallbackRegressor
    RandomForestRegressor = SimpleFallbackRegressor
    StandardScaler = SimpleFallbackScaler
    IsolationForest = IsolationForestFallback

# ============================================================================
# DATA STRUCTURES AND USER MODEL
# ============================================================================


@dataclass
class UserPreferences:
    """User preferences for coaching"""
    notification_frequency: str = "medium"  # low, medium, high
    intervention_style: str = "balanced"    # minimal, balanced, assertive
    preferred_times: List[int] = None       # Preferred hours (0-23)
    avoided_times: List[int] = None         # Hours to avoid notifications
    break_reminder_frequency: int = 180     # Minutes between break reminders (increased from 120)
    focus_session_duration: int = 45        # Preferred focus session length
    coaching_language: str = "specific"     # generic, specific, technical
    privacy_level: str = "standard"         # minimal, standard, detailed


@dataclass
class UserProfile:
    """Complete user behavioral profile"""
    user_id: str
    persona: str = "generic"                # analyst, developer, manager, etc.
    confidence_level: float = 0.5          # How confident we are in persona detection
    productivity_baseline: float = 0.5     # Personal productivity baseline
    focus_baseline: float = 0.5            # Personal focus baseline
    stress_tolerance: float = 0.5          # How much stress user can handle
    energy_patterns: Dict[int, float] = None  # Energy by hour of day
    productivity_patterns: Dict[str, float] = None  # Productivity by context
    intervention_effectiveness: Dict[str,
                                     float] = None  # Effectiveness by type
    preferences: UserPreferences = None
    last_updated: str = ""
    total_interactions: int = 0


@dataclass
class FeedbackEntry:
    """Single feedback entry for learning"""
    intervention_id: str
    user_id: str
    timestamp: str
    intervention_type: str
    intervention_message: str
    feedback_method: str  # explicit, implicit, behavioral
    effectiveness_score: float  # 0-1
    response_time_seconds: float
    user_rating: Optional[int] = None  # 1-5 star rating
    user_comment: Optional[str] = None
    context_at_intervention: Dict = None
    behavioral_response: Dict = None

# ============================================================================
# ULTIMATE AI COACH - ALL FUNCTIONALITY CONSOLIDATED
# ============================================================================


class AICoach:
    """
    Ultimate AI Coach - Complete consolidated coaching system combining all functionality:
    - Base AI Coach, Personalized Coach, Enhanced ML Coach
    - Pattern Learning, User Modeling, Predictive Analytics
    - Feedback Collection, Burnout Prediction, Context Tracking
    - All in one comprehensive class with automatic capability detection
    """

    def __init__(self, data_dir: str = "ultimate_coach_data"):
        """Initialize the ultimate coaching system with all capabilities"""

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Initialize Anthropic client
        self.claude_client = None
        self._setup_anthropic_client()

        # Core coaching thresholds and strategies - RAISED THRESHOLDS FOR LESS FREQUENT NOTIFICATIONS
        self.coaching_strategies = {
            'productivity_thresholds': {'low': 0.25, 'high': 0.7},     # lowered 'low' from 0.3 to 0.25
            'focus_thresholds': {'low': 0.35, 'high': 0.8},           # lowered 'low' from 0.4 to 0.35
            'stress_thresholds': {'moderate': 0.65, 'high': 0.8},     # raised from 0.5/0.7 to 0.65/0.8
            'energy_thresholds': {'low': 0.25, 'critical': 0.15}      # lowered from 0.3/0.2 to 0.25/0.15
        }

        # Notification & alert config - REDUCED FREQUENCY FOR LESS SPAM
        self.notification_config = {
            'max_per_hour': 2,                 # global cap (reduced from 4)
            'min_minutes_between': 20,         # global cooldown (increased from 12 minutes)
            'per_type_cooldown_minutes': {
                'stress_reduction': 20,        # increased from 10
                'energy_boost': 30,           # increased from 20
                'productivity_boost': 35,     # increased from 25
                'focus_enhancement': 30,      # increased from 20
                'break_reminder': 45          # increased from 30
            },
            'suppress_in_meeting': True,       # gate non-critical nudges when in meeting
            'allow_in_meeting_types': ['stress_reduction'],
            'repeat_suppression_minutes': 90,  # don't repeat same message text within this window
            'default_channel': 'system_banner'  # optional: 'toast', 'modal', 'push'
        }

        # Standardize canonical keys used throughout
        self.keys = {
            'session_duration': 'session_duration_hours',
            'current_app': 'current_application',
            'in_meeting': 'in_meeting'
        }

        # Intervention history for cooldowns and analytics
        self.intervention_history: Dict[str, Dict] = {}

        # ML components
        self.pattern_learner_enabled = SKLEARN_AVAILABLE
        self.effectiveness_predictor = RandomForestClassifier(
            n_estimators=50, random_state=42)
        self.timing_optimizer = GradientBoostingRegressor(
            n_estimators=50, random_state=42)
        self.scaler = StandardScaler()

        # User modeling and context tracking
        self.user_profiles: Dict[str, UserProfile] = {}
        self.user_contexts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100))
        self.adaptive_thresholds: Dict[str, Dict] = {}

        # Feedback and learning systems
        self.feedback_history: List[FeedbackEntry] = []
        self.intervention_contexts = {}
        self.learning_callbacks: List[Callable[[FeedbackEntry], None]] = []

        # Interaction data for ML training
        self.interaction_data: List[Dict] = []

        # Model performance tracking
        self.model_performance = {
            'effectiveness_accuracy': 0.0,
            'timing_mse': 0.0,
            'last_trained': None
        }

        # Intervention history
        self.intervention_history = {}

        # Persona detection patterns
        self.persona_patterns = {
            'developer': {
                'apps': ['vscode', 'visual studio code', 'intellij', 'pycharm', 'xcode', 'sublime', 'atom', 'vim', 'emacs', 'github desktop', 'terminal'],
                'keywords': ['code', 'debug', 'commit', 'pull request', 'repository', 'function', 'class'],
                'coaching_messages': {
                    'productivity_boost': [
                        "🚀 Focus on your current coding task. Try a 25/5 cycle next.",
                        "🚀 Commit to one function now—set a 25‑min timer.",
                        "🚀 Silence alerts and full‑screen the editor for a 25‑minute push."
                    ],
                    'focus_enhancement': [
                        "Close non‑essential tabs and notifications—give the code your full attention.",
                        "Switch to distraction‑free mode in your editor for the next 20 minutes."
                    ],
                    'stress_reduction': [
                        "Take a step back—2 minutes of deep breathing, then retry the bug.",
                        "Quick reset: walk for 3 minutes, then return."
                    ],
                    'break_reminder': [
                        "You've been coding for a while—take 5 minutes to prevent bugs.",
                        "Stand, stretch, water break—be back in 5."
                    ]
                }
            },
            'analyst': {
                'apps': ['excel', 'tableau', 'power bi', 'r studio', 'jupyter', 'spss', 'stata', 'sas'],
                'keywords': ['data', 'analysis', 'report', 'dashboard', 'visualization', 'spreadsheet'],
                'coaching_messages': {
                    'productivity_boost': [
                        "📊 Focus on your current analysis. Break down complex data problems into smaller, manageable pieces.",
                        "📊 Time-box this analysis: 30 minutes max, then review progress."
                    ],
                    'focus_enhancement': [
                        "Minimize data distractions. Close extra spreadsheets and focus on your primary analysis.",
                        "Single dataset focus: hide other tabs until this analysis is complete."
                    ],
                    'stress_reduction': [
                        "Data can be overwhelming. Take a break to clear your mind and return with fresh perspective.",
                        "Step back from the numbers—5 minutes away will bring clarity."
                    ],
                    'break_reminder': [
                        "You've been analyzing data intensively. A break will help you spot patterns more clearly.",
                        "Data fatigue is real—take 10 minutes to reset your analytical mind."
                    ]
                }
            },
            'manager': {
                'apps': ['slack', 'teams', 'zoom', 'outlook', 'calendar', 'asana', 'trello', 'jira'],
                'keywords': ['meeting', 'review', 'team', 'project', 'deadline', 'status', 'planning'],
                'coaching_messages': {
                    'productivity_boost': [
                        "🎯 Prioritize your most important management task. Delegate what you can to maximize impact.",
                        "🎯 Pick your top 3 priorities—everything else can wait 25 minutes."
                    ],
                    'focus_enhancement': [
                        "Batch your communications. Set specific times for emails and messages to maintain focus.",
                        "Deep work block: close Slack/Teams for the next 30 minutes."
                    ],
                    'stress_reduction': [
                        "Leadership is demanding. Take a moment to breathe and remember you don't have to solve everything right now.",
                        "Pause: delegate one task, then take 5 minutes for yourself."
                    ],
                    'break_reminder': [
                        "Even great managers need breaks. Step away to return with better decision-making clarity.",
                        "Strategic pause: 10 minutes away will improve your next 3 decisions."
                    ]
                }
            },
            'generic': {
                'apps': [],
                'keywords': [],
                'coaching_messages': {
                    'productivity_boost': [
                        "💪 Focus on your most important task. Break it down into smaller, actionable steps.",
                        "💪 Pick one thing and do it well for the next 25 minutes."
                    ],
                    'focus_enhancement': [
                        "Minimize distractions around you. Single-tasking will improve your work quality.",
                        "Close everything except what you need for this one task."
                    ],
                    'stress_reduction': [
                        "Take a few deep breaths. A brief break can help reset your mindset.",
                        "Quick reset: 2-3 deep breaths, then back to focused work."
                    ],
                    'break_reminder': [
                        "You've been working steadily. A short break will help maintain your productivity.",
                        "Time for a 5-minute reset—stretch and hydrate."
                    ]
                }
            }
        }

        # Load existing data
        self._load_user_profiles()
        self._load_training_data()
        self._load_feedback_history()

        # Determine system capabilities
        self.capabilities = self._determine_capabilities()

        logger.info(
            f"Ultimate AI Coach initialized with {len(self.capabilities)} capabilities: {', '.join(self.capabilities)}")

    def _setup_anthropic_client(self):
        """Setup Anthropic Claude client if available"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if ANTHROPIC_AVAILABLE and api_key:
                self.claude_client = anthropic.AsyncAnthropic(api_key=api_key)
                logger.info("✅ Anthropic Claude client initialized")
            else:
                logger.info(
                    "ℹ️  Using rule-based coaching (no Anthropic API key or library)")
        except Exception as e:
            logger.warning(f"Anthropic client setup failed: {e}")

    def _determine_capabilities(self) -> List[str]:
        """Determine system capabilities based on available libraries"""
        capabilities = ['basic_coaching',
                        'persona_detection', 'context_awareness']

        if SKLEARN_AVAILABLE:
            capabilities.extend(
                ['ml_pattern_learning', 'predictive_analytics', 'burnout_prediction'])

        if PANDAS_AVAILABLE:
            capabilities.append('advanced_data_analysis')

        if self.claude_client:
            capabilities.append('ai_generated_coaching')

        capabilities.extend(
            ['user_modeling', 'feedback_learning', 'adaptive_thresholds'])

        return capabilities

    # ========================================================================
    # MAIN COACHING INTERFACE
    # ========================================================================

    async def analyze_telemetry(self, telemetry_data: Dict, user_id: str = "default",
                                context_history: List[Dict] = None) -> Optional[Dict]:
        """
        Main entry point for telemetry analysis and coaching.
        Automatically uses the best available method based on system capabilities.
        """

        # CRITICAL: Only coach when WorkSmart is actively monitoring
        if not telemetry_data.get('worksmart_session_active', False):
            logger.info("Skipping coaching - WorkSmart not actively monitoring")
            return None

        try:
            # Enhanced ML coaching if available
            if 'ml_pattern_learning' in self.capabilities:
                result = await self._analyze_telemetry_enhanced(telemetry_data, user_id, context_history)
                if result:
                    return result

            # Personalized coaching fallback
            if 'persona_detection' in self.capabilities:
                result = await self._analyze_telemetry_personalized(telemetry_data, user_id)
                if result:
                    return result

            # Basic coaching final fallback
            return await self._analyze_telemetry_basic(telemetry_data, user_id)

        except Exception as e:
            logger.error(f"Telemetry analysis failed: {e}")
            return await self._analyze_telemetry_basic(telemetry_data, user_id)

    async def _analyze_telemetry_enhanced(self, telemetry_data: Dict, user_id: str,
                                          context_history: List[Dict] = None) -> Optional[Dict]:
        """Enhanced ML-based telemetry analysis"""

        # Update user model with new context
        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)

        self._update_user_context(user_id, context, analysis)

        # Get user profile and coaching strategy
        user_profile = self._get_user_profile(user_id)
        coaching_strategy = self._get_coaching_strategy(user_id, context)

        if not coaching_strategy['intervention_needed']:
            return None

        # Get predictive insights
        user_history = list(self.user_contexts.get(user_id, []))
        predictive_insights = self._analyze_predictive_insights(
            user_id, user_history, context)

        # Check optimal timing
        intervention_type = coaching_strategy['intervention_type']
        optimal_timing = self._get_optimal_intervention_timing(
            user_id, intervention_type)

        if optimal_timing['delay_minutes'] > 30:
            logger.info(
                f"Delaying intervention for {user_id}: {optimal_timing['reason']}")
            return None

        # Generate AI coaching
        ai_coaching = await self._generate_ai_coaching(user_profile, context, coaching_strategy, predictive_insights)

        if not ai_coaching:
            return None

        # Create comprehensive intervention
        intervention_id = str(uuid.uuid4())
        intervention = {
            'id': intervention_id,
            'user_id': user_id,
            'type': intervention_type,
            'nudge_type': intervention_type,  # Backward compatibility
            'message': ai_coaching.get('message', ''),
            'priority': ai_coaching.get('priority', 2),
            'reasoning': ai_coaching.get('reasoning', ''),
            'confidence': ai_coaching.get('confidence', 0.5),
            'source': 'enhanced_ml',
            'persona': user_profile.persona,
            'predicted_effectiveness': self._predict_intervention_effectiveness(context, ai_coaching) if self.pattern_learner_enabled else 0.5,
            'coaching_strategy': coaching_strategy,
            'predictive_insights': predictive_insights
        }

        # Suppression/cooldown check BEFORE recording
        suppress, reason = self._should_suppress_notification(user_id, intervention, context)
        if suppress:
            self._log_notification_event('suppressed', user_id, intervention, reason=reason)
            return None

        # Record (adds timestamp) and log
        self._record_intervention(intervention, user_id)
        self._log_notification_event('sent', user_id, intervention)

        # Record for feedback tracking
        self._record_intervention_for_feedback(intervention_id, user_id, intervention_type,
                                               ai_coaching.get('message', ''), context, ai_coaching.get('priority', 2))

        logger.info(
            f"Generated enhanced ML intervention: {intervention_type} (confidence: {intervention['confidence']:.2f})")
        return intervention

    async def _analyze_telemetry_personalized(self, telemetry_data: Dict, user_id: str) -> Optional[Dict]:
        """Personalized coaching with persona detection"""

        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)

        # Detect persona
        persona = self._detect_user_persona(context)

        # Get persona-specific coaching
        coaching_result = self._get_persona_specific_coaching(
            persona, context, analysis, user_id)

        if not coaching_result:
            return None

        coaching_result['source'] = 'personalized'
        coaching_result['persona'] = persona

        # Check suppression
        suppress, reason = self._should_suppress_notification(user_id, coaching_result, context)
        if suppress:
            logger.info(f"Suppressed {coaching_result['type']} for {user_id}: {reason}")
            self._log_notification_event('suppressed', user_id, coaching_result, reason=reason)
            return None

        # Record and return
        self._record_intervention(coaching_result, user_id)
        self._log_notification_event('sent', user_id, coaching_result)
        return coaching_result

    async def _analyze_telemetry_basic(self, telemetry_data: Dict, user_id: str) -> Optional[Dict]:
        """Basic rule-based coaching"""

        productivity = telemetry_data.get('productivity_score', 0.5)
        focus = telemetry_data.get('focus_quality', 0.5)
        stress = telemetry_data.get('stress_level', 0.5)
        energy = telemetry_data.get('energy_level', 0.5)
        session_hours = telemetry_data.get('session_duration_hours', 0)

        coaching_type, urgency = self._determine_coaching_need(
            productivity, focus, stress, energy, session_hours)

        if not coaching_type:
            return None

        # Try Anthropic API first
        intervention = None
        if self.claude_client:
            ai_response = await self._get_anthropic_coaching(telemetry_data, coaching_type, urgency, user_id)
            if ai_response:
                ai_response['source'] = 'anthropic_ai'
                intervention = ai_response
        
        # Rule-based fallback
        if not intervention:
            intervention = self._get_rule_based_coaching(coaching_type, urgency, telemetry_data, user_id)
        
        if not intervention:
            return None
        
        # Check suppression
        analysis = self._basic_analysis(telemetry_data)
        context = self._extract_context(telemetry_data, analysis)
        suppress, reason = self._should_suppress_notification(user_id, intervention, context)
        if suppress:
            logger.info(f"Suppressed {intervention['type']} for {user_id}: {reason}")
            self._log_notification_event('suppressed', user_id, intervention, reason=reason)
            return None
        
        # Record and return
        self._record_intervention(intervention, user_id)
        self._log_notification_event('sent', user_id, intervention)
        return intervention

    # ========================================================================
    # USER MODELING AND CONTEXT ANALYSIS
    # ========================================================================

    def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                energy_patterns={},
                productivity_patterns={},
                intervention_effectiveness={},
                preferences=UserPreferences(),
                last_updated=datetime.now().isoformat(),
                total_interactions=0
            )
        return self.user_profiles[user_id]

    def _update_user_context(self, user_id: str, context: Dict, analysis: Dict):
        """Update user context and learn patterns"""
        profile = self._get_user_profile(user_id)

        # Add context to history
        enriched_context = {
            **context,
            **analysis,
            'timestamp': datetime.now().isoformat(),
            'hour_of_day': datetime.now().hour
        }
        self.user_contexts[user_id].append(enriched_context)

        # Update patterns
        self._update_energy_patterns(profile, enriched_context)
        self._update_productivity_patterns(profile, enriched_context)
        self._update_persona_confidence(profile, enriched_context)
        self._update_adaptive_thresholds(user_id, enriched_context)

        profile.total_interactions += 1
        profile.last_updated = datetime.now().isoformat()

        # Save updated profile
        self._save_user_profile(profile)

    def _update_energy_patterns(self, profile: UserProfile, context: Dict):
        """Learn user's energy patterns by hour"""
        hour = context['hour_of_day']
        energy = context.get('energy_level', 0.5)

        if profile.energy_patterns is None:
            profile.energy_patterns = {}

        # Exponential moving average
        if hour in profile.energy_patterns:
            profile.energy_patterns[hour] = 0.8 * \
                profile.energy_patterns[hour] + 0.2 * energy
        else:
            profile.energy_patterns[hour] = energy

    def _update_productivity_patterns(self, profile: UserProfile, context: Dict):
        """Learn productivity patterns by application"""
        app = context.get('current_application', 'unknown')
        productivity = context.get('productivity_score', 0.5)

        if profile.productivity_patterns is None:
            profile.productivity_patterns = {}

        if app in profile.productivity_patterns:
            profile.productivity_patterns[app] = 0.7 * \
                profile.productivity_patterns[app] + 0.3 * productivity
        else:
            profile.productivity_patterns[app] = productivity

    def _update_persona_confidence(self, profile: UserProfile, context: Dict):
        """Update persona detection confidence"""
        app = context.get('current_application', '').lower()

        persona_scores = defaultdict(float)

        # Score each persona based on application usage
        for persona, patterns in self.persona_patterns.items():
            if persona == 'generic':
                continue
            for indicator_app in patterns['apps']:
                if indicator_app in app:
                    persona_scores[persona] += 0.1

        # Update profile with highest scoring persona
        if persona_scores:
            best_persona = max(persona_scores.items(), key=lambda x: x[1])
            if best_persona[1] > 0.05:
                if profile.persona != best_persona[0]:
                    profile.persona = best_persona[0]
                    profile.confidence_level = min(
                        0.9, profile.confidence_level + 0.1)

    def _update_adaptive_thresholds(self, user_id: str, context: Dict):
        """Update personalized thresholds based on user's baselines"""
        if user_id not in self.adaptive_thresholds:
            self.adaptive_thresholds[user_id] = {
                'productivity_low': 0.3,
                'productivity_high': 0.7,
                'focus_low': 0.4,
                'focus_high': 0.8,
                'stress_high': 0.6,
                'energy_low': 0.3
            }

        profile = self.user_profiles[user_id]
        thresholds = self.adaptive_thresholds[user_id]

        # Update baselines with exponential moving average
        productivity_score = context.get('productivity_score', 0.5)
        focus_quality = context.get('focus_quality', 0.5)

        profile.productivity_baseline = 0.95 * \
            profile.productivity_baseline + 0.05 * productivity_score
        profile.focus_baseline = 0.95 * profile.focus_baseline + 0.05 * focus_quality

        # Adapt thresholds to be relative to user's baseline
        thresholds['productivity_low'] = max(
            0.1, profile.productivity_baseline - 0.2)
        thresholds['productivity_high'] = min(
            0.9, profile.productivity_baseline + 0.2)
        thresholds['focus_low'] = max(0.1, profile.focus_baseline - 0.2)
        thresholds['focus_high'] = min(0.9, profile.focus_baseline + 0.2)

    def _get_coaching_strategy(self, user_id: str, context: Dict) -> Dict:
        """Get personalized coaching strategy"""
        profile = self._get_user_profile(user_id)
        thresholds = self.adaptive_thresholds.get(user_id, {
            'productivity_low': 0.3, 'productivity_high': 0.7,
            'focus_low': 0.4, 'focus_high': 0.8,
            'stress_high': 0.6, 'energy_low': 0.3
        })

        productivity = context.get('productivity_score', 0.5)
        focus = context.get('focus_quality', 0.5)
        stress = context.get('stress_level', 0.5)
        energy = context.get('energy_level', 0.5)

        # Determine intervention need and type
        intervention_needed = False
        intervention_type = None
        urgency_level = 'low'
        confidence = 0.5

        if stress > thresholds.get('stress_high', 0.6):
            intervention_needed = True
            intervention_type = 'stress_reduction'
            urgency_level = 'high' if stress > 0.8 else 'medium'
            confidence = 0.8
        elif energy < thresholds.get('energy_low', 0.3):
            intervention_needed = True
            intervention_type = 'energy_boost'
            urgency_level = 'medium'
            confidence = 0.6
        elif productivity < thresholds.get('productivity_low', 0.3):
            intervention_needed = True
            intervention_type = 'productivity_boost'
            urgency_level = 'medium' if productivity < 0.2 else 'low'
            confidence = 0.7
        elif focus < thresholds.get('focus_low', 0.4):
            intervention_needed = True
            intervention_type = 'focus_enhancement'
            urgency_level = 'medium' if focus < 0.3 else 'low'
            confidence = 0.6

        return {
            'intervention_needed': intervention_needed,
            'intervention_type': intervention_type,
            'urgency_level': urgency_level,
            'confidence': confidence,
            'personalized_thresholds': thresholds,
            'user_baseline': {
                'productivity': profile.productivity_baseline,
                'focus': profile.focus_baseline
            }
        }

    # ========================================================================
    # PREDICTIVE ANALYTICS AND BURNOUT PREVENTION
    # ========================================================================

    def _analyze_predictive_insights(self, user_id: str, user_history: List[Dict],
                                     current_context: Dict) -> Dict[str, Any]:
        """Generate predictive insights including burnout risk"""

        insights = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'prediction_confidence': min(0.8, len(user_history) / 50)
        }

        # Burnout risk analysis
        burnout_analysis = self._predict_burnout_risk(
            user_history, current_context)
        insights['burnout_risk'] = burnout_analysis

        # Anomaly detection
        anomaly_analysis = self._detect_anomalies(
            user_id, current_context, user_history)
        insights['anomalies'] = anomaly_analysis

        # Generate predictions
        insights['predictions'] = self._generate_predictions(
            user_history, current_context)

        # Overall risk assessment
        insights['overall_risk_level'] = self._calculate_overall_risk(
            burnout_analysis, anomaly_analysis)

        return insights

    def _predict_burnout_risk(self, user_history: List[Dict], current_context: Dict) -> Dict[str, Any]:
        """Predict burnout risk based on patterns"""

        if len(user_history) < 5:
            return {
                'risk_score': 0.3,
                'risk_level': 'low',
                'factors': ['insufficient_data'],
                'recommendation': 'Continue monitoring'
            }

        # Analyze risk factors
        risk_indicators = self._analyze_risk_factors(
            user_history, current_context)

        # Risk factor weights
        risk_factors = {
            'prolonged_high_stress': 0.3,
            'declining_productivity': 0.25,
            'extended_work_hours': 0.2,
            'poor_sleep_indicators': 0.15,
            'lack_of_breaks': 0.1
        }

        # Calculate overall risk score
        risk_score = sum(risk_indicators.get(factor, 0) * weight
                         for factor, weight in risk_factors.items())

        risk_score = max(0.0, min(1.0, risk_score))

        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'critical'
        elif risk_score > 0.5:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        active_factors = [factor for factor,
                          score in risk_indicators.items() if score > 0.3]

        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'factors': active_factors,
            'factor_details': risk_indicators,
            'recommendation': self._generate_burnout_recommendation(risk_level, active_factors),
            'confidence': min(0.8, len(user_history) / 50)
        }

    def _analyze_risk_factors(self, user_history: List[Dict], current_context: Dict) -> Dict[str, float]:
        """Analyze specific burnout risk factors"""

        risk_indicators = {}

        # Analyze stress patterns
        stress_levels = [ctx.get('stress_level', 0.5)
                         for ctx in user_history[-20:]]
        avg_stress = sum(stress_levels) / \
            len(stress_levels) if stress_levels else 0.5
        high_stress_days = len([s for s in stress_levels if s > 0.7])

        risk_indicators['prolonged_high_stress'] = min(
            1.0, (avg_stress - 0.4) * 2 + high_stress_days / 20)

        # Analyze productivity decline
        productivity_scores = [ctx.get('productivity_score', 0.5)
                               for ctx in user_history[-15:]]
        if len(productivity_scores) >= 10:
            recent_prod = sum(productivity_scores[-5:]) / 5
            earlier_prod = sum(productivity_scores[:5]) / 5
            decline = max(0, earlier_prod - recent_prod)
            risk_indicators['declining_productivity'] = min(1.0, decline * 3)
        else:
            risk_indicators['declining_productivity'] = 0.2

        # Analyze work hours
        session_hours = [ctx.get('session_duration_hours', 0)
                         for ctx in user_history[-10:]]
        avg_hours = sum(session_hours) / \
            len(session_hours) if session_hours else 4
        excessive_hours = len([h for h in session_hours if h > 8])

        risk_indicators['extended_work_hours'] = min(
            1.0, (avg_hours - 6) / 4 + excessive_hours / 10)

        # Analyze break patterns
        break_counts = [1 if ctx.get(
            'break_taken', False) else 0 for ctx in user_history[-10:]]
        if break_counts:
            break_frequency = sum(break_counts) / len(break_counts)
            risk_indicators['lack_of_breaks'] = max(
                0, 1.0 - break_frequency * 2)
        else:
            risk_indicators['lack_of_breaks'] = 0.4

        # Sleep indicators (energy levels)
        energy_levels = [ctx.get('energy_level', 0.5)
                         for ctx in user_history[-10:]]
        avg_energy = sum(energy_levels) / \
            len(energy_levels) if energy_levels else 0.5
        low_energy_frequency = len(
            [e for e in energy_levels if e < 0.3]) / len(energy_levels) if energy_levels else 0

        risk_indicators['poor_sleep_indicators'] = min(
            1.0, low_energy_frequency + (0.5 - avg_energy))

        return risk_indicators

    def _generate_burnout_recommendation(self, risk_level: str, factors: List[str]) -> str:
        """Generate burnout prevention recommendation"""

        if risk_level == 'critical':
            return "🚨 High burnout risk detected. Consider taking time off and reducing workload immediately."
        elif risk_level == 'high':
            return "⚠️ Elevated burnout risk. Implement stress reduction and ensure regular breaks."
        elif risk_level == 'medium':
            return "💡 Monitor stress levels and maintain work-life balance to prevent escalation."
        else:
            return "✅ Burnout risk is low. Continue current healthy work patterns."

    def _detect_anomalies(self, user_id: str, current_context: Dict, user_history: List[Dict]) -> Dict[str, Any]:
        """Detect unusual patterns that might indicate problems"""

        if len(user_history) < 10:
            return {'anomalies_detected': False, 'message': 'Insufficient data for anomaly detection'}

        # Simple rule-based anomaly detection
        # Check for sudden productivity drops
        recent_prod = [ctx.get('productivity_score', 0.5)
                       for ctx in user_history[-5:]]
        earlier_prod = [ctx.get('productivity_score', 0.5)
                        for ctx in user_history[-15:-5]]

        if len(recent_prod) >= 3 and len(earlier_prod) >= 5:
            recent_avg = sum(recent_prod) / len(recent_prod)
            earlier_avg = sum(earlier_prod) / len(earlier_prod)

            if earlier_avg - recent_avg > 0.3:  # Significant drop
                return {
                    'anomalies_detected': True,
                    'anomaly_type': 'productivity_decline',
                    'severity': 'medium',
                    'description': f'Productivity dropped from {earlier_avg:.1%} to {recent_avg:.1%}',
                    'recommendation': 'Investigate potential causes of productivity decline'
                }

        # Check for stress spikes
        current_stress = current_context.get('stress_level', 0.5)
        avg_stress = sum(ctx.get('stress_level', 0.5)
                         for ctx in user_history[-10:]) / min(10, len(user_history))

        if current_stress > avg_stress + 0.4:  # Stress spike
            return {
                'anomalies_detected': True,
                'anomaly_type': 'stress_spike',
                'severity': 'high',
                'description': f'Stress level {current_stress:.1%} significantly above normal {avg_stress:.1%}',
                'recommendation': 'Immediate stress reduction intervention recommended'
            }

        return {'anomalies_detected': False, 'message': 'No anomalies detected'}

    def _generate_predictions(self, history: List[Dict], context: Dict) -> List[Dict]:
        """Generate specific predictions about user behavior"""

        predictions = []

        # Predict productivity recovery
        current_prod = context.get('productivity_score', 0.5)
        if current_prod < 0.4:
            recovery_prob = min(0.9, 0.3 + len(history) / 100)
            predictions.append({
                'type': 'productivity_recovery',
                'probability': recovery_prob,
                'timeframe': 'next_30_minutes',
                'description': 'Productivity likely to improve with appropriate intervention'
            })

        # Predict break need
        session_hours = context.get('session_duration_hours', 0)
        if session_hours > 3:  # increased from 2
            break_urgency = min(0.95, session_hours / 5)  # reduced urgency calculation
            predictions.append({
                'type': 'break_needed',
                'probability': break_urgency,
                'timeframe': 'immediate',
                'description': 'Break recommended to maintain productivity and well-being'
            })

        return predictions

    def _calculate_overall_risk(self, burnout_analysis: Dict, anomaly_analysis: Dict) -> str:
        """Calculate overall risk level"""

        burnout_risk = burnout_analysis.get('risk_score', 0.3)
        has_anomalies = anomaly_analysis.get('anomalies_detected', False)

        if burnout_risk > 0.7 or (has_anomalies and anomaly_analysis.get('severity') == 'high'):
            return 'high'
        elif burnout_risk > 0.5 or has_anomalies:
            return 'medium'
        else:
            return 'low'

    def _get_optimal_intervention_timing(self, user_id: str, intervention_type: str) -> Dict[str, Any]:
        """Get optimal timing for intervention based on user patterns"""

        profile = self._get_user_profile(user_id)
        current_hour = datetime.now().hour

        # Check user preferences
        if profile.preferences and profile.preferences.avoided_times:
            if current_hour in profile.preferences.avoided_times:
                return {
                    'delay_minutes': 60,
                    'reason': 'User prefers not to be disturbed at this time'
                }

        # Check energy patterns
        if profile.energy_patterns and current_hour in profile.energy_patterns:
            energy_level = profile.energy_patterns[current_hour]

            # Avoid interventions during low energy periods for productivity/focus
            if intervention_type in ['productivity_boost', 'focus_enhancement'] and energy_level < 0.3:
                return {
                    'delay_minutes': 30,
                    'reason': 'User typically has low energy at this time'
                }

        # Check recent intervention frequency
        recent_contexts = list(self.user_contexts[user_id])[-10:]
        recent_interventions = sum(
            1 for ctx in recent_contexts if ctx.get('intervention_received', False))

        if recent_interventions > 3:
            return {
                'delay_minutes': 45,
                'reason': 'Too many recent interventions - giving user space'
            }

        return {
            'delay_minutes': 0,
            'reason': 'Optimal timing for intervention'
        }

    # ========================================================================
    # PERSONA DETECTION AND PERSONALIZED COACHING
    # ========================================================================

    def _detect_user_persona(self, context: Dict) -> str:
        """Detect user persona based on application usage and context"""

        current_app = context.get('current_application', '').lower()
        current_window = context.get('current_window', '').lower()

        persona_scores = {'developer': 0, 'analyst': 0, 'manager': 0}

        # Score based on application usage
        for persona, patterns in self.persona_patterns.items():
            if persona == 'generic':
                continue
            for app in patterns['apps']:
                if app in current_app:
                    persona_scores[persona] += 2
                if app in current_window:
                    persona_scores[persona] += 1

            # Score based on keywords in window titles
            for keyword in patterns['keywords']:
                if keyword in current_window:
                    persona_scores[persona] += 1

        # Return highest scoring persona, or 'generic' if no clear winner
        if max(persona_scores.values()) >= 2:
            return max(persona_scores.items(), key=lambda x: x[1])[0]
        return 'generic'

    def _get_persona_specific_coaching(self, persona: str, context: Dict, analysis: Dict, user_id: str = "default") -> Optional[Dict]:
        """Generate persona-specific coaching"""

        # Determine coaching need
        productivity = analysis.get('productivity_score', 0.5)
        focus = analysis.get('focus_quality', 0.5)
        stress = analysis.get('stress_level', 0.5)
        energy = analysis.get('energy_level', 0.5)
        session_hours = analysis.get('session_duration_hours', analysis.get('session_hours', 0))

        coaching_type, urgency = self._determine_coaching_need(
            productivity, focus, stress, energy, session_hours)

        if not coaching_type:
            return None

        # Get persona-specific message
        persona_messages = self.persona_patterns.get(
            persona, self.persona_patterns['generic'])['coaching_messages']
        message_or_list = persona_messages.get(coaching_type,
                                               f"Consider optimizing your current work state for better {coaching_type.replace('_', ' ')}.")
        message = self._choose_copy_variant(message_or_list, user_id)
        message = self._adjust_message_for_context(message, context, urgency, coaching_type)

        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1

        # Allow medium urgency if not in meeting and within caps (suppression handled upstream)
        if urgency not in ['high', 'critical', 'medium']:
            return None

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message,
            'priority': priority,
            'urgency': urgency,
            'persona': persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'reasoning': f"Persona-specific {coaching_type} for {persona}",
                'confidence': 0.8 if persona != 'generic' else 0.6,
                'source': 'personalized',
                'cooldown_applied': False
            }
        }

    # ========================================================================
    # AI COACHING GENERATION (ANTHROPIC INTEGRATION)
    # ========================================================================

    async def _generate_ai_coaching(self, user_profile: UserProfile, context: Dict,
                                    coaching_strategy: Dict, predictive_insights: Dict) -> Optional[Dict]:
        """Generate AI coaching using Anthropic API with dynamic prompts"""

        try:
            # Determine prompt type
            if predictive_insights.get('burnout_risk', {}).get('risk_score', 0) > 0.6:
                prompt_type = 'burnout_prevention'
            else:
                prompt_type = 'productivity_analysis'

            # Generate dynamic prompt
            prompt = self._generate_dynamic_prompt(
                prompt_type, user_profile, context, predictive_insights)

            # Get AI response using Anthropic
            if self.claude_client:
                try:
                    message = await self.claude_client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=300,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    response_text = message.content[0].text.strip()

                    # Try to parse JSON response
                    try:
                        ai_coaching = json.loads(response_text)
                        if all(field in ai_coaching for field in ['message', 'priority', 'reasoning', 'confidence']):
                            return ai_coaching
                    except json.JSONDecodeError:
                        # Use text response as message
                        return {
                            'message': response_text,
                            'priority': coaching_strategy.get('urgency_level', 'medium') == 'high' and 3 or 2,
                            'reasoning': f"AI-generated {coaching_strategy['intervention_type']} advice",
                            'confidence': 0.7
                        }

                except Exception as e:
                    logger.warning(f"Anthropic API call failed: {e}")

            # Fallback to persona-specific coaching
            return self._generate_fallback_coaching(coaching_strategy, context, user_profile)

        except Exception as e:
            logger.error(f"AI coaching generation failed: {e}")
            return self._generate_fallback_coaching(coaching_strategy, context, user_profile)

    def _generate_dynamic_prompt(self, prompt_type: str, user_profile: UserProfile,
                                 context: Dict, predictive_insights: Dict) -> str:
        """Generate dynamic, personalized prompts for Anthropic API"""

        if prompt_type == 'burnout_prevention':
            prompt = f"""You are an expert wellness coach specializing in burnout prevention. Analyze the user's risk factors and provide preventive guidance.

User Profile:
- Persona: {user_profile.persona} (confidence: {user_profile.confidence_level:.1%})
- Productivity baseline: {user_profile.productivity_baseline:.1%}
- Total interactions: {user_profile.total_interactions}

Current Context:
- Current productivity: {context.get('productivity_score', 0.5):.1%}
- Current focus quality: {context.get('focus_quality', 0.5):.1%}
- Current stress level: {context.get('stress_level', 0.5):.1%}
- Session duration: {context.get('session_duration_hours', 0):.1f} hours
- Current application: {context.get('current_application', 'Unknown')}

Burnout Risk Analysis: {predictive_insights.get('burnout_risk', {})}

Provide specific, evidence-based advice to prevent burnout while maintaining productivity.

Respond in JSON format:
{{
    "message": "Caring, supportive message about burnout prevention",
    "priority": 2-3 (always medium-high for burnout),
    "reasoning": "Specific risk factors identified",
    "confidence": 0.0-1.0,
    "expected_impact": "How this will reduce burnout risk"
}}"""
        else:
            prompt = f"""You are an expert productivity coach analyzing telemetry data. Based on the user's current state and historical patterns, provide specific, actionable coaching advice.

User Profile:
- Persona: {user_profile.persona} (confidence: {user_profile.confidence_level:.1%})
- Productivity baseline: {user_profile.productivity_baseline:.1%}
- Focus baseline: {user_profile.focus_baseline:.1%}
- Total interactions: {user_profile.total_interactions}

Current Context:
- Current productivity: {context.get('productivity_score', 0.5):.1%}
- Current focus quality: {context.get('focus_quality', 0.5):.1%}
- Current stress level: {context.get('stress_level', 0.5):.1%}
- Current energy level: {context.get('energy_level', 0.5):.1%}
- Session duration: {context.get('session_duration_hours', 0):.1f} hours
- Current application: {context.get('current_application', 'Unknown')}

Predictive Insights: {predictive_insights}

Focus on immediate actionable advice based on current state and personalized recommendations based on user patterns.

Respond in JSON format:
{{
    "message": "Brief, actionable coaching message (1-2 sentences)",
    "priority": 1-3 (1=low, 2=medium, 3=urgent),
    "reasoning": "Why this advice is relevant now",
    "confidence": 0.0-1.0,
    "expected_impact": "What improvement this should achieve"
}}"""

        return prompt

    async def _get_anthropic_coaching(self, telemetry: Dict, coaching_type: str, urgency: str, user_id: str = "default") -> Optional[Dict]:
        """Get AI coaching from Anthropic Claude"""

        if not self.claude_client:
            return None

        try:
            context = self._build_simple_coaching_context(
                telemetry, coaching_type, urgency)

            message = await self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": context}]
            )

            ai_message = message.content[0].text.strip()
            return self._normalize_ai_response(coaching_type, urgency, ai_message, user_id=user_id)

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            return None

    def _build_simple_coaching_context(self, telemetry: Dict, coaching_type: str, urgency: str) -> str:
        """Build context-aware prompt for Anthropic"""

        productivity = telemetry.get('productivity_score', 0.5)
        focus = telemetry.get('focus_quality', 0.5)
        stress = telemetry.get('stress_level', 0.5)
        energy = telemetry.get('energy_level', 0.5)
        session_hours = telemetry.get('session_duration_hours', 0)
        current_app = telemetry.get('current_application', 'unknown')

        prompt = f"""You are an expert productivity coach. Based on the user's current state, provide a brief, actionable coaching message (1-2 sentences max).

Current metrics:
- Productivity: {productivity:.0%}
- Focus: {focus:.0%}
- Stress: {stress:.0%}
- Energy: {energy:.0%}
- Session duration: {session_hours:.1f} hours
- Current app: {current_app}

Coaching needed: {coaching_type} (urgency: {urgency})

Provide specific, actionable advice that addresses the {coaching_type} issue. Be encouraging and practical."""

        return prompt

    def _generate_fallback_coaching(self, coaching_strategy: Dict, context: Dict,
                                    user_profile: UserProfile) -> Dict:
        """Generate rule-based coaching when AI is unavailable"""

        intervention_type = coaching_strategy['intervention_type']

        # Use persona-specific messaging if available
        persona_messages = self.persona_patterns.get(user_profile.persona,
                                                     self.persona_patterns['generic'])['coaching_messages']

        message = persona_messages.get(intervention_type,
                                       f"Consider optimizing your current work state for better {intervention_type.replace('_', ' ')}.")

        return {
            'message': message,
            'priority': coaching_strategy.get('urgency_level', 'medium') == 'high' and 3 or 2,
            'reasoning': f"ML-enhanced {intervention_type} recommendation for {user_profile.persona}",
            'confidence': 0.6
        }

    # ========================================================================
    # BASIC ANALYSIS AND RULE-BASED COACHING
    # ========================================================================

    def _determine_coaching_need(self, productivity: float, focus: float,
                                 stress: float, energy: float, session_hours: float) -> tuple:
        """Determine if coaching is needed and what type"""

        # Priority order: stress > energy > productivity > focus > breaks
        if stress >= self.coaching_strategies['stress_thresholds']['high']:
            return 'stress_reduction', 'high'
        elif stress >= self.coaching_strategies['stress_thresholds']['moderate']:
            return 'stress_reduction', 'medium'

        if energy <= self.coaching_strategies['energy_thresholds']['critical']:
            return 'energy_boost', 'high'
        elif energy <= self.coaching_strategies['energy_thresholds']['low']:
            return 'energy_boost', 'medium'

        if productivity <= 0.15:  # Very low productivity (lowered from 0.2)
            return 'productivity_boost', 'high'
        elif productivity <= self.coaching_strategies['productivity_thresholds']['low']:
            return 'productivity_boost', 'medium'

        if focus <= 0.25:  # Very low focus (lowered from 0.3)
            return 'focus_enhancement', 'medium'
        elif focus <= self.coaching_strategies['focus_thresholds']['low']:
            return 'focus_enhancement', 'low'

        if session_hours > 4:  # Long session (increased from 3)
            return 'break_reminder', 'medium'
        elif session_hours > 3:   # increased from 2
            return 'break_reminder', 'low'

        return None, None  # No coaching needed

    def _get_rule_based_coaching(self, coaching_type: str, urgency: str, telemetry: Dict, user_id: str = "default") -> Dict:
        """Get rule-based coaching advice"""

        templates = {
            'productivity_boost': {
                'very_low': "🎯 Focus boost needed! Try the Pomodoro technique: 25 minutes of focused work, then a 5-minute break.",
                'low': "Your productivity seems low. Consider focusing on your most important task for the next 25 minutes."
            },
            'focus_enhancement': {
                'distracted': "🧘 Focus time! Close distractions and concentrate on one task for improved productivity.",
                'scattered': "You've been switching between applications. Consider closing unnecessary tabs/apps for better focus."
            },
            'stress_reduction': {
                'high': "⚠️ High stress detected! Consider taking a 5-10 minute break to reset and recharge.",
                'moderate': "Your stress level seems elevated. Take 3 deep breaths or step away for a 2-minute break."
            },
            'energy_boost': {
                'critical': "💪 Energy boost time! Try some light stretching or a quick walk to restore your energy.",
                'low': "Your energy seems low. Stand up, stretch, or take a brief walk to reinvigorate yourself."
            },
            'break_reminder': {
                'urgent': "🛑 Break time! You've been working for a while. Take 5-10 minutes to rest and recharge.",
                'gentle': "You've been working steadily. A short 5-minute break can help maintain your productivity."
            }
        }

        # Select appropriate template based on urgency and specific metrics
        template_group = templates.get(coaching_type, {})

        if coaching_type == 'productivity_boost':
            productivity = telemetry.get('productivity_score', 0.5)
            message = template_group.get('very_low' if productivity < 0.2 else 'low',
                                         "Focus on your most important task to boost productivity.")

        elif coaching_type == 'focus_enhancement':
            message = template_group.get('distracted' if urgency == 'high' else 'scattered',
                                         "Close unnecessary applications to improve focus.")

        elif coaching_type == 'stress_reduction':
            message = template_group.get('high' if urgency == 'high' else 'moderate',
                                         "Take deep breaths and consider a brief break.")

        elif coaching_type == 'energy_boost':
            energy = telemetry.get('energy_level', 0.5)
            message = template_group.get('critical' if energy < 0.2 else 'low',
                                         "Stand up and stretch to boost your energy.")

        elif coaching_type == 'break_reminder':
            session_hours = telemetry.get('session_duration_hours', 0)
            message = template_group.get('urgent' if session_hours > 3 else 'gentle',
                                         "Consider taking a short break to maintain productivity.")

        else:
            message = "Take a moment to assess your current work state and make any needed adjustments."

        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message,
            'priority': priority,
            'urgency': urgency,
            'persona': self._get_user_profile(user_id).persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'reasoning': f"Rule-based {coaching_type} recommendation",
                'confidence': 0.6,
                'source': 'rule_based',
                'cooldown_applied': False
            }
        }

    def _basic_analysis(self, telemetry_data: Dict) -> Dict:
        """Perform basic analysis on telemetry data"""
        return {
            'productivity_score': telemetry_data.get('productivity_score', 0.5),
            'focus_quality': telemetry_data.get('focus_quality', 0.5),
            'stress_level': telemetry_data.get('stress_level', 0.5),
            'energy_level': telemetry_data.get('energy_level', 0.5),
            'session_duration_hours': telemetry_data.get('session_duration_hours', telemetry_data.get('session_hours', 0)),
            'activity_level': telemetry_data.get('activity_level', 'MEDIUM'),
            'current_app': telemetry_data.get('current_app', 'Unknown'),
            'in_call': telemetry_data.get('in_call', False)
        }

    def _extract_context(self, telemetry_data: Dict, analysis: Dict) -> Dict:
        """Extract context from telemetry data and analysis"""
        return {
            'current_application': analysis.get('current_app', 'Unknown'),
            'productivity_score': analysis.get('productivity_score', 0.5),
            'focus_quality': analysis.get('focus_quality', 0.5),
            'stress_level': analysis.get('stress_level', 0.5),
            'energy_level': analysis.get('energy_level', 0.5),
            'session_duration_hours': analysis.get('session_duration_hours', analysis.get('session_hours', 0)),
            'activity_level': analysis.get('activity_level', 'MEDIUM'),
            'in_meeting': analysis.get('in_call', False),
            'keyboard_count': telemetry_data.get('total_keystrokes', 0),
            'mouse_count': telemetry_data.get('total_mouse_events', 0),
            'current_window': telemetry_data.get('current_window', ''),
            'break_taken': telemetry_data.get('break_taken', False) or self._infer_break_taken(telemetry_data.get('event_buffer', []))
        }

    # ========================================================================
    # ML PATTERN LEARNING AND PREDICTION
    # ========================================================================

    def _predict_intervention_effectiveness(self, context: Dict, intervention: Dict) -> float:
        """Predict how effective an intervention will be"""

        if not self.pattern_learner_enabled or len(self.interaction_data) < 5:
            return self._rule_based_effectiveness_prediction(context, intervention)

        try:
            features = self._extract_features(context, intervention)
            features_scaled = self.scaler.transform([features])
            prediction = self.effectiveness_predictor.predict_proba(features_scaled)[
                0]
            return prediction[1] if len(prediction) > 1 else 0.5
        except Exception as e:
            logger.warning(f"ML effectiveness prediction failed: {e}")
            return self._rule_based_effectiveness_prediction(context, intervention)

    def _rule_based_effectiveness_prediction(self, context: Dict, intervention: Dict) -> float:
        """Simple rule-based effectiveness prediction"""

        intervention_type = intervention.get(
            'intervention_type', intervention.get('nudge_type', 'productivity_boost'))
        productivity = context.get('productivity_score', 0.5)
        focus = context.get('focus_quality', 0.5)
        stress = context.get('stress_level', 0.5)
        energy = context.get('energy_level', 0.5)

        # Simple rules based on context-intervention match
        if intervention_type == 'productivity_boost':
            return max(0.2, 1.0 - productivity)
        elif intervention_type == 'focus_enhancement':
            return max(0.2, 1.0 - focus)
        elif intervention_type == 'stress_reduction':
            return max(0.2, stress)
        elif intervention_type == 'energy_boost':
            return max(0.2, 1.0 - energy)

        return 0.5

    def _extract_features(self, context: Dict, intervention: Dict) -> List[float]:
        """Extract numerical features for ML models"""

        features = [
            context.get('productivity_score', 0.5),
            context.get('focus_quality', 0.5),
            context.get('stress_level', 0.5),
            context.get('energy_level', 0.5),
            context.get('session_duration_hours', 0),
            intervention.get('priority', 2),
            datetime.now().hour / 24.0,
            len(context.get('current_application', '')) / 50.0,
        ]

        # Intervention type encoding
        intervention_types = ['productivity_boost', 'focus_enhancement',
                              'stress_reduction', 'energy_boost', 'break_reminder']
        intervention_type = (
            intervention.get('type')
            or intervention.get('intervention_type')
            or intervention.get('nudge_type', 'productivity_boost')
        )
        for i, itype in enumerate(intervention_types):
            features.append(1.0 if intervention_type == itype else 0.0)

        # persona one-hot
        persona = (intervention.get('persona') or context.get('persona') or 'generic')
        features.extend([
            1.0 if persona == 'developer' else 0.0,
            1.0 if persona == 'analyst' else 0.0,
            1.0 if persona == 'manager' else 0.0,
        ])

        # contextual flags
        features.append(1.0 if context.get('in_meeting') else 0.0)

        # message length (scaled)
        features.append(len(intervention.get('message','')) / 200.0)

        # urgency flags
        urg = intervention.get('urgency', 'medium')
        features.extend([
            1.0 if urg == 'high' else 0.0,
            1.0 if urg == 'medium' else 0.0,
        ])

        # hour bucket (coarse)
        hr = int((datetime.now().hour // 6))  # 0..3
        for b in range(4):
            features.append(1.0 if hr == b else 0.0)

        # optional: variant id (if available)
        features.append(float(intervention.get('meta', {}).get('variant_id', 0)) / 10.0)

        return features

    def learn_from_interaction(self, context: Dict, intervention: Dict,
                               effectiveness_score: float, response_time_seconds: float):
        """Learn from a single coaching interaction"""

        interaction_record = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'intervention': intervention,
            'effectiveness_score': effectiveness_score,
            'response_time_seconds': response_time_seconds,
            'user_id': intervention.get('user_id', 'default')
        }

        self.interaction_data.append(interaction_record)

        # Retrain models periodically
        if self.pattern_learner_enabled and len(self.interaction_data) % 10 == 0:
            self._retrain_models()

        # Save interaction data
        self._save_interaction_record(interaction_record)

    def _retrain_models(self):
        """Retrain ML models with accumulated data"""

        if not self.pattern_learner_enabled or len(self.interaction_data) < 5:
            return

        try:
            # Prepare training data
            X = []
            y_effectiveness = []

            for record in self.interaction_data:
                features = self._extract_features(
                    record['context'], record['intervention'])
                X.append(features)
                y_effectiveness.append(
                    1 if record['effectiveness_score'] > 0.6 else 0)

            if len(set(y_effectiveness)) > 1:  # Need multiple classes
                # Scale features
                X_scaled = self.scaler.fit_transform(X)

                # Train effectiveness predictor
                self.effectiveness_predictor.fit(X_scaled, y_effectiveness)

                self.model_performance['last_trained'] = datetime.now(
                ).isoformat()

                logger.info(
                    f"Retrained ML models with {len(self.interaction_data)} interactions")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    # ========================================================================
    # FEEDBACK COLLECTION AND LEARNING
    # ========================================================================

    def _record_intervention_for_feedback(self, intervention_id: str, user_id: str,
                                          intervention_type: str, message: str, context: Dict,
                                          priority: int = 2) -> None:
        """Record an intervention for feedback tracking"""

        self.intervention_contexts[intervention_id] = {
            'user_id': user_id,
            'type': intervention_type,
            'message': message,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'post_contexts': []
        }

        logger.info(
            f"Recorded intervention {intervention_id} for feedback tracking")

    def record_post_intervention_context(self, intervention_id: str, context: Dict) -> None:
        """Record context after intervention for behavioral analysis"""

        if intervention_id in self.intervention_contexts:
            self.intervention_contexts[intervention_id]['post_contexts'].append({
                **context,
                'timestamp': datetime.now().isoformat()
            })

    def analyze_intervention_effectiveness(self, intervention_id: str) -> Optional[FeedbackEntry]:
        """Analyze intervention effectiveness using behavioral data"""

        if intervention_id not in self.intervention_contexts:
            return None

        intervention_data = self.intervention_contexts[intervention_id]
        post_contexts = intervention_data['post_contexts']

        # Skip analysis if intervention is too recent
        intervention_time = datetime.fromisoformat(
            intervention_data['timestamp'])
        if datetime.now() - intervention_time < timedelta(minutes=10):
            return None

        # Skip if no post-intervention data
        if not post_contexts:
            return None

        # Calculate behavioral effectiveness
        before_context = intervention_data['context']
        effectiveness_score = self._calculate_behavioral_effectiveness(
            before_context, post_contexts, intervention_data['type'])

        # Create feedback entry
        feedback_entry = FeedbackEntry(
            intervention_id=intervention_id,
            user_id=intervention_data['user_id'],
            timestamp=datetime.now().isoformat(),
            intervention_type=intervention_data['type'],
            intervention_message=intervention_data['message'],
            feedback_method='implicit',
            effectiveness_score=effectiveness_score,
            response_time_seconds=(
                datetime.now() - intervention_time).total_seconds(),
            context_at_intervention=intervention_data['context'],
            behavioral_response={
                'effectiveness_calculated': effectiveness_score}
        )

        # Store feedback and learn from it
        self.feedback_history.append(feedback_entry)
        self._save_feedback_entry(feedback_entry)

        # Update user model
        self._record_intervention_feedback(
            intervention_data['user_id'], intervention_data['type'],
            effectiveness_score, feedback_entry.response_time_seconds)

        # Learn from this interaction
        if self.pattern_learner_enabled:
            self.learn_from_interaction(
                intervention_data['context'],
                {'intervention_type': intervention_data['type'],
                    'priority': intervention_data['priority']},
                effectiveness_score,
                feedback_entry.response_time_seconds
            )

        # Clean up intervention context
        del self.intervention_contexts[intervention_id]

        logger.info(
            f"Analyzed intervention {intervention_id}: effectiveness={effectiveness_score:.2f}")
        return feedback_entry

    def _calculate_behavioral_effectiveness(self, before_context: Dict,
                                            after_contexts: List[Dict], intervention_type: str) -> float:
        """Calculate effectiveness based on behavioral changes"""

        if not after_contexts:
            return 0.5

        # Take average of next few measurements
        after_avg = {}
        for key in ['productivity_score', 'focus_quality', 'stress_level', 'energy_level']:
            values = [ctx.get(key, 0.5)
                      for ctx in after_contexts if key in ctx]
            after_avg[key] = sum(values) / len(values) if values else 0.5

        # Calculate improvements based on intervention type
        if intervention_type in ['productivity_boost', 'focus_enhancement']:
            prod_improvement = after_avg['productivity_score'] - \
                before_context.get('productivity_score', 0.5)
            focus_improvement = after_avg['focus_quality'] - \
                before_context.get('focus_quality', 0.5)
            return max(0.2, min(1.0, 0.5 + prod_improvement + focus_improvement))

        elif intervention_type == 'stress_reduction':
            stress_reduction = before_context.get(
                'stress_level', 0.5) - after_avg['stress_level']
            return max(0.2, min(1.0, 0.5 + stress_reduction))

        elif intervention_type == 'energy_boost':
            energy_improvement = after_avg['energy_level'] - \
                before_context.get('energy_level', 0.5)
            return max(0.2, min(1.0, 0.5 + energy_improvement))

        return 0.5

    def _record_intervention_feedback(self, user_id: str, intervention_type: str,
                                      effectiveness_score: float, response_time_seconds: float):
        """Record feedback about intervention effectiveness for user model"""
        profile = self._get_user_profile(user_id)

        if profile.intervention_effectiveness is None:
            profile.intervention_effectiveness = {}

        # Update effectiveness with exponential moving average
        if intervention_type in profile.intervention_effectiveness:
            current_eff = profile.intervention_effectiveness[intervention_type]
            profile.intervention_effectiveness[intervention_type] = 0.7 * \
                current_eff + 0.3 * effectiveness_score
        else:
            profile.intervention_effectiveness[intervention_type] = effectiveness_score

        self._save_user_profile(profile)

    # ========================================================================
    # UTILITY METHODS FOR PERSONALIZED SCORING
    # ========================================================================

    def calculate_personalized_productivity_score(self, event_buffer: List[Dict], context: Dict, user_id: str = "default") -> float:
        """Calculate personalized productivity score with user model adjustments"""

        if not event_buffer:
            return 0.5

        # Base calculation
        activities = [e.get('keyboard_count', 0) +
                      e.get('mouse_count', 0) for e in event_buffer]
        if not activities:
            return 0.5

        avg_activity = sum(activities) / len(activities)

        # Calculate base score
        if avg_activity > 100:
            base_score = min(1.0, 0.7 + (avg_activity - 100) / 300)
        elif avg_activity < 20:
            base_score = max(0.2, avg_activity / 40)
        else:
            base_score = 0.5 + (avg_activity - 20) / 160

        # Personalize based on user profile
        user_profile = self._get_user_profile(user_id)

        # Persona-specific adjustments
        persona = self._detect_user_persona(context)
        current_app = context.get('current_application', '').lower()

        # Developers: coding apps = higher baseline
        if persona == 'developer' and any(app in current_app for app in ['vscode', 'intellij', 'pycharm', 'terminal']):
            base_score = min(1.0, base_score + 0.1)

        # Analysts: data apps = higher baseline
        elif persona == 'analyst' and any(app in current_app for app in ['excel', 'tableau', 'jupyter', 'r studio']):
            base_score = min(1.0, base_score + 0.1)

        # Managers: communication apps during work hours = normal productivity
        elif persona == 'manager' and any(app in current_app for app in ['slack', 'teams', 'outlook']):
            if 9 <= datetime.now().hour <= 17:
                base_score = min(1.0, base_score + 0.05)

        # Adjust based on user's personal baseline
        personalized_score = (base_score * 0.7) + \
            (user_profile.productivity_baseline * 0.3)

        return max(0.0, min(1.0, personalized_score))

    def calculate_personalized_focus_quality(self, event_buffer: List[Dict], context: Dict, user_id: str = "default") -> float:
        """Calculate personalized focus quality with user model awareness"""

        if not event_buffer:
            return 0.5

        # Analyze app switching patterns
        apps = [e.get('process_name', '') for e in event_buffer]
        unique_apps = len(set(apps))

        # Base focus calculation
        if unique_apps == 1:
            base_focus = 0.9
        elif unique_apps <= 2:
            base_focus = 0.7
        else:
            base_focus = max(0.3, 1.0 - (unique_apps * 0.15))

        # Persona-specific adjustments
        persona = self._detect_user_persona(context)
        user_profile = self._get_user_profile(user_id)

        # Developers: expect fewer app switches
        if persona == 'developer' and unique_apps > 3:
            base_focus *= 0.8

        # Managers: more app switching is normal
        elif persona == 'manager' and unique_apps <= 4:
            base_focus = min(1.0, base_focus + 0.1)

        # Personalize based on user's focus baseline
        personalized_focus = (base_focus * 0.8) + \
            (user_profile.focus_baseline * 0.2)

        return max(0.0, min(1.0, personalized_focus))

    def detect_user_persona(self, context: Dict) -> str:
        """Public interface for persona detection"""
        return self._detect_user_persona(context)

    def get_persona_specific_coaching(self, persona: str, context: Dict, analysis: Dict) -> Optional[Dict]:
        """Public interface for persona-specific coaching"""
        return self._get_persona_specific_coaching(persona, context, analysis)

    # ========================================================================
    # SYSTEM INFORMATION AND INSIGHTS
    # ========================================================================

    def get_user_insights(self, user_id: str = "default") -> Dict[str, Any]:
        """Get comprehensive insights about user and system performance"""

        profile = self._get_user_profile(user_id)
        contexts = list(self.user_contexts.get(user_id, []))

        # Recent trends
        recent_contexts = contexts[-20:] if len(contexts) >= 20 else contexts
        trends = {}
        if recent_contexts:
            recent_productivity = [
                ctx.get('productivity_score', 0.5) for ctx in recent_contexts]
            recent_focus = [ctx.get('focus_quality', 0.5)
                            for ctx in recent_contexts]
            recent_stress = [ctx.get('stress_level', 0.5)
                             for ctx in recent_contexts]

            trends = {
                'productivity_trend': 'improving' if len(recent_productivity) > 5 and recent_productivity[-1] > recent_productivity[0] else 'stable',
                'focus_trend': 'improving' if len(recent_focus) > 5 and recent_focus[-1] > recent_focus[0] else 'stable',
                'stress_trend': 'improving' if len(recent_stress) > 5 and recent_stress[-1] < recent_stress[0] else 'stable'
            }

        return {
            'profile': {
                'persona': profile.persona,
                'confidence': profile.confidence_level,
                'productivity_baseline': profile.productivity_baseline,
                'focus_baseline': profile.focus_baseline,
                'total_interactions': profile.total_interactions
            },
            'patterns': {
                'energy_patterns': profile.energy_patterns or {},
                'productivity_patterns': profile.productivity_patterns or {},
                'intervention_effectiveness': profile.intervention_effectiveness or {}
            },
            'recent_trends': {
                **trends,
                'data_points': len(recent_contexts)
            },
            'learning_performance': {
                'total_feedback_points': len(self.feedback_history),
                'total_interactions': len(self.interaction_data),
                'learning_trend': {'direction': 'stable'}
            },
            'pattern_recognition': {
                'total_interactions': len(self.interaction_data),
                'pattern_quality': 'good' if len(self.interaction_data) > 20 else 'developing'
            },
            'overall_performance': {
                'total_interventions': len(self.intervention_history),
                'ml_confidence': self._calculate_overall_ml_confidence(),
                'personalization_level': self._calculate_personalization_level(user_id)
            }
        }

    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the coaching system capabilities"""

        return {
            'coach_type': 'enhanced_ml' if 'ml_pattern_learning' in self.capabilities else
            'personalized' if 'persona_detection' in self.capabilities else 'base',
            'capabilities': self.capabilities,
            'anthropic_api_available': self.claude_client is not None,
            'ml_libraries_available': {
                'pandas': PANDAS_AVAILABLE,
                'sklearn': SKLEARN_AVAILABLE,
                'anthropic': ANTHROPIC_AVAILABLE
            },
            'data_stats': {
                'user_profiles': len(self.user_profiles),
                'interaction_data': len(self.interaction_data),
                'feedback_history': len(self.feedback_history),
                'intervention_history': len(self.intervention_history)
            }
        }

    def _calculate_overall_ml_confidence(self) -> float:
        """Calculate overall confidence in ML predictions"""
        components = []

        # Pattern learner confidence
        if len(self.interaction_data) > 20:
            components.append(0.8)
        elif len(self.interaction_data) > 5:
            components.append(0.6)
        else:
            components.append(0.3)

        # Feedback system confidence
        if len(self.feedback_history) > 10:
            components.append(0.7)
        else:
            components.append(0.4)

        return sum(components) / len(components) if components else 0.5

    def _calculate_personalization_level(self, user_id: str) -> float:
        """Calculate how personalized the coaching is for this user"""
        user_profile = self._get_user_profile(user_id)

        factors = [
            user_profile.confidence_level,  # Persona confidence
            # Interaction history
            min(1.0, user_profile.total_interactions / 50),
            # Effectiveness data
            min(1.0, len(user_profile.intervention_effectiveness or {}) / 5)
        ]

        return sum(factors) / len(factors)

    def _infer_break_taken(self, event_buffer: List[Dict]) -> bool:
        """Infer if user has taken a break from activity gaps"""
        if not event_buffer:
            return False
        
        # Consider a break if >= 5 consecutive minutes with near-zero input
        recent = event_buffer[-10:]  # last ~10 minutes
        idle_minutes = sum(1 for e in recent if (e.get('keyboard_count', 0) + e.get('mouse_count', 0)) < 5)
        return idle_minutes >= 5

    # ========================================================================
    # NOTIFICATION SUPPRESSION AND COOLDOWN SYSTEM
    # ========================================================================

    def _should_suppress_notification(self, user_id: str, intervention: Dict, context: Dict) -> Tuple[bool, str]:
        """Check if notification should be suppressed based on cooldowns, meeting state, etc."""
        now = datetime.now()
        
        # 1) In-meeting suppression for non-critical
        if self.notification_config.get('suppress_in_meeting', True):
            if context.get(self.keys['in_meeting'], False):
                if intervention['type'] not in self.notification_config.get('allow_in_meeting_types', []):
                    return True, 'in_meeting_suppression'
        
        # 2) Global per-hour cap
        recent = [i for i in self.intervention_history.values()
                  if i.get('user_id') == user_id and
                     datetime.fromisoformat(i.get('timestamp', now.isoformat())) > now - timedelta(hours=1)]
        if len(recent) >= self.notification_config.get('max_per_hour', 4):
            return True, 'hourly_cap'
        
        # 3) Global cooldown
        min_gap = self.notification_config.get('min_minutes_between', 12)
        if recent:
            last_ts = max(datetime.fromisoformat(i.get('timestamp', now.isoformat())) for i in recent)
            if (now - last_ts).total_seconds() < min_gap * 60:
                return True, 'global_cooldown'
        
        # 4) Per-type cooldown
        per_type = self.notification_config.get('per_type_cooldown_minutes', {})
        tcd = per_type.get(intervention['type'])
        if tcd:
            pertype_recent = [i for i in recent if i.get('type') == intervention['type']]
            if pertype_recent:
                last_t = max(datetime.fromisoformat(i.get('timestamp', now.isoformat())) for i in pertype_recent)
                if (now - last_t).total_seconds() < tcd * 60:
                    return True, 'type_cooldown'
        
        # 5) Repeat-text suppression
        rpt = self.notification_config.get('repeat_suppression_minutes', 90)
        for i in recent:
            if i.get('message') == intervention['message']:
                if (now - datetime.fromisoformat(i.get('timestamp', now.isoformat()))).total_seconds() < rpt * 60:
                    return True, 'repeat_suppression'
        
        return False, ''

    def _envelope(self, *, coaching_type: str, urgency: str, message: str,
                  persona: str, priority: Optional[int] = None,
                  source: str = 'rule_based', confidence: float = 0.6,
                  channel: Optional[str] = None, context: Optional[Dict] = None,
                  meta_extra: Optional[Dict]=None) -> Dict:
        """Create standardized notification envelope"""
        prio = priority if priority is not None else (3 if urgency == 'high' else 2 if urgency == 'medium' else 1)
        ctx = context or {}
        selected_channel = channel or self._select_channel(urgency, ctx)
        
        env = {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': message[:500],
            'priority': prio,
            'urgency': urgency,
            'persona': persona,
            'channel': selected_channel,
            'interaction': self._build_interaction(coaching_type, urgency, ctx),
            'meta': {
                'confidence': max(0.0, min(1.0, confidence)),
                'reasoning': f"{source} {coaching_type} advice",
                'source': source,
                'cooldown_applied': False
            }
        }
        if meta_extra:
            env['meta'].update(meta_extra)
        return env

    def _choose_copy_variant(self, texts: Any, user_id: str) -> str:
        """Choose copy variant deterministically but varied by user & hour"""
        if not isinstance(texts, list):
            return str(texts or "")
        if not texts:
            return ""
        # deterministic but varied by user & hour
        seed = (hash(user_id) + datetime.now().hour) % len(texts)
        return texts[seed]

    def _adjust_message_for_context(self, message: str, context: Dict, urgency: str, coaching_type: str) -> str:
        """Adjust message phrasing based on context and urgency"""
        msg = message

        # Meeting softening
        if context.get('in_meeting'):
            replace_map = {
                " now": " when the meeting wraps",
                "Now ": "When the meeting wraps ",
                "Take 5-10 minutes": "Plan 5–10 minutes next",
                "Stand up": "Plan to stand up",
                "Start ": "Queue "
            }
            for k, v in replace_map.items():
                msg = msg.replace(k, v)

        # Energy-sensitive tone
        if coaching_type in ('productivity_boost','focus_enhancement') and context.get('energy_level', 0.5) < 0.2:
            msg = "Restore first: 2–3 minutes gentle movement or hydration. " + msg

        # Explicitly name long sessions
        sd = context.get('session_duration_hours', 0)
        if coaching_type == 'break_reminder' and sd >= 3.0:
            msg = f"You've been at it for {sd:.1f} hours. " + msg

        return msg

    def _build_interaction(self, coaching_type: str, urgency: str, context: Dict) -> Dict:
        """Build interaction metadata (title and CTA)"""
        if coaching_type == 'productivity_boost':
            return {'title': "Focus for 25 minutes",
                    'cta': {'label': "Start Timer", 'action': 'start_pomodoro', 'payload': {'minutes': 25}}}
        if coaching_type == 'break_reminder':
            return {'title': "Take a 5–10 min break",
                    'cta': {'label': "Snooze 10 min", 'action': 'snooze', 'payload': {'minutes': 10}}}
        if coaching_type == 'stress_reduction':
            return {'title': "Quick reset",
                    'cta': {'label': "Open Breathing", 'action': 'open_breathing', 'payload': {'minutes': 2}}}
        if coaching_type == 'focus_enhancement':
            return {'title': "Eliminate distractions",
                    'cta': {'label': "Do Not Disturb", 'action': 'enable_dnd', 'payload': {'minutes': 30}}}
        if coaching_type == 'energy_boost':
            return {'title': "Restore energy",
                    'cta': {'label': "Quick Walk", 'action': 'start_movement', 'payload': {'minutes': 5}}}
        return {'title': "Heads‑up", 'cta': None}

    def _select_channel(self, urgency: str, context: Dict) -> str:
        """Select notification channel based on urgency and context"""
        if context.get('in_meeting'):
            return 'system_banner'  # soft in meeting
        return 'modal' if urgency == 'high' else 'toast' if urgency == 'medium' else 'system_banner'

    def _normalize_ai_response(self, coaching_type: str, urgency: str, raw_text: str, user_id: str = "default") -> Dict:
        """Normalize AI response to standard schema with validation"""
        priority = 3 if urgency == 'high' else 2 if urgency == 'medium' else 1
        
        try:
            payload = json.loads(raw_text)
            msg = payload.get('message') or raw_text
            prio = int(payload.get('priority', priority))
            conf = float(payload.get('confidence', 0.7))
            reasoning = payload.get('reasoning', f"AI-generated {coaching_type} advice")
        except Exception:
            msg, prio, conf, reasoning = raw_text, priority, 0.7, f"AI-generated {coaching_type} advice"

        return {
            'id': str(uuid.uuid4()),
            'type': coaching_type,
            'message': msg[:500],  # guardrail length
            'priority': min(3, max(1, prio)),
            'urgency': urgency,
            'persona': self._get_user_profile(user_id).persona,
            'channel': self.notification_config.get('default_channel', 'system_banner'),
            'meta': {
                'confidence': max(0.0, min(1.0, conf)),
                'reasoning': reasoning,
                'source': 'anthropic_ai',
                'cooldown_applied': False
            }
        }

    def _log_notification_event(self, event: str, user_id: str, intervention: Dict, reason: str = ""):
        """Log notification events for analytics"""
        try:
            path = self.data_dir / 'notification_events.jsonl'
            payload = {
                'timestamp': datetime.now().isoformat(),
                'event': event,  # 'sent', 'suppressed', 'clicked', 'dismissed', 'snoozed'
                'user_id': user_id,
                'type': intervention.get('type'),
                'urgency': intervention.get('urgency'),
                'persona': intervention.get('persona'),
                'channel': intervention.get('channel'),
                'reason': reason,
                'message_hash': intervention.get('meta', {}).get('message_hash')
            }
            with open(path, 'a') as f:
                f.write(json.dumps(payload) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log notification event: {e}")

    def _record_intervention(self, intervention: Dict, user_id: str = "default"):
        """Record intervention in history for cooldown tracking"""
        intervention['timestamp'] = datetime.now().isoformat()
        intervention['user_id'] = user_id
        intervention_id = intervention.get('id', str(uuid.uuid4()))
        intervention['id'] = intervention_id
        self.intervention_history[intervention_id] = intervention

    # ========================================================================
    # DATA PERSISTENCE
    # ========================================================================

    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to disk"""
        try:
            profile_file = self.data_dir / f"profile_{profile.user_id}.json"
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save user profile {profile.user_id}: {e}")

    def _load_user_profiles(self):
        """Load user profiles from disk"""
        try:
            for profile_file in self.data_dir.glob("profile_*.json"):
                with open(profile_file, 'r') as f:
                    data = json.load(f)

                    # Reconstruct UserProfile
                    if 'preferences' in data and data['preferences']:
                        data['preferences'] = UserPreferences(
                            **data['preferences'])
                    else:
                        data['preferences'] = UserPreferences()

                    profile = UserProfile(**data)
                    self.user_profiles[profile.user_id] = profile

            logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.warning(f"Failed to load user profiles: {e}")

    def _save_interaction_record(self, record: Dict):
        """Save interaction record to disk"""
        try:
            interactions_file = self.data_dir / 'interactions.jsonl'
            with open(interactions_file, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save interaction record: {e}")

    def _load_training_data(self):
        """Load existing training data"""
        try:
            interactions_file = self.data_dir / 'interactions.jsonl'
            if interactions_file.exists():
                with open(interactions_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            self.interaction_data.append(record)

                logger.info(
                    f"Loaded {len(self.interaction_data)} interaction records")
        except Exception as e:
            logger.warning(f"Failed to load training data: {e}")

    def _save_feedback_entry(self, entry: FeedbackEntry):
        """Save feedback entry to disk"""
        try:
            feedback_file = self.data_dir / 'feedback_history.jsonl'
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(asdict(entry), default=str) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback entry: {e}")

    def _load_feedback_history(self):
        """Load feedback history from disk"""
        try:
            feedback_file = self.data_dir / 'feedback_history.jsonl'
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry_dict = json.loads(line)
                            entry = FeedbackEntry(**entry_dict)
                            self.feedback_history.append(entry)

                logger.info(
                    f"Loaded {len(self.feedback_history)} feedback entries")
        except Exception as e:
            logger.warning(f"Failed to load feedback history: {e}")

# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================


__all__ = ['AICoach']
