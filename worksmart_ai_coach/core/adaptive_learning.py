#!/usr/bin/env python3
"""
Adaptive Learning System
========================
Learns from coaching effectiveness to improve future interventions.
Tracks intervention outcomes and adapts coaching strategies.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class InterventionOutcome:
    """Track outcomes of coaching interventions"""
    intervention_id: str
    intervention_type: str
    timestamp: str
    context_before: Dict
    productivity_before: float
    focus_before: float
    message: str
    reasoning: str
    
    # Outcome measurements
    productivity_after_5min: Optional[float] = None
    productivity_after_15min: Optional[float] = None
    focus_after_5min: Optional[float] = None
    focus_after_15min: Optional[float] = None
    user_followed_suggestion: Optional[bool] = None
    effectiveness_score: Optional[float] = None  # 0.0 - 1.0
    
    # Learning metadata
    pattern_matched: Optional[str] = None
    success_indicators: List[str] = None
    failure_indicators: List[str] = None

class AdaptiveLearningSystem:
    """Adaptive learning system for coaching effectiveness"""
    
    def __init__(self):
        self.intervention_outcomes = []
        self.learning_patterns = defaultdict(list)
        self.effectiveness_stats = {}
        self.adaptation_rules = {}
        self.load_learning_data()
        
    def track_intervention(self, intervention_data: Dict, context: Dict, 
                          analysis: Dict) -> str:
        """Track new intervention for effectiveness learning"""
        
        intervention_id = f"intervention_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        outcome = InterventionOutcome(
            intervention_id=intervention_id,
            intervention_type=intervention_data.get('action', 'unknown'),
            timestamp=datetime.now().isoformat(),
            context_before=context.copy(),
            productivity_before=analysis.get('productivity_score', 0.5),
            focus_before=analysis.get('focus_quality', 0.5),
            message=intervention_data.get('message', ''),
            reasoning=intervention_data.get('reasoning', ''),
            pattern_matched=intervention_data.get('context_data', {}).get('pattern_type'),
            success_indicators=[],
            failure_indicators=[]
        )
        
        self.intervention_outcomes.append(outcome)
        print(f"📊 Tracking intervention {intervention_id} for effectiveness learning")
        
        return intervention_id
    
    def update_outcome(self, intervention_id: str, productivity_score: float,
                      focus_score: float, context: Dict, minutes_elapsed: int) -> None:
        """Update intervention outcome with follow-up measurements"""
        
        # Find the intervention
        outcome = None
        for o in self.intervention_outcomes:
            if o.intervention_id == intervention_id:
                outcome = o
                break
        
        if not outcome:
            return
        
        # Update based on time elapsed
        if minutes_elapsed <= 7:  # 5-minute follow-up
            outcome.productivity_after_5min = productivity_score
            outcome.focus_after_5min = focus_score
        elif minutes_elapsed <= 17:  # 15-minute follow-up
            outcome.productivity_after_15min = productivity_score
            outcome.focus_after_15min = focus_score
            
            # Calculate effectiveness after 15 minutes
            self._calculate_effectiveness(outcome, context)
        
        print(f"📈 Updated outcome for {intervention_id}: productivity={productivity_score:.2f}, focus={focus_score:.2f}")
    
    def _calculate_effectiveness(self, outcome: InterventionOutcome, current_context: Dict) -> None:
        """Calculate intervention effectiveness score"""
        
        # Productivity improvement
        prod_improvement_5min = 0
        prod_improvement_15min = 0
        
        if outcome.productivity_after_5min:
            prod_improvement_5min = outcome.productivity_after_5min - outcome.productivity_before
        if outcome.productivity_after_15min:
            prod_improvement_15min = outcome.productivity_after_15min - outcome.productivity_before
        
        # Focus improvement
        focus_improvement_5min = 0
        focus_improvement_15min = 0
        
        if outcome.focus_after_5min:
            focus_improvement_5min = outcome.focus_after_5min - outcome.focus_before
        if outcome.focus_after_15min:
            focus_improvement_15min = outcome.focus_after_15min - outcome.focus_before
        
        # Check if user followed suggestions
        followed_suggestion = self._check_suggestion_compliance(outcome, current_context)
        outcome.user_followed_suggestion = followed_suggestion
        
        # Calculate composite effectiveness score
        effectiveness = 0.0
        
        # Weight productivity improvement (40%)
        if prod_improvement_15min > 0.1:
            effectiveness += 0.4 * min(1.0, prod_improvement_15min / 0.3)
        elif prod_improvement_5min > 0.1:
            effectiveness += 0.3 * min(1.0, prod_improvement_5min / 0.2)
        
        # Weight focus improvement (30%)
        if focus_improvement_15min > 0.1:
            effectiveness += 0.3 * min(1.0, focus_improvement_15min / 0.3)
        elif focus_improvement_5min > 0.1:
            effectiveness += 0.2 * min(1.0, focus_improvement_5min / 0.2)
        
        # Weight suggestion compliance (30%)
        if followed_suggestion:
            effectiveness += 0.3
        
        outcome.effectiveness_score = min(1.0, effectiveness)
        
        # Record success/failure indicators
        if effectiveness > 0.6:
            outcome.success_indicators = self._identify_success_factors(outcome, current_context)
        else:
            outcome.failure_indicators = self._identify_failure_factors(outcome, current_context)
        
        # Update learning patterns
        self._update_learning_patterns(outcome)
        
        print(f"🎯 Effectiveness calculated: {outcome.effectiveness_score:.2f} for {outcome.intervention_type}")
    
    def _check_suggestion_compliance(self, outcome: InterventionOutcome, 
                                   current_context: Dict) -> bool:
        """Check if user followed the intervention suggestion"""
        
        intervention_type = outcome.intervention_type
        
        # AI tool suggestions
        if 'ai_tool' in intervention_type or 'ChatGPT' in outcome.message or 'Grok' in outcome.message:
            current_app = current_context.get('current_application', '')
            current_url = current_context.get('chrome_context', {}).get('active_tab_url', '')
            
            ai_tools = ['ChatGPT', 'Terminal']
            ai_urls = ['grok.com', 'chat.openai.com', 'openai.com']
            
            return (current_app in ai_tools or 
                   any(url in current_url for url in ai_urls))
        
        # Tab consolidation suggestions
        if 'tab' in intervention_type or 'consolidat' in outcome.message.lower():
            current_tabs = current_context.get('chrome_context', {}).get('total_tabs', 0)
            before_tabs = outcome.context_before.get('chrome_context', {}).get('total_tabs', 0)
            
            return current_tabs < before_tabs  # Tab count reduced
        
        # Focus suggestions
        if 'focus' in intervention_type:
            # Check for reduced app switching
            return True  # Simplified for now
        
        return None  # Cannot determine compliance
    
    def _identify_success_factors(self, outcome: InterventionOutcome, 
                                 current_context: Dict) -> List[str]:
        """Identify factors that contributed to successful intervention"""
        factors = []
        
        # Context factors
        before_app = outcome.context_before.get('current_application', '')
        current_app = current_context.get('current_application', '')
        
        if before_app != current_app:
            factors.append(f"app_switch_from_{before_app}_to_{current_app}")
        
        # Timing factors
        intervention_hour = datetime.fromisoformat(outcome.timestamp).hour
        if 9 <= intervention_hour <= 11:
            factors.append("morning_timing")
        elif 13 <= intervention_hour <= 15:
            factors.append("afternoon_timing")
        
        # Productivity level at intervention
        if outcome.productivity_before < 0.5:
            factors.append("low_productivity_intervention")
        elif outcome.productivity_before > 0.7:
            factors.append("high_productivity_intervention")
        
        # Message characteristics
        if len(outcome.message) < 100:
            factors.append("concise_message")
        if 'AI' in outcome.message or 'ChatGPT' in outcome.message:
            factors.append("ai_tool_suggestion")
        
        return factors
    
    def _identify_failure_factors(self, outcome: InterventionOutcome,
                                 current_context: Dict) -> List[str]:
        """Identify factors that contributed to intervention failure"""
        factors = []
        
        # No productivity improvement
        if (outcome.productivity_after_15min and 
            outcome.productivity_after_15min <= outcome.productivity_before):
            factors.append("no_productivity_improvement")
        
        # No focus improvement
        if (outcome.focus_after_15min and
            outcome.focus_after_15min <= outcome.focus_before):
            factors.append("no_focus_improvement")
        
        # User ignored suggestions
        if outcome.user_followed_suggestion is False:
            factors.append("ignored_suggestions")
        
        # Timing issues
        intervention_hour = datetime.fromisoformat(outcome.timestamp).hour
        if intervention_hour < 9 or intervention_hour > 17:
            factors.append("off_hours_timing")
        
        # Message too long
        if len(outcome.message) > 150:
            factors.append("verbose_message")
        
        return factors
    
    def _update_learning_patterns(self, outcome: InterventionOutcome) -> None:
        """Update learning patterns based on intervention outcome"""
        
        pattern_key = outcome.intervention_type
        effectiveness = outcome.effectiveness_score or 0.0
        
        self.learning_patterns[pattern_key].append({
            'effectiveness': effectiveness,
            'timestamp': outcome.timestamp,
            'context_factors': {
                'app': outcome.context_before.get('current_application'),
                'productivity_before': outcome.productivity_before,
                'focus_before': outcome.focus_before,
                'hour': datetime.fromisoformat(outcome.timestamp).hour
            },
            'success_factors': outcome.success_indicators or [],
            'failure_factors': outcome.failure_indicators or []
        })
        
        # Update adaptation rules
        self._generate_adaptation_rules(pattern_key)
    
    def _generate_adaptation_rules(self, intervention_type: str) -> None:
        """Generate adaptation rules from learning patterns"""
        
        outcomes = self.learning_patterns[intervention_type]
        if len(outcomes) < 3:  # Need minimum data
            return
        
        # Calculate average effectiveness
        avg_effectiveness = sum(o['effectiveness'] for o in outcomes) / len(outcomes)
        
        # Find success patterns
        successful_outcomes = [o for o in outcomes if o['effectiveness'] > 0.6]
        failed_outcomes = [o for o in outcomes if o['effectiveness'] < 0.4]
        
        rules = {
            'avg_effectiveness': avg_effectiveness,
            'total_attempts': len(outcomes),
            'success_rate': len(successful_outcomes) / len(outcomes)
        }
        
        # Timing rules
        if successful_outcomes:
            successful_hours = [o['context_factors']['hour'] for o in successful_outcomes]
            most_successful_hour = max(set(successful_hours), key=successful_hours.count)
            rules['best_timing_hour'] = most_successful_hour
        
        # Context rules
        successful_apps = [o['context_factors']['app'] for o in successful_outcomes if o['context_factors']['app']]
        if successful_apps:
            rules['effective_contexts'] = list(set(successful_apps))
        
        # Productivity level rules
        successful_prod_levels = [o['context_factors']['productivity_before'] for o in successful_outcomes]
        if successful_prod_levels:
            rules['optimal_productivity_range'] = (min(successful_prod_levels), max(successful_prod_levels))
        
        self.adaptation_rules[intervention_type] = rules
        print(f"📚 Generated adaptation rules for {intervention_type}: {avg_effectiveness:.2f} avg effectiveness")
    
    def get_adaptation_recommendation(self, intervention_type: str, 
                                    context: Dict, analysis: Dict) -> Optional[Dict]:
        """Get adaptation recommendation for proposed intervention"""
        
        if intervention_type not in self.adaptation_rules:
            return None
        
        rules = self.adaptation_rules[intervention_type]
        recommendations = {}
        
        # Check timing
        current_hour = datetime.now().hour
        if 'best_timing_hour' in rules and abs(current_hour - rules['best_timing_hour']) > 2:
            recommendations['timing_warning'] = f"This intervention works best around {rules['best_timing_hour']:02d}:00"
        
        # Check context
        current_app = context.get('current_application', '')
        if 'effective_contexts' in rules and current_app not in rules['effective_contexts']:
            recommendations['context_suggestion'] = f"More effective in: {', '.join(rules['effective_contexts'])}"
        
        # Check productivity level
        current_productivity = analysis.get('productivity_score', 0.5)
        if 'optimal_productivity_range' in rules:
            min_prod, max_prod = rules['optimal_productivity_range']
            if not (min_prod <= current_productivity <= max_prod):
                recommendations['productivity_warning'] = f"Most effective when productivity is {min_prod:.1f}-{max_prod:.1f}"
        
        # Overall effectiveness prediction
        predicted_effectiveness = rules['avg_effectiveness']
        if len(recommendations) == 0:
            predicted_effectiveness *= 1.1  # Boost if conditions are optimal
        else:
            predicted_effectiveness *= 0.8  # Reduce if conditions are suboptimal
        
        return {
            'predicted_effectiveness': min(1.0, predicted_effectiveness),
            'success_rate': rules['success_rate'],
            'recommendations': recommendations,
            'confidence': min(1.0, rules['total_attempts'] / 10)  # More attempts = higher confidence
        }
    
    def save_learning_data(self) -> None:
        """Save learning data to file"""
        data = {
            'saved_at': datetime.now().isoformat(),
            'intervention_outcomes': [asdict(outcome) for outcome in self.intervention_outcomes],
            'learning_patterns': dict(self.learning_patterns),
            'adaptation_rules': self.adaptation_rules,
            'effectiveness_stats': self.effectiveness_stats
        }
        
        filename = "adaptive_learning.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_learning_data(self) -> None:
        """Load learning data from file"""
        filename = "adaptive_learning.json"
        
        if not os.path.exists(filename):
            return
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Restore intervention outcomes
            for outcome_dict in data.get('intervention_outcomes', []):
                outcome = InterventionOutcome(**outcome_dict)
                self.intervention_outcomes.append(outcome)
            
            # Restore learning patterns and rules
            self.learning_patterns = defaultdict(list, data.get('learning_patterns', {}))
            self.adaptation_rules = data.get('adaptation_rules', {})
            self.effectiveness_stats = data.get('effectiveness_stats', {})
            
            print(f"📚 Loaded adaptive learning data: {len(self.intervention_outcomes)} outcomes")
            
        except Exception as e:
            print(f"⚠️ Error loading learning data: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning progress"""
        if not self.intervention_outcomes:
            return {"status": "no_data"}
        
        # Calculate overall effectiveness
        completed_outcomes = [o for o in self.intervention_outcomes if o.effectiveness_score is not None]
        
        if not completed_outcomes:
            return {"status": "learning_in_progress", "tracked_interventions": len(self.intervention_outcomes)}
        
        avg_effectiveness = sum(o.effectiveness_score for o in completed_outcomes) / len(completed_outcomes)
        
        # Breakdown by intervention type
        type_breakdown = defaultdict(list)
        for outcome in completed_outcomes:
            type_breakdown[outcome.intervention_type].append(outcome.effectiveness_score)
        
        type_stats = {}
        for int_type, scores in type_breakdown.items():
            type_stats[int_type] = {
                'avg_effectiveness': sum(scores) / len(scores),
                'attempts': len(scores),
                'success_rate': len([s for s in scores if s > 0.6]) / len(scores)
            }
        
        return {
            'status': 'learning_active',
            'total_outcomes': len(completed_outcomes),
            'avg_effectiveness': avg_effectiveness,
            'intervention_types': len(type_stats),
            'type_breakdown': type_stats,
            'adaptation_rules_generated': len(self.adaptation_rules)
        }