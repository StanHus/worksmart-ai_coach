#!/usr/bin/env python3
"""
WorkSmart Enhanced AI Coach Launcher
====================================

Enhanced version with persistent session tracking and date-based logging.
Maintains productivity data across restarts and provides historical analysis.
"""

from ..core.telemetry import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer
from ..core.personalized_coach import PersonalizedAICoach
from ..core.coach import AICoach
import asyncio
import json
import os
import sys
import time
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables from .env file


def load_env_file():
    """Load environment variables from .env file in ~/.worksmart-ai-coach/"""
    env_file = os.path.expanduser("~/.worksmart-ai-coach/.env")
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Handle tilde expansion for paths
                    if key.endswith('_PATH') and value.startswith('~'):
                        value = os.path.expanduser(value)
                    os.environ[key] = value
        print("✅ Environment variables loaded from ~/.worksmart-ai-coach/.env")
    else:
        print("⚠️ No .env file found at ~/.worksmart-ai-coach/.env")


# Load environment before importing AI coach components
load_env_file()

# Import our AI coach components


class EnhancedProductionLauncher:
    """Enhanced production launcher with persistent session tracking"""

    def __init__(self):
        self.coach = PersonalizedAICoach()  # Use personalized coach instead
        self.telemetry_collector = WorkSmartTelemetryCollector()
        self.telemetry_analyzer = WorkSmartTelemetryAnalyzer()
        self.running = False
        self.java_process = None

        # Single file naming (no date suffix)
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.session_file = "worksmart_session.json"
        self.coaching_log_file = "coaching_log.json"
        self.daily_stats_file = "daily_stats.json"

        # Load or initialize persistent session
        self.session_data = self.load_session_data()

    def load_session_data(self):
        """Load persistent session data for today"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                
                # Check if session is from a different day
                session_date = data.get('date', '')
                if session_date != self.today:
                    print(f"📅 New day detected ({session_date} → {self.today}), creating fresh session")
                    self._cleanup_old_logs()  # Clean up old logs on new day
                    return self._create_new_session()
                
                # Same day - accumulate previous session time and start fresh timer
                if 'session_start' in data:
                    old_start = datetime.fromisoformat(data['session_start'])
                    previous_session_hours = (datetime.now() - old_start).total_seconds() / 3600
                    accumulated_hours = data.get('accumulated_hours_today', 0)
                    data['accumulated_hours_today'] = accumulated_hours + previous_session_hours
                    data['session_start'] = datetime.now().isoformat()  # Reset timer for this restart
                    print(f"📅 Restarted - accumulated {data['accumulated_hours_today']:.1f}h today")
                
                # Remove legacy fields that are now handled by WorkSmart
                legacy_fields = ['apps_used',
                                 'total_keystrokes', 'total_mouse_events']
                for field in legacy_fields:
                    data.pop(field, None)
                return data
            except:
                pass
        
        return self._create_new_session()

    def _create_new_session(self):
        """Create a fresh session for today"""
        # Minimal session data - WorkSmart handles the rest
        session_data = {
            "date": self.today,
            "session_start": datetime.now().isoformat(),
            "accumulated_hours_today": 0,  # Track total hours across restarts
            "coaching_count": 0,
            "last_activity": datetime.now().isoformat(),
            "productivity_scores": [],
            "focus_scores": []
        }
        print(f"📅 Created new session data for {self.today}")
        return session_data

    def _cleanup_old_logs(self):
        """Clean up old log files to prevent accumulation"""
        files_to_clean = [
            self.coaching_log_file,
            self.daily_stats_file,
            "context_history.json"
        ]
        
        cleaned_count = 0
        for file_path in files_to_clean:
            try:
                if os.path.exists(file_path):
                    # Keep only today's entries
                    self._filter_log_to_today(file_path)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Warning: Could not clean {file_path}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} log files for new day")

    def _filter_log_to_today(self, file_path: str):
        """Filter log file to keep only today's entries"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # If it's a list of log entries
            if isinstance(data, list):
                today_entries = [
                    entry for entry in data 
                    if entry.get('date') == self.today or 
                       (entry.get('timestamp', '').startswith(self.today))
                ]
                
                with open(file_path, 'w') as f:
                    json.dump(today_entries, f, indent=2)
                    
            # If it's a single object with date field, reset it if old
            elif isinstance(data, dict) and data.get('date') != self.today:
                # Reset the file for new day
                os.remove(file_path)
                
        except Exception as e:
            # If file is corrupted or unreadable, just remove it
            try:
                os.remove(file_path)
            except:
                pass

    def save_session_data(self):
        """Save minimal session data to file"""
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            print(f"Error saving session data: {e}")

    def update_session_stats(self, event):
        """Use WorkSmart official statistics only"""
        # Get WorkSmart official data
        worksmart_stats = self.telemetry_collector.worksmart_reader.get_current_session_stats()
        worksmart_analysis = self.telemetry_collector.worksmart_reader.analyze_productivity_patterns(
            hours=8)

        # Update event with WorkSmart official data only
        event['worksmart_hours_today'] = worksmart_stats.get(
            'total_hours_today', '0:0')
        event['worksmart_session_active'] = worksmart_stats.get(
            'session_active', False)
        event['worksmart_total_keystrokes'] = worksmart_analysis.get(
            'total_keystrokes', 0)
        event['worksmart_total_clicks'] = worksmart_analysis.get(
            'total_mouse_clicks', 0)
        event['worksmart_apps_count'] = 1  # Current app

        # Use AI Coach session duration for today's total coaching time
        # If restarted multiple times today, continue from where we left off
        start_time = datetime.fromisoformat(self.session_data['session_start'])
        current_session_hours = (datetime.now() - start_time).total_seconds() / 3600
        
        # Add any accumulated time from previous restarts today
        accumulated_hours = self.session_data.get('accumulated_hours_today', 0)
        total_coaching_hours_today = accumulated_hours + current_session_hours
        
        event['coaching_session_hours'] = total_coaching_hours_today

        return event

    def check_worksmart_running(self):
        """Check if WorkSmart Java tracker is running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'virtualoffice'],
                                    capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except:
            return False

    async def monitor_loop(self):
        """Enhanced monitoring loop with persistent tracking"""
        print("🤖 Enhanced AI Coach monitoring started", flush=True)
        print(f"📊 Date: {self.today}", flush=True)
        print(
            f"📈 Session started at: {self.session_data['session_start'][:16]}", flush=True)
        print(
            f"💾 Data files: {self.session_file}, {self.coaching_log_file}", flush=True)
        print("⌨️  Press Ctrl+C to stop\n", flush=True)

        event_buffer = []
        analysis_interval = 6  # Analyze every 6 events (10 minutes)

        while self.running:
            try:
                # Collect telemetry event
                event = self.telemetry_collector.collect_event()

                # Update with persistent session data
                event = self.update_session_stats(event)

                event_buffer.append(event)

                # Display current telemetry (simplified)
                app_name = event['process_name'][:20] if len(event['process_name']) > 20 else event['process_name']
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {app_name} | Activity: {event['keyboard_count']}🔤 {event['mouse_count']}🖱️ | Session: {event.get('coaching_session_hours', 0):.1f}h")

                # Save updated session data
                self.save_session_data()

                # Analyze when buffer is full
                if len(event_buffer) >= analysis_interval:
                    await self.analyze_and_coach(event_buffer)
                    # Keep only recent events for continuity
                    event_buffer = event_buffer[-3:]

                # Wait 100 seconds (WorkSmart's standard interval)
                await asyncio.sleep(100)

            except KeyboardInterrupt:
                self._save_session_on_shutdown()
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        self._save_session_on_shutdown()
        print("🛑 AI Coach monitoring stopped")
        print(f"📊 Final session stats saved to {self.session_file}")

    def _save_session_on_shutdown(self):
        """Save accumulated time before shutdown for restarts"""
        try:
            start_time = datetime.fromisoformat(self.session_data['session_start'])
            current_session_hours = (datetime.now() - start_time).total_seconds() / 3600
            accumulated_hours = self.session_data.get('accumulated_hours_today', 0)
            self.session_data['accumulated_hours_today'] = accumulated_hours + current_session_hours
            self.save_session_data()
            print(f"💾 Saved {self.session_data['accumulated_hours_today']:.1f}h total for today")
        except Exception as e:
            print(f"⚠️ Error saving session on shutdown: {e}")

    async def analyze_and_coach(self, event_buffer):
        """Enhanced analysis with personalized coaching"""
        # Analyze silently

        # Analyze telemetry pattern
        analysis = self.telemetry_analyzer.analyze(event_buffer)

        # Add historical context
        analysis['cumulative_session_hours'] = event_buffer[-1].get(
            'cumulative_session_hours', 0)
        analysis['total_apps_today'] = event_buffer[-1].get(
            'total_apps_today', 0)
        analysis['coaching_count_today'] = self.session_data['coaching_count']

        # Get latest event context for personalized analysis
        latest_event = event_buffer[-1] if event_buffer else {}
        context = {
            'current_application': analysis['current_app'],
            'current_window': latest_event.get('current_window', ''),
            'chrome_context': latest_event.get('chrome_context', {}),
            'file_activity': latest_event.get('file_activity', {}),
            'window_count': latest_event.get('window_count', 0),
            # Enhanced context for advanced pattern detection
            'session_duration_hours': latest_event.get('session_duration_hours', 0),
            'keyboard_count': latest_event.get('keyboard_count', 0),
            'mouse_count': latest_event.get('mouse_count', 0),
            'activity_level': analysis['activity_level']
        }

        # Use personalized algorithms
        personal_productivity = self.coach.calculate_personalized_productivity_score(
            event_buffer, context)
        personal_focus = self.coach.calculate_personalized_focus_quality(
            event_buffer, context)

        # Update analysis with personalized scores
        analysis['productivity_score'] = personal_productivity
        analysis['focus_quality'] = personal_focus

        # Only show if productivity is critically low
        if analysis['productivity_score'] < 0.15:
            print(f"\n🚨 Critically low productivity: {analysis['productivity_score']:.0%}")
        elif analysis['focus_quality'] < 0.2:
            print(f"\n🚨 Critically low focus: {analysis['focus_quality']:.0%}")

        # Store scores for historical analysis
        self.session_data['productivity_scores'].append(
            analysis['productivity_score'])
        self.session_data['focus_scores'].append(analysis['focus_quality'])

        # Get personalized coaching with persona detection
        persona = self.coach.detect_user_persona(context)
        persona_coaching = self.coach.get_persona_specific_coaching(persona, context, analysis)
        
        if persona_coaching:
            # Use persona-specific coaching from research
            coaching_result = persona_coaching
            if persona != 'generic':
                print(f"🎭 Using {persona} persona coaching")
        else:
            # NO fallback to generic AI coaching - it generates useless alerts
            coaching_result = None

        if coaching_result:
            message = coaching_result.get('message', 'No specific advice')
            priority = coaching_result.get('priority', 2)
            reasoning = coaching_result.get('reasoning', 'Generic advice')
            detailed_guidance = coaching_result.get('detailed_guidance', '')
            confidence = coaching_result.get('confidence', 0)
            timing = coaching_result.get('timing', 'immediate')
            impact_prediction = coaching_result.get('impact_prediction', '')

            # Show all priority messages with Nudge DNA structure
            if priority >= 1:
                priority_icons = {1: "🔔", 2: "💡", 3: "🚨"}
                icon = priority_icons.get(priority, "💡")
                
                # Nudge DNA: Confidence badge + Expected benefit + Trigger explanation
                confidence_badge = self._get_confidence_badge(confidence)
                expected_benefit = self._calculate_expected_benefit(message, priority)
                trigger_explanation = self._get_trigger_explanation(reasoning)
                
                print(f"\n{icon} {message}")
                print(f"   🎯 {confidence_badge} | ⚡ Expected benefit: {expected_benefit}")
                print(f"   📋 Trigger: {trigger_explanation}")
                
                if detailed_guidance and priority >= 2:
                    print(f"   → {detailed_guidance}")
                
                # Only desktop notifications for absolute emergencies (priority 3 + very bad metrics)
                is_emergency = (priority == 3 and 
                              analysis.get('productivity_score', 1.0) < 0.1 and
                              self._is_smart_timing_appropriate(priority))
                
                if is_emergency:
                    notification_message = detailed_guidance if detailed_guidance else message
                    self.show_coaching_notification(notification_message, priority)

            # Log coaching with enhanced data
            self.log_coaching(coaching_result, analysis)

            # Update coaching count
            self.session_data['coaching_count'] += 1
        # Update daily stats
        self.update_daily_stats(analysis)

    def _get_confidence_badge(self, confidence: float) -> str:
        """Generate confidence badge for trust calibration"""
        if confidence >= 0.8:
            return "High confidence"
        elif confidence >= 0.5:
            return "Medium confidence" 
        else:
            return "Low confidence"

    def _calculate_expected_benefit(self, message: str, priority: int) -> str:
        """Calculate expected benefit quantified in time saved"""
        if "break" in message.lower():
            return "5-15 min energy recovery"
        elif "focus" in message.lower() or "tab" in message.lower():
            return "20-40% efficiency gain"
        elif "stress" in message.lower():
            return "Reduced burnout risk"
        elif priority == 3:
            return "Critical productivity protection"
        else:
            return "2-5 min improvement"

    def _get_trigger_explanation(self, reasoning: str) -> str:
        """Provide trigger explanation for transparency"""
        if not reasoning:
            return "Pattern-based recommendation"
        
        # Simplify technical reasoning for user transparency
        simplified = reasoning.replace("productivity_score", "productivity")
        simplified = simplified.replace("focus_quality", "focus level")
        simplified = simplified.replace("session_minutes", "session time")
        return simplified[:60] + "..." if len(simplified) > 60 else simplified

    def _is_smart_timing_appropriate(self, priority: int) -> bool:
        """Smart timing: avoid first hour, lunch, post-5pm unless urgent"""
        current_hour = datetime.now().hour
        
        # Always allow urgent notifications
        if priority == 3:
            return True
        
        # Avoid: first hour (8-9am), lunch (12-1pm), evening (5-6pm)
        if current_hour in [8, 12, 17]:
            return False
        
        # Avoid very late/early hours
        if current_hour < 7 or current_hour > 20:
            return False
            
        return True

    def show_coaching_notification(self, message, priority):
        """Show macOS desktop notification with multiple fallback methods"""
        try:
            title_map = {1: "URGENT - AI Coach",
                         2: "AI Coach Alert", 3: "AI Coach Tip"}
            title = title_map.get(priority, "AI Coach")

            # Clean message for notification - be more careful with quotes
            clean_message = message.replace('"', "'").replace(
                '\n', ' ').replace('\\', '').strip()[:120]

            import subprocess

            # Method 1: Try terminal-notifier first (more reliable)
            try:
                result = subprocess.run(['terminal-notifier', '-title', title, '-message', clean_message, '-sound', 'Ping'],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Method 2: AppleScript with better escaping
            escaped_message = clean_message.replace(
                '"', '\\"').replace("'", "\\'")
            escaped_title = title.replace('"', '\\"').replace("'", "\\'")

            script = f'''
            display notification "{escaped_message}" with title "{escaped_title}" sound name "Ping"
            delay 0.1
            '''

            result = subprocess.run(['osascript', '-e', script],
                                    capture_output=True, text=True, timeout=5)

            # Notification attempted

        except Exception:
            pass

    def show_log_status(self):
        """Show current log file status"""
        log_files = [
            self.coaching_log_file,
            self.daily_stats_file,
            self.session_file
        ]
        
        print(f"\n📁 Log Files Status:")
        total_size = 0
        for file_path in log_files:
            if os.path.exists(file_path):
                size_kb = os.path.getsize(file_path) / 1024
                total_size += size_kb
                
                # Count entries if it's a JSON array
                entry_count = "N/A"
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        entry_count = len(data)
                    elif isinstance(data, dict):
                        entry_count = "1 object"
                except:
                    pass
                
                print(f"   📄 {file_path}: {size_kb:.1f}KB, {entry_count} entries")
            else:
                print(f"   📄 {file_path}: Not found")
        
        print(f"   💾 Total log size: {total_size:.1f}KB")
        return total_size

    def log_coaching(self, coaching_result, analysis):
        """Log coaching to date-based file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "date": self.today,
            "coaching": coaching_result,
            "analysis": analysis,
            "session_context": {
                "cumulative_hours": analysis.get('cumulative_session_hours', 0),
                "total_apps": analysis.get('total_apps_today', 0),
                "coaching_count": self.session_data['coaching_count']
            },
            "source": "enhanced_ai_coach"
        }

        # Load existing log
        logs = []
        if os.path.exists(self.coaching_log_file):
            try:
                with open(self.coaching_log_file, 'r') as f:
                    logs = json.load(f)
            except:
                pass

        # Add new entry
        logs.append(log_entry)

        # Save log
        with open(self.coaching_log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def update_daily_stats(self, analysis):
        """Update daily statistics summary"""
        # Get WorkSmart official stats instead
        worksmart_stats = self.telemetry_collector.worksmart_reader.get_current_session_stats()
        worksmart_analysis = self.telemetry_collector.worksmart_reader.analyze_productivity_patterns(
            hours=8)

        stats = {
            "date": self.today,
            "last_updated": datetime.now().isoformat(),
            "total_session_hours": analysis.get('cumulative_session_hours', 0),
            "productivity_average": sum(self.session_data['productivity_scores']) / len(self.session_data['productivity_scores']) if self.session_data['productivity_scores'] else 0,
            "focus_average": sum(self.session_data['focus_scores']) / len(self.session_data['focus_scores']) if self.session_data['focus_scores'] else 0,
            "worksmart_hours_today": worksmart_stats.get('total_hours_today', '0:0'),
            "worksmart_total_keystrokes": worksmart_analysis.get('total_keystrokes', 0),
            "worksmart_total_clicks": worksmart_analysis.get('total_mouse_clicks', 0),
            "coaching_sessions": self.session_data['coaching_count']
        }

        with open(self.daily_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    async def start_production_system(self):
        """Start the enhanced production system"""
        print("🚀 WORKSMART ENHANCED AI COACH", flush=True)
        print("="*60, flush=True)
        print(
            f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"📅 Session date: {self.today}", flush=True)
        print(flush=True)

        print("🔗 Setting up enhanced AI coach integration...", flush=True)

        # Setup signal handlers
        def signal_handler(sig, frame):
            print(f"\n🛑 Received signal {sig} - shutting down...")
            self.save_session_data()
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.running = True

        # Start monitoring
        await self.monitor_loop()

        # Final save
        self.save_session_data()

        return True

    def status_report(self):
        """Enhanced status report with historical data"""
        print("📊 ENHANCED AI COACH STATUS")
        print("="*50)
        print(f"Date: {self.today}")
        print(
            f"WorkSmart Running: {'✅' if self.check_worksmart_running() else '❌'}")
        print(
            f"AI Coach Status: {'🟢 Active' if self.running else '🔴 Inactive'}")

        # Get WorkSmart stats
        worksmart_stats = self.telemetry_collector.worksmart_reader.get_current_session_stats()
        print(
            f"WorkSmart Hours Today: {worksmart_stats.get('total_hours_today', '0:0')}")

        # Session stats
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r') as f:
                session = json.load(f)

            start_time = datetime.fromisoformat(session['session_start'])
            duration = (datetime.now() - start_time).total_seconds() / 3600

            print(f"Coaching Session Duration: {duration:.2f} hours")
            print(f"Coaching Sessions: {session['coaching_count']}")

        # Daily stats
        if os.path.exists(self.daily_stats_file):
            with open(self.daily_stats_file, 'r') as f:
                stats = json.load(f)

            print(
                f"Avg Productivity: {stats.get('productivity_average', 0):.1%}")
            print(f"Avg Focus: {stats.get('focus_average', 0):.1%}")


async def main():
    """Main entry point"""
    launcher = EnhancedProductionLauncher()

    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            launcher.status_report()
        elif sys.argv[1] == "stats":
            # Show daily statistics
            if os.path.exists(launcher.daily_stats_file):
                with open(launcher.daily_stats_file, 'r') as f:
                    stats = json.load(f)
                print(json.dumps(stats, indent=2))
            else:
                print("No stats available yet")
        elif sys.argv[1] == "help":
            print("WorkSmart Enhanced AI Coach")
            print()
            print("Usage:")
            print(
                "  python3 worksmart_enhanced_launcher.py        # Start enhanced system")
            print("  python3 worksmart_enhanced_launcher.py status  # Show status")
            print("  python3 worksmart_enhanced_launcher.py stats   # Show daily stats")
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        # Start enhanced system
        success = await launcher.start_production_system()
        sys.exit(0 if success else 1)


def main():
    """Main entry point for worksmart-enhanced command"""
    try:
        import asyncio
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def async_main():
    """Async main entry point"""
    launcher = EnhancedProductionLauncher()

    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            launcher.status_report()
        elif sys.argv[1] == "stats":
            # Show daily statistics
            if os.path.exists(launcher.daily_stats_file):
                with open(launcher.daily_stats_file, 'r') as f:
                    stats = json.load(f)
                print(json.dumps(stats, indent=2))
            else:
                print("No stats available yet")
        elif sys.argv[1] == "help":
            print("WorkSmart Enhanced AI Coach")
            print()
            print("Usage:")
            print("  worksmart-enhanced        # Start enhanced system")
            print("  worksmart-enhanced status  # Show status")
            print("  worksmart-enhanced stats   # Show daily stats")
        else:
            print(f"Unknown command: {sys.argv[1]}")
    else:
        # Start enhanced system
        success = await launcher.start_production_system()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
