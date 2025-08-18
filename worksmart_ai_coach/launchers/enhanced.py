#!/usr/bin/env python3
"""
WorkSmart Enhanced AI Coach Launcher
====================================

Enhanced version with persistent session tracking and date-based logging.
Maintains productivity data across restarts and provides historical analysis.
"""

import asyncio
import json
import os
import sys
import time
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Import our AI coach components
from ..core.coach import AICoach
from ..core.personalized_coach import PersonalizedAICoach
from ..core.telemetry import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer

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
                # Remove legacy fields that are now handled by WorkSmart
                legacy_fields = ['apps_used', 'total_keystrokes', 'total_mouse_events']
                for field in legacy_fields:
                    data.pop(field, None)
                print(f"📅 Loaded existing session data")
                return data
            except:
                pass
        
        # Minimal session data - WorkSmart handles the rest
        session_data = {
            "date": self.today,
            "session_start": datetime.now().isoformat(),
            "coaching_count": 0,
            "last_activity": datetime.now().isoformat(),
            "productivity_scores": [],
            "focus_scores": []
        }
        print(f"📅 Created new session data")
        return session_data
    
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
        worksmart_analysis = self.telemetry_collector.worksmart_reader.analyze_productivity_patterns(hours=8)
        
        # Update event with WorkSmart official data only
        event['worksmart_hours_today'] = worksmart_stats.get('total_hours_today', '0:0')
        event['worksmart_session_active'] = worksmart_stats.get('session_active', False)
        event['worksmart_total_keystrokes'] = worksmart_analysis.get('total_keystrokes', 0)
        event['worksmart_total_clicks'] = worksmart_analysis.get('total_mouse_clicks', 0)
        event['worksmart_apps_count'] = 1  # Current app
        
        # Use AI Coach session duration only for current coaching session context
        start_time = datetime.fromisoformat(self.session_data['session_start'])
        coaching_session_hours = (datetime.now() - start_time).total_seconds() / 3600
        event['coaching_session_hours'] = coaching_session_hours
        
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
        print(f"📈 Session started at: {self.session_data['session_start'][:16]}", flush=True)
        print(f"💾 Data files: {self.session_file}, {self.coaching_log_file}", flush=True)
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
                
                # Display current telemetry with enhanced info
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 Activity captured:")
                print(f"  App: {event['process_name']}")
                if event.get('visiting_url'):
                    print(f"  🌐 URL: {event['visiting_url'][:60]}...")
                    print(f"  📄 Tab: {event['current_window']}")
                else:
                    print(f"  Window: {event['current_window'][:50]}...")
                print(f"  Coaching Session: {event.get('coaching_session_hours', 0):.2f}h")
                print(f"  WorkSmart Today: {event.get('worksmart_hours_today', '0:0')}")
                print(f"  Activity: {event['keyboard_count']}🔤 {event['mouse_count']}🖱️")
                print(f"  WorkSmart Total: {event.get('worksmart_total_keystrokes', 0)}🔤 | {event.get('worksmart_apps_count', 1)} apps")
                
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
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        print("🛑 AI Coach monitoring stopped")
        print(f"📊 Final session stats saved to {self.session_file}")
    
    async def analyze_and_coach(self, event_buffer):
        """Enhanced analysis with personalized coaching"""
        print(f"\n{'='*70}")
        print("🧠 PERSONALIZED AI ANALYSIS IN PROGRESS...")
        
        # Analyze telemetry pattern
        analysis = self.telemetry_analyzer.analyze(event_buffer)
        
        # Add historical context
        analysis['cumulative_session_hours'] = event_buffer[-1].get('cumulative_session_hours', 0)
        analysis['total_apps_today'] = event_buffer[-1].get('total_apps_today', 0)
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
        personal_productivity = self.coach.calculate_personalized_productivity_score(event_buffer, context)
        personal_focus = self.coach.calculate_personalized_focus_quality(event_buffer, context)
        
        # Update analysis with personalized scores
        analysis['productivity_score'] = personal_productivity
        analysis['focus_quality'] = personal_focus
        
        print(f"📊 PERSONALIZED Analysis Results:")
        print(f"   Activity Level: {analysis['activity_level']}")
        print(f"   Productivity (Personal): {analysis['productivity_score']:.1%}")
        print(f"   Focus Quality (Personal): {analysis['focus_quality']:.1%}")
        print(f"   Current Session: {analysis['session_hours']:.2f}h")
        print(f"   Today Total: {analysis['cumulative_session_hours']:.2f}h")
        print(f"   Apps Used Today: {analysis['total_apps_today']}")
        
        # Store scores for historical analysis
        self.session_data['productivity_scores'].append(analysis['productivity_score'])
        self.session_data['focus_scores'].append(analysis['focus_quality'])
        
        # Get personalized coaching
        print("🤖 Requesting PERSONALIZED AI coaching...")
        coaching_result = await self.coach.generate_personalized_coaching(context, analysis)
        
        if coaching_result:
            message = coaching_result.get('message', 'No specific advice')
            priority = coaching_result.get('priority', 2)
            reasoning = coaching_result.get('reasoning', 'Generic advice')
            
            print(f"\n💡 PERSONALIZED AI COACHING:")
            print(f"   {message}")
            print(f"   Priority: {priority}/3")
            print(f"   🧠 Reasoning: {reasoning}")
            
            # Show desktop notification
            self.show_coaching_notification(message, priority)
            
            # Log coaching with enhanced data
            self.log_coaching(coaching_result, analysis)
            
            # Update coaching count
            self.session_data['coaching_count'] += 1
        else:
            print("ℹ️  No personalized coaching needed at this time")
        
        # Update daily stats
        self.update_daily_stats(analysis)
        
        print("="*70 + "\n")
    
    def show_coaching_notification(self, message, priority):
        """Show macOS desktop notification with multiple fallback methods"""
        try:
            title_map = {1: "URGENT - AI Coach", 2: "AI Coach Alert", 3: "AI Coach Tip"}
            title = title_map.get(priority, "AI Coach")
            
            # Clean message for notification - be more careful with quotes
            clean_message = message.replace('"', "'").replace('\n', ' ').replace('\\', '').strip()[:120]
            
            import subprocess
            
            # Method 1: Try terminal-notifier first (more reliable)
            try:
                result = subprocess.run(['terminal-notifier', '-title', title, '-message', clean_message, '-sound', 'Ping'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    print(f"   📱 Desktop notification sent via terminal-notifier: {title}")
                    return
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Method 2: AppleScript with better escaping
            escaped_message = clean_message.replace('"', '\\"').replace("'", "\\'")
            escaped_title = title.replace('"', '\\"').replace("'", "\\'")
            
            script = f'''
            display notification "{escaped_message}" with title "{escaped_title}" sound name "Ping"
            delay 0.1
            '''
            
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                print(f"   📱 Desktop notification sent via AppleScript: {title}")
            else:
                print(f"   ⚠️ Notification failed: {result.stderr}")
                # Method 3: System beep as fallback
                subprocess.run(['osascript', '-e', 'beep'], capture_output=True)
                print(f"   🔔 Audio notification sent (visual notification may be blocked)")
            
        except Exception as e:
            print(f"   ⚠️ Notification error: {e}")
            # Final fallback - just beep
            try:
                subprocess.run(['osascript', '-e', 'beep'], capture_output=True)
                print(f"   🔔 Audio fallback notification sent")
            except:
                pass
    
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
        worksmart_analysis = self.telemetry_collector.worksmart_reader.analyze_productivity_patterns(hours=8)
        
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
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"📅 Session date: {self.today}", flush=True)
        print(flush=True)
        
        # Check WorkSmart status
        if self.check_worksmart_running():
            print("✅ WorkSmart is running - will monitor actual data", flush=True)
        else:
            print("⚠️ WorkSmart not detected - will run in standalone mode", flush=True)
        
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
        print(f"WorkSmart Running: {'✅' if self.check_worksmart_running() else '❌'}")
        print(f"AI Coach Status: {'🟢 Active' if self.running else '🔴 Inactive'}")
        
        # Get WorkSmart stats
        worksmart_stats = self.telemetry_collector.worksmart_reader.get_current_session_stats()
        print(f"WorkSmart Hours Today: {worksmart_stats.get('total_hours_today', '0:0')}")
        
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
            
            print(f"Avg Productivity: {stats.get('productivity_average', 0):.1%}")
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
            print("  python3 worksmart_enhanced_launcher.py        # Start enhanced system")
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