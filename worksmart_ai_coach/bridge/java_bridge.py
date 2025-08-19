#!/usr/bin/env python3
"""
AI Coach Bridge - Safe Integration with Java WorkSmart
======================================================

This bridge allows the Java WorkSmart tracker to safely communicate 
with the Python AI coach system through file-based messaging.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from ..core.ai_coach import AICoach


class AICoachBridge:
    """Bridge between Java WorkSmart and Python AI Coach"""

    def __init__(self, bridge_dir="/tmp/worksmart_aicoach"):
        self.bridge_dir = Path(bridge_dir)
        self.bridge_dir.mkdir(exist_ok=True)

        # Communication files
        self.telemetry_file = self.bridge_dir / "telemetry.json"
        self.coaching_file = self.bridge_dir / "coaching.json"
        self.status_file = self.bridge_dir / "status.json"

        self.coach = AICoach()
        self.running = False

        # Write initial status
        self._write_status("initialized")

    def _write_status(self, status, details=""):
        """Write current status to file for Java to read"""
        status_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "pid": os.getpid()
        }

        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            print(f"Error writing status: {e}")

    def _read_telemetry(self):
        """Read telemetry data from Java"""
        if not self.telemetry_file.exists():
            return None

        try:
            with open(self.telemetry_file, 'r') as f:
                data = json.load(f)

            # Mark as processed by removing file
            self.telemetry_file.unlink()
            return data

        except Exception as e:
            print(f"Error reading telemetry: {e}")
            return None

    def _write_coaching(self, coaching_result):
        """Write coaching result for Java to read"""
        if not coaching_result:
            return

        coaching_data = {
            "message": coaching_result.get("message", ""),
            "priority": coaching_result.get("priority", 2),
            "notification_type": coaching_result.get("notification_type", "general"),
            "timestamp": datetime.now().isoformat(),
            "confidence": coaching_result.get("confidence", 0.5)
        }

        try:
            with open(self.coaching_file, 'w') as f:
                json.dump(coaching_data, f, indent=2)
        except Exception as e:
            print(f"Error writing coaching: {e}")

    async def start_bridge(self):
        """Start the bridge monitoring loop"""
        print("🌉 AI Coach Bridge starting...")
        print(f"📁 Bridge directory: {self.bridge_dir}")
        print("🔗 Waiting for telemetry from Java WorkSmart...")

        self.running = True
        self._write_status("running", "Bridge active, waiting for telemetry")

        while self.running:
            try:
                # Check for new telemetry data
                telemetry = self._read_telemetry()

                if telemetry:
                    print(f"\n📊 Received telemetry from WorkSmart:")
                    print(
                        f"   Session: {telemetry.get('session_duration_hours', 0):.1f} hours")
                    print(
                        f"   Productivity: {telemetry.get('productivity_score', 0):.1%}")

                    self._write_status(
                        "analyzing", "Processing telemetry with AI")

                    # Process with AI coach
                    user_id = telemetry.get("user_id", "worksmart_user")
                    coaching_result = await self.coach.analyze_telemetry(telemetry, user_id)

                    if coaching_result:
                        print(
                            f"🤖 AI Coaching generated: {coaching_result.get('message', '')[:50]}...")
                        self._write_coaching(coaching_result)
                        self._write_status(
                            "coaching_delivered", f"Delivered coaching to user")
                    else:
                        self._write_status(
                            "no_coaching", "No coaching needed at this time")

                # Check every 5 seconds
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                print("\n🛑 Bridge shutdown requested")
                break
            except Exception as e:
                print(f"❌ Bridge error: {e}")
                self._write_status("error", str(e))
                await asyncio.sleep(10)

        self.running = False
        self._write_status("stopped", "Bridge shutdown completed")
        print("👋 AI Coach Bridge stopped")

    def stop_bridge(self):
        """Stop the bridge"""
        self.running = False
        self._write_status("stopping", "Shutdown requested")


def main():
    """Main bridge runner"""
    bridge = AICoachBridge()

    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "status":
                # Show current status
                status_file = Path("/tmp/worksmart_aicoach/status.json")
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        status = json.load(f)
                    print(
                        f"🔗 Bridge Status: {status.get('status', 'unknown')}")
                    print(
                        f"📅 Last Update: {status.get('timestamp', 'unknown')}")
                    print(f"🔢 PID: {status.get('pid', 'unknown')}")
                    if status.get('details'):
                        print(f"ℹ️  Details: {status.get('details')}")
                else:
                    print("❌ Bridge not running")
                return

            elif sys.argv[1] == "test":
                # Send test telemetry
                test_telemetry = {
                    "user_id": "test_user",
                    "session_duration_hours": 2.5,
                    "productivity_score": 0.7,
                    "focus_quality": 0.6,
                    "stress_level": 0.4,
                    "energy_level": 0.8,
                    "keystrokes_per_min": 45,
                    "app_switches_per_hour": 12
                }

                telemetry_file = Path("/tmp/worksmart_aicoach/telemetry.json")
                with open(telemetry_file, 'w') as f:
                    json.dump(test_telemetry, f, indent=2)

                print("📋 Test telemetry sent to bridge")
                return

        # Start bridge
        asyncio.run(bridge.start_bridge())

    except Exception as e:
        print(f"❌ Bridge startup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
