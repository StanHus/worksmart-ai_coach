#!/usr/bin/env python3
"""
WorkSmart AI Coach - Bridge CLI
"""

import click
import sys
import os
from pathlib import Path

def main():
    """Bridge CLI entry point"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            click.echo("🌉 WorkSmart AI Coach Bridge")
            click.echo("")
            click.echo("Usage: worksmart-bridge [command]")
            click.echo("")
            click.echo("Commands:")
            click.echo("  (no args)    Start the bridge")
            click.echo("  status       Show bridge status")
            click.echo("  test         Send test data")
            click.echo("  --help       Show this help")
            return
        elif sys.argv[1] == "status":
            show_bridge_status()
        elif sys.argv[1] == "test":
            test_bridge()
        else:
            start_bridge()
    else:
        start_bridge()

def start_bridge():
    """Start the bridge"""
    click.echo("🌉 Starting WorkSmart AI Coach Bridge...")
    
    try:
        from ..bridge.java_bridge import AICoachBridge
        import asyncio
        
        bridge = AICoachBridge()
        asyncio.run(bridge.start_bridge())
    except KeyboardInterrupt:
        click.echo("\n👋 Bridge stopped")
    except Exception as e:
        click.echo(f"❌ Bridge error: {e}")
        sys.exit(1)

def show_bridge_status():
    """Show bridge status"""
    import json
    
    status_file = Path("/tmp/worksmart_aicoach/status.json")
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            click.echo("🔗 Bridge Status")
            click.echo(f"Status: {status.get('status', 'unknown')}")
            click.echo(f"Last Update: {status.get('timestamp', 'unknown')}")
            click.echo(f"PID: {status.get('pid', 'unknown')}")
            if status.get('details'):
                click.echo(f"Details: {status.get('details')}")
        except Exception as e:
            click.echo(f"❌ Error reading status: {e}")
    else:
        click.echo("❌ Bridge not running")

def test_bridge():
    """Test the bridge with sample data"""
    import json
    from pathlib import Path
    
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
    
    bridge_dir = Path("/tmp/worksmart_aicoach")
    bridge_dir.mkdir(exist_ok=True)
    
    telemetry_file = bridge_dir / "telemetry.json"
    with open(telemetry_file, 'w') as f:
        json.dump(test_telemetry, f, indent=2)
    
    click.echo("📋 Test telemetry sent to bridge")

if __name__ == '__main__':
    main()