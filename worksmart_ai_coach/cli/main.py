#!/usr/bin/env python3
"""
WorkSmart AI Coach - Main CLI Entry Point
"""

import click
import sys
import os
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

@click.group()
@click.version_option(version="1.0.0", prog_name="WorkSmart AI Coach")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """WorkSmart AI Coach - AI-powered productivity coaching"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@cli.command()
@click.option('--mode', '-m', type=click.Choice(['standalone', 'production', 'enhanced', 'learning']), 
              default='enhanced', help='Launch mode')
@click.option('--config', '-c', type=click.Path(exists=True), help='Config file path')
@click.pass_context
def start(ctx, mode, config):
    _ = ctx, config  # Mark unused parameters
    """Start the AI coach system"""
    click.echo(f"🚀 Starting WorkSmart AI Coach in {mode} mode...")
    
    try:
        if mode == 'learning':
            # Learning mode not yet implemented
            click.echo("Learning mode not yet available")
            return
            import asyncio
            launcher = LearningModeLauncher()
            asyncio.run(launcher.start_learning_mode())
        elif mode == 'enhanced':
            from ..launchers.enhanced import EnhancedProductionLauncher
            import asyncio
            launcher = EnhancedProductionLauncher()
            asyncio.run(launcher.start_production_system())
        elif mode == 'production':
            # Production mode not yet implemented
            click.echo("Production mode not yet available")
            return
            import asyncio
            launcher = ProductionLauncher()
            asyncio.run(launcher.start_production_system())
        else:
            click.echo("Standalone mode - basic AI coach")
            from ..core.coach import AICoach
            from ..core.coach import AICoach
            coach = AICoach()
            _ = coach  # Mark unused variable
            click.echo("✅ AI Coach initialized")
            
    except KeyboardInterrupt:
        click.echo("\n👋 Shutting down gracefully")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)

@cli.command()
@click.pass_context
def status(ctx):
    _ = ctx  # Mark unused parameter
    """Show system status"""
    click.echo("📊 WorkSmart AI Coach Status")
    click.echo("=" * 40)
    
    # Check if processes are running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'worksmart'], capture_output=True, text=True)
        worksmart_running = len(result.stdout.strip()) > 0
        click.echo(f"WorkSmart: {'🟢 Running' if worksmart_running else '🔴 Not running'}")
    except:
        click.echo("WorkSmart: ❓ Status unknown")
    
    # Check log files
    log_files = [
        'worksmart_session.json',
        'coaching_log.json',
        'daily_stats.json',
        'context_history.json',
        'adaptive_learning.json'
    ]
    
    for filename in log_files:
        if os.path.exists(filename):
            mtime = datetime.fromtimestamp(os.path.getmtime(filename))
            click.echo(f"Log file: {filename} (last updated: {mtime.strftime('%H:%M:%S')})")

@cli.command()
@click.option('--days', '-d', default=7, help='Number of days to show')
def stats(days):
    """Show productivity statistics"""
    click.echo(f"📈 Productivity Stats (last {days} days)")
    click.echo("=" * 40)
    
    # glob imported at top
    import json
    # datetime imported at top
    
    # Find recent stat files
    stat_files = ['daily_stats.json'] if os.path.exists('daily_stats.json') else []
    
    if not stat_files:
        click.echo("No statistics available")
        return
    
    for file in sorted(stat_files)[-days:]:
        try:
            with open(file, 'r') as f:
                stats = json.load(f)
            
            date = stats.get('date', 'Unknown')
            hours = stats.get('total_session_hours', 0)
            productivity = stats.get('productivity_average', 0)
            focus = stats.get('focus_average', 0)
            
            click.echo(f"{date}: {hours:.1f}h, Productivity: {productivity:.1%}, Focus: {focus:.1%}")
        except:
            continue

@cli.command()
def test():
    """Run system tests"""
    click.echo("🧪 Running WorkSmart AI Coach tests...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, '-m', 'pytest', 'test_ai_coach.py', '-v'], 
                              capture_output=True, text=True)
        
        click.echo(result.stdout)
        if result.stderr:
            click.echo("Errors:", err=True)
            click.echo(result.stderr, err=True)
            
        sys.exit(result.returncode)
    except FileNotFoundError:
        click.echo("❌ pytest not found. Install with: pip install pytest")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Test error: {e}")
        sys.exit(1)

@cli.command()
@click.option('--bridge-dir', default='/tmp/worksmart_aicoach', help='Bridge directory')
def bridge(bridge_dir):
    """Start bridge for Java WorkSmart integration"""
    click.echo("🌉 Starting AI Coach Bridge...")
    
    try:
        from ..bridge.java_bridge import AICoachBridge
        import asyncio
        
        bridge_instance = AICoachBridge(bridge_dir)
        asyncio.run(bridge_instance.start_bridge())
    except KeyboardInterrupt:
        click.echo("\n👋 Bridge stopped")
    except Exception as e:
        click.echo(f"❌ Bridge error: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    try:
        cli()
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()