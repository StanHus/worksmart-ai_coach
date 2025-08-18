#!/usr/bin/env python3
"""
WorkSmart AI Coach - Simple Launcher
"""

def launch():
    """Simple launcher entry point"""
    import click
    import sys
    from .main import cli
    
    # Check if help is requested
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h', 'help']:
        click.echo("🚀 WorkSmart AI Coach - Quick Launcher")
        click.echo("")
        click.echo("Usage: worksmart-coach [--help]")
        click.echo("")
        click.echo("This command automatically starts the AI coach in enhanced mode.")
        click.echo("For full options, use: worksmart-ai-coach --help")
        click.echo("")
        click.echo("Quick commands:")
        click.echo("  worksmart-coach          Start enhanced AI coach")
        click.echo("  wsai status              Check system status")
        click.echo("  wsai stats               View productivity stats")
        return
    
    click.echo("🚀 WorkSmart AI Coach Launcher")
    click.echo("Starting enhanced mode by default...")
    
    # Auto-launch in enhanced mode
    sys.argv = ['worksmart-coach', 'start', '--mode', 'enhanced']
    cli()

if __name__ == '__main__':
    launch()