# WorkSmart AI Coach

## Overview

Production-ready AI productivity coaching system with real-time telemetry analysis. Features 3-pass Claude AI integration, intelligent session detection, and configurable desktop notifications. Self-contained DMG distribution with embedded Python runtime.

## Architecture

### Core Components

- **AI Coach Engine** (`core/ai_coach.py`): 4,400+ line consolidated coaching system with persona detection, pattern analysis, and intervention generation
- **Enhanced Launcher** (`launchers/enhanced.py`): Production monitoring loop with telemetry collection and coaching orchestration  
- **WorkSmart Bridge** (`bridge/worksmart_reader.py`): CrossOver telemetry data ingestion and analysis
- **Telemetry System** (`core/telemetry.py`): Event collection, storage, and analytics pipeline

### System Capabilities

- **3-Pass AI Analysis**: Historical context + current activity + synthesis for comprehensive coaching
- **Smart Session Detection**: Flexible WorkSmart integration with fallback mechanisms
- **Rate Limiting**: Configurable notification frequency (gentle/balanced/active)
- **Multi-Channel Notifications**: terminal-notifier, AppleScript dialogs, console fallbacks
- **Production Ready**: Clean logging, error handling, self-contained distribution

## Installation

### End User (DMG Distribution)

```bash
# Download and mount DMG
curl -LO https://example.com/WorkSmart-AI-Coach.dmg
open WorkSmart-AI-Coach.dmg

# Launch self-contained application
/Applications/WorkSmart\ AI\ Coach/Launch\ WorkSmart\ AI\ Coach.command
```

### Developer Setup

```bash
# Clone and install dependencies
git clone https://github.com/StanHus/worksmart-ai_coach.git
cd worksmart-ai_coach
pip install -e . --user

# Run development instance
python3 -m worksmart_ai_coach.launchers.enhanced

# Build production DMG
./build_dmg.sh
```

## Configuration

### Environment Variables

```bash
# ~/.worksmart-ai-coach/.env
CROSSOVER_FILES_PATH=~/crossoverFiles
ANTHROPIC_API_KEY=sk-ant-...
NOTIFICATION_FREQUENCY=gentle  # gentle|balanced|active
DEBUG=false
```

### Notification Frequency Settings

- **gentle**: 2/hour, 25min intervals, minimal interruption (default)
- **balanced**: 4/hour, 12min intervals, regular coaching
- **active**: 6/hour, 8min intervals, frequent guidance

## Technical Implementation

### 3-Pass AI Coaching Pipeline

```python
# Pass 1: Historical Analysis (7-day context)
historical_analysis = claude_analyze_history(user_data, current_context)

# Pass 2: Current Activity Analysis
current_analysis = claude_analyze_current_activity(telemetry_data)

# Pass 3: Synthesis & Decision
final_coaching = claude_synthesize_and_decide(historical_analysis, current_analysis)
```

### Telemetry Data Flow

1. **Collection**: Enhanced launcher polls CrossOver logs every 60s
2. **Processing**: Event buffer analysis every 3 events (5-6min intervals)
3. **Storage**: Persistent session data in ~/.worksmart-ai-coach/data/
4. **Analysis**: Productivity scoring, focus detection, pattern matching

### Notification Delivery

```python
# Multi-channel notification system
channels = {
    'terminal_notifier': subprocess_call_with_sound,
    'system_banner': applescript_dialog_for_critical,
    'console': fallback_logging
}
```

## Development

### Key Files

```
worksmart_ai_coach/
├── core/
│   ├── ai_coach.py          # Main coaching engine (4,400+ lines)
│   ├── telemetry.py         # Event collection and storage
│   └── detectors/           # Pattern detection modules
├── launchers/
│   └── enhanced.py          # Production monitoring loop
├── bridge/
│   └── worksmart_reader.py  # CrossOver integration
└── cli/
    └── main.py              # Command-line interface

build_dmg.sh                 # Self-contained DMG builder
```

### Build Process

```bash
# Creates embedded Python runtime DMG
./build_dmg.sh

# Output: WorkSmart-AI-Coach.dmg (19MB self-contained app)
# Includes: Python 3.x runtime, all dependencies, launcher scripts
```

### Debug Mode

```bash
# Enable comprehensive logging
export DEBUG=true
python3 -m worksmart_ai_coach.launchers.enhanced

# Log output includes:
# - Event buffer status
# - Productivity calculations  
# - Detector flag analysis
# - Notification delivery results
```

## API Integration

### Anthropic Claude

```python
# AI-generated coaching with context awareness
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=150,
    messages=[{"role": "user", "content": coaching_prompt}]
)
```

### WorkSmart Telemetry

```python
# Real-time session statistics
stats = {
    'total_hours_today': '4:30',
    'productivity_score': 0.67,
    'focus_score': 0.54,
    'session_active': True
}
```

## System Requirements

- **macOS 10.12+** (High Sierra or later)
- **Python 3.8+** (for development, embedded in DMG)
- **terminal-notifier** (bundled with DMG)
- **CrossOver WorkSmart** (for telemetry data source)

## Troubleshooting

### Common Issues

```bash
# Notification delivery failures
[DEBUG] 📱 Terminal-notifier result: returncode=1
# Solution: Check terminal-notifier installation

# Rate limiting blocking notifications
[DEBUG] 🚫 Rate limited - skipping coaching
# Solution: Adjust NOTIFICATION_FREQUENCY in .env

# Missing telemetry data
[DEBUG] 📚 History digests built: 7d=0, 30d=0
# Solution: Verify CROSSOVER_FILES_PATH configuration
```

### Performance Monitoring

```bash
# Session statistics
coaching_count: 12
accumulated_hours_today: 5.5
productivity_scores: [0.67, 0.54, 0.72, ...]
focus_scores: [0.82, 0.66, 0.54, ...]
```

## License

MIT License - Production-ready system for CrossOver productivity monitoring.