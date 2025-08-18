# WorkSmart AI Coach Package Structure

## CURRENT PRODUCTION CODE (RELEVANT)

### Core Package Structure
```
worksmart_ai_coach/
├── __init__.py                    # Main package exports
├── core/                          # Core functionality
│   ├── __init__.py
│   ├── coach.py                   # Main AI coach implementation
│   ├── telemetry.py              # Data collection and analysis
│   ├── adaptive_learning.py      # Learning algorithms
│   ├── context_tracker.py        # Context tracking
│   ├── micro_interventions.py    # Intervention system
│   └── personalized_coach.py     # Personalized coaching
├── cli/                          # Command-line interface
│   ├── __init__.py
│   ├── main.py                   # Main CLI entry points (wsai command)
│   ├── launcher.py               # Alternative launchers
│   └── bridge.py                 # Bridge functionality
├── bridge/                       # Java integration
│   ├── __init__.py
│   └── java_bridge.py           # Java WorkSmart bridge
├── launchers/                    # Production launchers
│   ├── __init__.py
│   └── enhanced.py              # Enhanced production launcher
└── data/                        # Data directory (runtime)
```

### Configuration Files
- `pyproject.toml` - Main project configuration and dependencies
- `requirements.txt` - Python dependencies
- `setup.sh` - Installation script
- `LICENSE` - MIT license
- `README.md` - Package documentation

### Available Commands
- `wsai` - Main command-line interface
- `worksmart-ai-coach` - Alternative entry point
- `worksmart-coach` - Launcher
- `worksmart-enhanced` - Enhanced launcher
- `worksmart-bridge` - Bridge mode

## ARCHIVED CONTENT (NOT RELEVANT FOR CURRENT USE)

### Development Archive (`archive/development-docs/`)
- Historical development documentation
- Algorithm adjustment notes
- Design documents
- Stage development notes

### Legacy Files (`archive/legacy-files/`)
- Old package versions (`backup/v1.0-stable/`)
- Deprecated scripts (`scripts/`)
- Old build artifacts (`dist/`, `*.egg-info/`)
- Standalone modules (`feedback.py`, `learning.py`, `production.py`)
- Old setup files (`setup.py`)

### Old Data (`archive/old-data/`)
- Historical session logs
- Old learning data files
- Context history files
- Coaching logs

## TESTING AND DEVELOPMENT

### Virtual Environment Setup
```bash
cd package/
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Dependencies
- **Core**: anthropic, pynput, psutil, python-dotenv, aiofiles, click, numpy
- **Optional**: plyer (notifications), pythonnet (.NET integration)
- **Dev**: pytest, black, flake8, mypy

### Basic Functionality Test
```bash
wsai --help      # Show help
wsai status      # Check system status
wsai stats       # Show productivity stats
```

## KEY POINTS

1. **Only files in the main `worksmart_ai_coach/` directory are currently relevant**
2. **Everything in `archive/` is legacy/historical content**
3. **The package is fully functional with the core dependencies**
4. **CLI commands work and provide the expected interface**
5. **No crossover-xo.clients repositories are affected by this organization**