# WorkSmart AI Coach Installation Guide

## Quick Installation

### Option 1: Automated Install (Recommended)
```bash
cd /path/to/package
chmod +x install.sh
./install.sh
```

### Option 2: Manual Install
```bash
cd /path/to/package
pip3 install -e . --user
cp .env.template .env
# Edit .env with your settings
```

### Option 3: From Wheel
```bash
pip3 install dist/worksmart_ai_coach-1.0.0-py3-none-any.whl --user
```

## Configuration

1. **Create .env file**:
   ```bash
   cp .env.template .env
   ```

2. **Edit .env** with your settings:
   ```bash
   # Required: Set your crossover files path
   CROSSOVER_FILES_PATH=/Users/yourusername/crossoverFiles
   
   # Optional: Add Anthropic API key for AI features
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

After installation, you can use these commands:

```bash
# Main CLI
wsai --help
wsai start --mode enhanced

# Direct launcher
worksmart-enhanced

# Full name CLI
worksmart-ai-coach --help
```

## PATH Issues

If commands aren't found, add to your PATH:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

For zsh users:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## Troubleshooting

### Module Not Found Error
If you get `ModuleNotFoundError: No module named 'worksmart_ai_coach'`:

1. Ensure you're in the correct directory with `pyproject.toml`
2. Reinstall: `pip3 install -e . --user --force-reinstall`
3. Check Python path: `python3 -c "import sys; print(sys.path)"`

### WorkSmart Not Detected
The coach can run in standalone mode without WorkSmart installed. To connect to WorkSmart:

1. Ensure WorkSmart is running
2. Set correct `CROSSOVER_FILES_PATH` in `.env`
3. Verify WorkSmart data files exist in that path

### API Key Issues
AI features are optional. Without an API key, the system uses rule-based coaching which still provides valuable insights.