# WorkSmart AI Coach 🤖

AI-powered productivity coaching system that integrates with WorkSmart tracker to provide real-time insights and personalized suggestions.

## ⚡ Quick Start

```bash
# Clone and run
git clone https://github.com/StanHus/worksmart-ai_coach.git
cd worksmart-ai_coach
python3 -m worksmart_ai_coach.launchers.enhanced
```

## 📦 Installation Options

### Option 1: Direct Run (Fastest)
```bash
python3 -m worksmart_ai_coach.launchers.enhanced
```

### Option 2: Install Package
```bash
pip install -e . --user
worksmart-enhanced
```

### Option 3: One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/StanHus/worksmart-ai_coach/main/install.sh | bash
```

## ⚙️ Configuration

Create `~/.worksmart-ai-coach/.env`:
```bash
# Required: Path to CrossOver files
CROSSOVER_FILES_PATH=~/crossoverFiles

# Optional: Anthropic API key for AI features  
ANTHROPIC_API_KEY=your_key_here

# Optional: Debug mode
DEBUG=true
```

## 🚀 Usage

```bash
# Start the coach
python3 -m worksmart_ai_coach.launchers.enhanced

# After installing package
worksmart-enhanced              # Start enhanced mode
wsai status                     # Check status  
wsai start --mode enhanced     # CLI version
```

## 🏗️ Project Structure

```
worksmart_ai_coach/
├── cli/           # Command-line interface
├── core/          # AI coaching logic
├── bridge/        # WorkSmart integration
└── launchers/     # Application launchers
```

## 🧠 Features

- **Real-time Activity Monitoring** - Tracks apps, windows, productivity patterns
- **AI-Powered Coaching** - Personalized suggestions via Anthropic Claude
- **WorkSmart Integration** - Reads telemetry data from WorkSmart tracker
- **Adaptive Learning** - Improves suggestions based on your behavior
- **Session Persistence** - Maintains data across restarts
- **Cross-Platform** - Works on macOS, Windows, Linux

## 📊 What It Does

1. **Monitors** your computer activity (apps, windows, keystrokes)
2. **Analyzes** productivity patterns and focus levels
3. **Provides** real-time coaching suggestions
4. **Tracks** daily statistics and progress
5. **Learns** from your habits to improve recommendations

## 🔧 Requirements

- **Python**: 3.8+
- **OS**: macOS 10.12+, Windows 10+, Linux 18.04+
- **WorkSmart**: Optional but recommended for full features
- **API Key**: Optional for AI features (uses rule-based fallback)

## 📈 Output Example

```
🚀 WORKSMART ENHANCED AI COACH
============================================================
🕐 Started at: 2025-08-18 20:42:16
📅 Session date: 2025-08-18

✅ Using WorkSmart telemetry as primary data source
📅 Loaded existing session data
🤖 Enhanced AI Coach monitoring started

[20:42:17] 📈 Activity captured:
  App: Cursor
  Window: .env — crossover...
  Coaching Session: 0.76h
  WorkSmart Today: 9:50
  Activity: 110🔤 7🖱️
```

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/StanHus/worksmart-ai_coach.git
cd worksmart-ai_coach

# Install in development mode
pip install -e . --user

# Run tests (basic validation)
python3 -c "import worksmart_ai_coach; print('✅ Package OK')"

# Check configuration
python3 -m worksmart_ai_coach.launchers.enhanced status
```

## 🔍 Troubleshooting

**Module Not Found**: `pip install -e . --user --force-reinstall`  
**Command Not Found**: Add `~/.local/bin` to your PATH  
**WorkSmart Issues**: Check `CROSSOVER_FILES_PATH` in `.env`  
**No AI Features**: Add `ANTHROPIC_API_KEY` to `.env`  

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🆘 Support

- [Issues](https://github.com/StanHus/worksmart-ai_coach/issues)
- [Discussions](https://github.com/StanHus/worksmart-ai_coach/discussions)

---

**Ready to boost your productivity? Just run:** `python3 -m worksmart_ai_coach.launchers.enhanced` 🚀