# WorkSmart AI Coach 🤖

**Self-Sustained AI-Powered Productivity Coaching**

NO Python installation required! Complete productivity coaching system that works out-of-the-box on any macOS system.

## 🎯 For Users (Zero Setup Required)

### 📦 Download & Run
1. **Download:** `WorkSmart-AI-Coach-Self-Sustained-v1.0.0.dmg`
2. **Mount:** Double-click the DMG file
3. **Install:** Drag "WorkSmart AI Coach" to Applications
4. **Launch:** Double-click "🚀 Launch WorkSmart AI Coach.command"
5. **Done!** Follow the setup prompts - that's it!

### ✨ What You Get
- **Real-time productivity monitoring** 
- **AI-powered coaching suggestions**
- **Activity pattern analysis**
- **Desktop notifications**
- **Session tracking and statistics**
- **Complete self-contained app** (NO Python needed!)

### 🔧 First Launch Setup
The app will prompt you for:
- **CrossOver files path** (default: ~/crossoverFiles)
- **Anthropic API key** (optional - get free key at https://console.anthropic.com)

Settings saved to `~/.worksmart-ai-coach/.env` - edit anytime to update.

## 🚀 What It Does

1. **Monitors** your computer activity (apps, windows, productivity patterns)
2. **Analyzes** focus levels and work habits  
3. **Provides** real-time coaching suggestions via desktop notifications
4. **Tracks** daily statistics and progress
5. **Learns** from your behavior to improve recommendations

## 📊 Example Output

```
🚀 WorkSmart AI Coach (Self-Sustained)
======================================
✨ NO Python installation required!

🔧 WorkSmart AI Coach Setup
============================

📁 CrossOver Files Path
Enter path (default: ~/crossoverFiles): 

🔑 Anthropic API Key
Get free API key: https://console.anthropic.com
1. Enter API key now
2. Skip (basic mode)

Choose (1/2): 1
🔐 API Key (hidden): 
✅ API key saved!

🎯 Starting AI Coach...

🚀 WORKSMART ENHANCED AI COACH
============================================================
🕐 Started at: 2025-08-18 21:48:02
📅 Session date: 2025-08-18

[21:48:04] 📈 Activity captured:
  App: Terminal
  Window: WorkSmart AI Coach
  Coaching Session: 0.00h
  WorkSmart Today: 11:0
  Activity: 24🔤 30🖱️
```

## 🔒 Privacy & Security

- **All data stays on your Mac** - nothing sent to external servers
- **API key only used for AI suggestions** - stored locally
- **Session files stored locally** in ~/.worksmart-ai-coach/data/
- **No system modifications** - completely self-contained

## 📋 System Requirements

- **macOS 10.12+** (High Sierra or later)
- **19MB disk space** for the app
- **No Python installation required!**
- **No external dependencies!**

## 🛠️ For Developers

### Building from Source
```bash
# Clone repository
git clone https://github.com/StanHus/worksmart-ai_coach.git
cd worksmart-ai_coach

# Build self-sustained DMG (requires Python 3.8+ to build)
./build_dmg.sh
```

This creates `WorkSmart-AI-Coach-Self-Sustained-v1.0.0.dmg` with embedded Python runtime.

### Development Setup
```bash
# For development (requires Python)
python3 -m worksmart_ai_coach.launchers.enhanced

# Install in development mode
pip install -e . --user
```

### Project Structure
```
worksmart_ai_coach/
├── cli/           # Command-line interface
├── core/          # AI coaching logic  
├── bridge/        # WorkSmart integration
└── launchers/     # Application launchers

build_dmg.sh       # Builds self-sustained DMG
```

## 🔍 Troubleshooting

**App won't launch**: Check that you're running macOS 10.12+  
**No AI features**: Add your Anthropic API key during setup  
**Permission issues**: Grant accessibility permissions in System Preferences  
**WorkSmart integration**: Set correct CrossOver files path during setup  

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🆘 Support

- [Issues](https://github.com/StanHus/worksmart-ai_coach/issues)
- [Discussions](https://github.com/StanHus/worksmart-ai_coach/discussions)

---

## 🎉 Perfect For

- **Users without Python installed**
- **Corporate environments with restrictions**
- **Easy deployment and distribution** 
- **Professional productivity coaching**
- **Anyone wanting zero-setup productivity tools**

**Ready to supercharge your productivity? Download the DMG and run!** 🚀