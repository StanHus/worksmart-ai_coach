# WorkSmart AI Coach

**Ultimate AI-powered productivity coaching system with advanced intelligence layers.**

## 🚀 Features

### Core Intelligence Systems
- **🧠 Personalized AI Coaching** - Uses your actual productivity patterns instead of generic advice
- **📊 Real-Time Context Tracking** - Maintains 6-hour activity memory for pattern analysis
- **🔮 Predictive Analysis** - Recommends optimal next actions with confidence scores
- **⚡ Momentum Detection** - Identifies post-meeting energy and AI tool recovery patterns
- **🔔 Micro-Interventions** - Gentle 2-3 minute nudges vs disruptive full interventions
- **🎓 Adaptive Learning** - Tracks coaching effectiveness and improves over time

### Key Capabilities
- **5-minute unproductive intervention rule** (vs generic 10+ minute delays)
- **AI tool suggestions** (Grok, ChatGPT) based on learned recovery patterns
- **Tab count intelligence** (optimal ≤9 tabs, penalty for 10+ tabs)
- **Flow state protection** to avoid interrupting high-productivity periods
- **Context-aware productivity scoring** with personal multipliers
- **Effectiveness tracking** with 5-minute and 15-minute outcome measurements

## 🛠 Installation

```bash
# Install the package
pip install -e .

# Verify installation
wsai --version
worksmart-enhanced help
```

## 🎯 Usage

### Enhanced AI Coach (Recommended)
```bash
# Start the advanced AI coaching system
worksmart-enhanced

# Check system status
worksmart-enhanced status

# View daily productivity stats
worksmart-enhanced stats
```

### Standard Interface
```bash
# Alternative interface
wsai start

# Show system information
wsai status

# View statistics
wsai stats
```

### Integration with WorkSmart
```bash
# Start bridge for Java WorkSmart integration
worksmart-bridge

# Manual feedback collection
worksmart-feedback
```

## 🧪 Example Intelligence

### Micro-Interventions (2-3 minutes)
```
🔔 "Noticed 14 tabs open. Consider focusing on 1-2 key tabs for better concentration."
```

### Momentum Detection
```
💡 "Perfect timing! Your 23-minute meeting showed excellent focus. 
    This is the ideal moment for AI-assisted creative work. 
    The momentum transfer window is now open."
```

### Adaptive Learning
```
🧠 System learns: "Tab consolidation suggestions work best around 14:00 
    with 85% effectiveness for this user"
```

### Predictive Analysis
```
🔮 "Start AI-assisted creative work. Confidence: 85%. 
    Post-meeting momentum transfer pattern detected."
```

## 📊 Personal Pattern Examples

The system automatically discovers and uses your personal productivity correlations:

- **High Productivity**: AI tools (Grok, ChatGPT) + focused tabs (≤9) + file activity
- **Meeting Excellence**: Safari + single focus + 20+ minute engagement  
- **Recovery Patterns**: Social media → Google → AI tools (4-5 minute sequence)
- **Tab Intelligence**: 10+ tabs = distraction, 6-9 tabs = optimal focus
- **Timing Patterns**: Post-meeting momentum windows, optimal intervention hours

## 🎛 Configuration

### Environment Variables
Create a `.env` file:
```env
# Claude AI API (required)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional settings
WORKSMART_COACH_INTERVAL=100  # seconds between checks
WORKSMART_COACH_DEBUG=false   # enable debug logging
```

### Personal Patterns
The system automatically learns your patterns from usage, but you can also provide historical data in `consolidated_learning_data.json` format.

## 📁 Data Files

The system creates several data files for learning and tracking:
- `context_history_YYYY-MM-DD.json` - Daily context snapshots
- `adaptive_learning_YYYY-MM-DD.json` - Intervention effectiveness data
- `worksmart_session_YYYY-MM-DD.json` - Session statistics
- `coaching_log_YYYY-MM-DD.json` - Coaching history
- `daily_stats_YYYY-MM-DD.json` - Daily productivity summaries

## 🔧 Architecture

```
worksmart_ai_coach/
├── core/                    # Core AI systems
│   ├── coach.py            # Base AI coaching engine
│   ├── personalized_coach.py  # Advanced personalized algorithms
│   ├── context_tracker.py     # Real-time context memory
│   ├── micro_interventions.py # Gentle nudge system
│   ├── adaptive_learning.py   # Effectiveness tracking
│   └── telemetry.py           # Activity monitoring
├── cli/                     # Command-line interfaces
├── launchers/              # System launchers
│   └── enhanced.py         # Advanced production launcher
└── bridge/                 # WorkSmart integration
```

## 🎓 Learning Evolution

**Stage 1**: Generic coaching based on activity levels
**Stage 2**: Pattern recognition from user feedback
**Stage 3**: Personalized algorithms using discovered correlations  
**Stage 4**: Predictive analysis and momentum detection
**Stage 5**: Micro-interventions and adaptive learning ← **Current**

## 📈 Performance Metrics

- **Response Time**: 2-3 minute micro-nudges vs previous 10+ minute delays
- **Accuracy**: Personal correlation-based vs generic activity scoring
- **Effectiveness**: Measured and tracked vs unknown outcomes
- **Intelligence**: Self-learning adaptive system vs static patterns
- **Context**: 6-hour memory vs single-event analysis

## 🤝 Contributing

The system is designed to learn and adapt. The most valuable contributions are:
1. Using the system and providing feedback
2. Sharing productivity pattern insights
3. Reporting effectiveness measurements
4. Contributing to pattern recognition algorithms

## 📜 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Documentation**: Enhanced coaching algorithms and personal pattern system
- **Archive**: `archive/` contains legacy files and development history
- **Learning Data**: Historical productivity patterns and effectiveness measurements

---

**From "Data Rich, Intelligence Poor" to "Data Rich, Intelligence Superior"** 🚀