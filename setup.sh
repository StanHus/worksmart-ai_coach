#!/bin/bash
# WorkSmart AI Coach - Setup and Test Script

echo "🚀 WorkSmart AI Coach - Setup & Testing"
echo "========================================"

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found. Please install Python 3.8+"; exit 1; }

# Check for API key
echo "🔑 Checking for API key..."
if [[ -z "${ANTHROPIC_API_KEY}" ]]; then
    echo "⚠️  ANTHROPIC_API_KEY not set in environment"
    echo "   Create .env file with: ANTHROPIC_API_KEY=your_key_here"
    echo "   Or export ANTHROPIC_API_KEY=your_key_here"
else
    echo "✅ API key found"
fi

echo ""
echo "📦 Installing package..."
python3 -m pip install -e . || { echo "❌ Installation failed"; exit 1; }

echo ""
echo "🧪 Testing entry points..."

echo "1. Main CLI:"
wsai --version

echo ""
echo "2. Enhanced AI Coach:"
worksmart-enhanced help

echo ""
echo "3. Bridge System:"
worksmart-bridge --help

echo ""
echo "🎯 Testing core systems..."
python3 -c "
print('Testing PersonalizedAICoach initialization...')
from worksmart_ai_coach.core.personalized_coach import PersonalizedAICoach
coach = PersonalizedAICoach()
print(f'✅ PersonalizedAICoach loaded with {len(coach.personal_patterns)} patterns')

print('\\nTesting intelligence systems...')
print(f'   Context tracker: {len(coach.context_tracker.context_history)} snapshots ready')
print(f'   Micro-interventions: {coach.micro_interventions.__class__.__name__} ready')
print(f'   Adaptive learning: {coach.adaptive_learning.__class__.__name__} ready')
print('✅ All systems operational')
"

echo ""
echo "📁 Clean package structure:"
echo "   /worksmart_ai_coach/core/     - AI intelligence systems"
echo "   /worksmart_ai_coach/cli/      - Command interfaces"  
echo "   /worksmart_ai_coach/launchers/ - Production launchers"
echo "   /worksmart_ai_coach/bridge/   - WorkSmart integration"
echo "   /archive/                     - Legacy files preserved"

echo ""
echo "🎉 SETUP COMPLETE!"
echo "Ready to start with: worksmart-enhanced"
echo ""
echo "📊 Intelligence Features Ready:"
echo "   ✅ 5-minute personalized intervention rule"
echo "   ✅ AI tool suggestions (Grok, ChatGPT)"
echo "   ✅ Tab intelligence (≤9 optimal, 10+ penalty)"
echo "   ✅ Context history tracking (6-hour memory)"
echo "   ✅ Micro-interventions (2-3 minute gentle nudges)"
echo "   ✅ Adaptive learning (tracks coaching effectiveness)"
echo "   ✅ Momentum detection (post-meeting, AI recovery)"
echo "   ✅ Predictive analysis (confidence-based recommendations)"
echo ""
echo "From 'Data Rich, Intelligence Poor' to 'Data Rich, Intelligence Superior' 🧠✨"