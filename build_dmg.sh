#!/bin/bash
# Create super simple DMG with just the working files

echo "🍎 Creating Simple DMG with Working Files"
echo "========================================"

DMG_NAME="WorkSmart-AI-Coach-v1.0.0"
FOLDER_NAME="WorkSmart AI Coach"

# Clean up
rm -rf dmg_temp "$DMG_NAME.dmg"
mkdir -p "dmg_temp/$FOLDER_NAME"

# Copy all the working files
echo "📁 Copying working project files..."
cp -r worksmart_ai_coach "dmg_temp/$FOLDER_NAME/"
cp pyproject.toml "dmg_temp/$FOLDER_NAME/"
cp requirements.txt "dmg_temp/$FOLDER_NAME/"
cp LICENSE "dmg_temp/$FOLDER_NAME/"
cp README.md "dmg_temp/$FOLDER_NAME/"

# Create a simple launch script that users can double-click
cat > "dmg_temp/$FOLDER_NAME/🚀 Launch WorkSmart AI Coach.command" << 'EOF'
#!/bin/bash
echo "🚀 WorkSmart AI Coach Launcher"
echo "=============================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"

# Check Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Python 3 not found!"
    echo "Please install Python 3 from python.org"
    read -p "Press Enter to open python.org..." 
    open "https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create config if needed
CONFIG_DIR="$HOME/.worksmart-ai-coach"
mkdir -p "$CONFIG_DIR"

# Check if user wants to reconfigure
if [ -f "$CONFIG_DIR/.env" ]; then
    echo "📋 Existing configuration found at $CONFIG_DIR/.env"
    echo ""
    read -p "Do you want to reconfigure? (y/N): " RECONFIG
    if [[ $RECONFIG =~ ^[Yy]$ ]]; then
        echo "🔄 Starting reconfiguration..."
        FORCE_SETUP=true
    else
        echo "✅ Using existing configuration"
        FORCE_SETUP=false
    fi
else
    FORCE_SETUP=true
fi

if [ "$FORCE_SETUP" = true ]; then
    echo "📝 First-time setup - let's configure your AI coach..."
    echo ""
    
    # Prompt for CrossOver files path
    echo "📁 CrossOver Files Path"
    echo "Default: ~/crossoverFiles"
    read -p "Enter path (or press Enter for default): " CROSSOVER_PATH
    if [ -z "$CROSSOVER_PATH" ]; then
        CROSSOVER_PATH="~/crossoverFiles"
    fi
    
    echo ""
    echo "🔑 Anthropic API Key Setup"
    echo "This enables AI-powered coaching suggestions."
    echo "You can get a free API key at: https://console.anthropic.com"
    echo ""
    echo "Options:"
    echo "1. Enter your API key now (recommended)"
    echo "2. Skip for now (basic coaching only)"
    echo ""
    read -p "Choose (1 or 2): " CHOICE
    
    API_KEY=""
    if [ "$CHOICE" = "1" ]; then
        echo ""
        echo "🔐 Enter your Anthropic API key:"
        echo "(Your input will be hidden for security)"
        read -s API_KEY
        echo ""
        if [ -n "$API_KEY" ]; then
            echo "✅ API key saved!"
        else
            echo "⚠️  No API key entered - you can add it later"
        fi
    else
        echo "⚠️  Skipping API key - using basic coaching mode"
        echo "💡 You can add your API key later by editing ~/.worksmart-ai-coach/.env"
    fi
    
    # Create config file
    cat > "$CONFIG_DIR/.env" << ENVEOF
# WorkSmart AI Coach Configuration
CROSSOVER_FILES_PATH=$CROSSOVER_PATH
ENVEOF
    
    if [ -n "$API_KEY" ]; then
        echo "ANTHROPIC_API_KEY=$API_KEY" >> "$CONFIG_DIR/.env"
    else
        echo "# ANTHROPIC_API_KEY=your_api_key_here" >> "$CONFIG_DIR/.env"
    fi
    
    echo "" >> "$CONFIG_DIR/.env"
    echo "# Optional: Enable debug mode" >> "$CONFIG_DIR/.env"
    echo "# DEBUG=true" >> "$CONFIG_DIR/.env"
    
    echo ""
    echo "✅ Configuration saved to $CONFIG_DIR/.env"
    echo "💡 You can edit this file anytime to update settings"
    echo ""
fi

echo ""
echo "🎯 Starting WorkSmart AI Coach..."
echo "   (Press Ctrl+C to stop)"
echo ""

# Create data directory in user's home for session files
DATA_DIR="$HOME/.worksmart-ai-coach/data"
mkdir -p "$DATA_DIR"

# Change to data directory so session files are saved there (not in read-only DMG)
cd "$DATA_DIR"

# Set PYTHONPATH to include the app directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the exact command that works
python3 -m worksmart_ai_coach.launchers.enhanced

echo ""
echo "👋 WorkSmart AI Coach stopped"
read -p "Press Enter to close..."
EOF

# Make the launch script executable
chmod +x "dmg_temp/$FOLDER_NAME/🚀 Launch WorkSmart AI Coach.command"

# Create simple installation instructions
cat > "dmg_temp/📖 How to Use.txt" << 'EOF'
WorkSmart AI Coach v1.0.0
========================

🚀 Quick Start:
1. Copy "WorkSmart AI Coach" folder to your Desktop or Applications
2. Double-click "🚀 Launch WorkSmart AI Coach.command"
3. The app will start in Terminal

📋 Requirements:
- Python 3.8+ (download from python.org if needed)
- macOS Terminal

🔧 Configuration:
- Config file: ~/.worksmart-ai-coach/.env
- Edit to set your CrossOver files path and API key

🆘 If double-click doesn't work:
1. Open Terminal
2. Navigate to the folder: cd "/path/to/WorkSmart AI Coach"
3. Run: python3 -m worksmart_ai_coach.launchers.enhanced

✨ That's it! Simple and reliable.
EOF

# Create the DMG
echo "💿 Creating DMG..."
if command -v hdiutil >/dev/null 2>&1; then
    hdiutil create -volname "WorkSmart AI Coach" \
                   -srcfolder dmg_temp \
                   -ov \
                   -format UDZO \
                   "$DMG_NAME.dmg"
    
    SIZE=$(du -h "$DMG_NAME.dmg" | cut -f1)
    
    echo ""
    echo "✅ Simple DMG Created!"
    echo "📦 File: $DMG_NAME.dmg"
    echo "📊 Size: $SIZE"
    echo ""
    echo "🎯 User experience:"
    echo "1. Download and open DMG"
    echo "2. Copy folder to Desktop/Applications"
    echo "3. Double-click the 🚀 Launch script"
    echo "4. Your working coach starts in Terminal!"
    
    # Clean up
    rm -rf dmg_temp
    
    echo ""
    echo "🎉 Ready! This DMG contains your exact working files."
    
else
    echo "❌ hdiutil not found"
    exit 1
fi