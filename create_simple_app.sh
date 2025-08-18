#!/bin/bash
# Create a simple Mac App bundle without py2app

set -e

echo "🍎 Creating Simple Mac App Bundle"
echo "================================="

APP_NAME="WorkSmart AI Coach"
APP_DIR="dist/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean and create directories
rm -rf dist/
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create Info.plist
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>WorkSmart AI Coach</string>
    <key>CFBundleIdentifier</key>
    <string>io.crossover.worksmart-ai-coach</string>
    <key>CFBundleName</key>
    <string>WorkSmart AI Coach</string>
    <key>CFBundleDisplayName</key>
    <string>WorkSmart AI Coach</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>WSAI</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.12</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 CrossOver. All rights reserved.</string>
</dict>
</plist>
EOF

# Create the launcher script
cat > "$MACOS_DIR/WorkSmart AI Coach" << 'EOF'
#!/bin/bash

# Get the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESOURCES_DIR="$DIR/../Resources"
APP_DIR="$DIR/../../.."

# Set up environment
export PYTHONPATH="$RESOURCES_DIR:$PYTHONPATH"
export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

# Create config directory if needed
CONFIG_DIR="$HOME/.worksmart-ai-coach"
mkdir -p "$CONFIG_DIR"

# Create .env if not exists
if [ ! -f "$CONFIG_DIR/.env" ]; then
    cat > "$CONFIG_DIR/.env" << 'ENVEOF'
# WorkSmart AI Coach Configuration
CROSSOVER_FILES_PATH=~/crossoverFiles

# Optional: Add your Anthropic API key
# ANTHROPIC_API_KEY=your_api_key_here

# Debug mode (optional)
# DEBUG=true
ENVEOF
    
    # Show setup dialog
    osascript << 'APPLESCRIPT'
display dialog "Welcome to WorkSmart AI Coach!

A configuration file has been created at:
~/.worksmart-ai-coach/.env

Please edit this file to set your CrossOver files path before using the app.

Default path: ~/crossoverFiles" buttons {"Open Config Folder", "Continue"} default button "Continue"
set the button_pressed to the button returned of the result
if the button_pressed is "Open Config Folder" then
    do shell script "open ~/.worksmart-ai-coach"
end if
APPLESCRIPT
fi

# Try to run the app
cd "$RESOURCES_DIR"

# Check if Python 3 is available
if command -v python3 >/dev/null 2>&1; then
    # Try to run the enhanced launcher
    if [ -f "worksmart_ai_coach/launchers/enhanced.py" ]; then
        python3 -m worksmart_ai_coach.launchers.enhanced
    else
        osascript -e 'display alert "Error" message "WorkSmart AI Coach files not found. Please reinstall the application."'
    fi
else
    # Show Python installation dialog
    osascript << 'APPLESCRIPT'
display alert "Python 3 Required" message "WorkSmart AI Coach requires Python 3 to run. Please install Python 3 from python.org or using Homebrew:

brew install python3

Then restart this application." buttons {"Open Python.org", "OK"} default button "OK"
set the button_pressed to the button returned of the result
if the button_pressed is "Open Python.org" then
    open location "https://www.python.org/downloads/"
end if
APPLESCRIPT
fi
EOF

# Make launcher executable
chmod +x "$MACOS_DIR/WorkSmart AI Coach"

# Copy the Python package to Resources
echo "📦 Copying Python package..."
cp -r worksmart_ai_coach "$RESOURCES_DIR/"
cp requirements.txt "$RESOURCES_DIR/"
cp .env.template "$RESOURCES_DIR/"
cp README.md "$RESOURCES_DIR/"
cp LICENSE "$RESOURCES_DIR/"

# Create a simple icon (you can replace this with a real icon file)
# For now, create a placeholder
mkdir -p "$RESOURCES_DIR/icon.iconset"
# Create basic icon placeholder - in real use, you'd have proper icon files

echo "✅ Mac App bundle created: $APP_DIR"

# Test the app
echo "🧪 Testing app bundle..."
if [ -d "$APP_DIR" ]; then
    echo "✅ App bundle structure created successfully"
    echo "📁 Location: $APP_DIR"
    echo ""
    echo "To test: open '$APP_DIR'"
    
    # Create the DMG
    echo "💿 Creating DMG..."
    DMG_NAME="WorkSmart-AI-Coach-v1.0.0"
    
    # Create temporary DMG directory
    mkdir -p dmg_temp
    cp -r "$APP_DIR" dmg_temp/
    
    # Create Applications symlink
    ln -sf /Applications dmg_temp/Applications
    
    # Create README for DMG
    cat > dmg_temp/README.txt << 'EOF'
WorkSmart AI Coach v1.0.0
========================

Installation:
1. Drag "WorkSmart AI Coach.app" to the Applications folder
2. Double-click to run
3. Follow the setup instructions

Requirements:
- macOS 10.12 or later  
- Python 3.8 or later (will prompt to install if needed)

Configuration:
- Config file: ~/.worksmart-ai-coach/.env
- Set CROSSOVER_FILES_PATH to your CrossOver files directory

Support:
https://github.com/crossover-io/worksmart-ai-coach

EOF
    
    # Create the DMG
    if command -v hdiutil >/dev/null 2>&1; then
        hdiutil create -volname "WorkSmart AI Coach" \
                       -srcfolder dmg_temp \
                       -ov \
                       -format UDZO \
                       -imagekey zlib-level=9 \
                       "$DMG_NAME.dmg"
        
        echo "✅ Created DMG: $DMG_NAME.dmg"
        
        # Get file size
        if [ -f "$DMG_NAME.dmg" ]; then
            SIZE=$(du -h "$DMG_NAME.dmg" | cut -f1)
            echo "📊 DMG size: $SIZE"
        fi
    else
        echo "⚠️  hdiutil not found - cannot create DMG"
    fi
    
    # Cleanup
    rm -rf dmg_temp
    
    echo ""
    echo "🎉 Complete! Your Mac app is ready:"
    echo "📱 App: $APP_DIR"
    echo "💿 DMG: $DMG_NAME.dmg"
    echo ""
    echo "To distribute:"
    echo "1. Share the DMG file"
    echo "2. Users drag the app to Applications"
    echo "3. Double-click to run"
    
else
    echo "❌ Failed to create app bundle"
    exit 1
fi