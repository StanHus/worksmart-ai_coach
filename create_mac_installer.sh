#!/bin/bash
# Create Mac App Bundle and Installer

set -e

echo "🍎 Building Mac App Bundle for WorkSmart AI Coach"
echo "==============================================="

# Check if py2app is installed
if ! python3 -c "import py2app" 2>/dev/null; then
    echo "📦 Installing py2app..."
    pip3 install py2app
fi

# Clean previous builds
rm -rf build/ dist/

# Build the app
echo "🔨 Building app bundle..."
python3 build_mac_app.py py2app

if [ -d "dist/WorkSmart AI Coach.app" ]; then
    echo "✅ App bundle created successfully!"
    
    # Create a simple installer package
    echo "📦 Creating installer package..."
    
    # Create temporary installer structure
    mkdir -p installer_temp/Applications
    cp -R "dist/WorkSmart AI Coach.app" installer_temp/Applications/
    
    # Create simple installer script
    cat > installer_temp/install.sh << 'EOF'
#!/bin/bash
echo "🚀 Installing WorkSmart AI Coach..."

# Copy app to Applications
sudo cp -R "WorkSmart AI Coach.app" /Applications/

# Create config directory
mkdir -p "$HOME/.worksmart-ai-coach"

# Copy env template if not exists
if [ ! -f "$HOME/.worksmart-ai-coach/.env" ]; then
    cat > "$HOME/.worksmart-ai-coach/.env" << 'ENVEOF'
# WorkSmart AI Coach Configuration
CROSSOVER_FILES_PATH=~/crossoverFiles

# Optional: Add your Anthropic API key
# ANTHROPIC_API_KEY=your_api_key_here
ENVEOF
    echo "📝 Created config file at ~/.worksmart-ai-coach/.env"
fi

echo "✅ Installation complete!"
echo "You can now find WorkSmart AI Coach in your Applications folder"
EOF
    
    chmod +x installer_temp/install.sh
    
    # Create DMG
    if command -v hdiutil &> /dev/null; then
        echo "💿 Creating DMG installer..."
        hdiutil create -volname "WorkSmart AI Coach" -srcfolder installer_temp -ov -format UDZO "WorkSmart-AI-Coach-Installer.dmg"
        echo "✅ Created: WorkSmart-AI-Coach-Installer.dmg"
    fi
    
    # Cleanup
    rm -rf installer_temp
    
    echo ""
    echo "🎉 Mac App Bundle created successfully!"
    echo "📁 Location: dist/WorkSmart AI Coach.app"
    echo "💿 Installer: WorkSmart-AI-Coach-Installer.dmg"
    echo ""
    echo "To test: open 'dist/WorkSmart AI Coach.app'"
    
else
    echo "❌ App bundle creation failed"
    exit 1
fi