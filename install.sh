#!/bin/bash

echo "🚀 WorkSmart AI Coach Installer"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the package directory containing pyproject.toml"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check pip installation
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is required but not installed"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo "✅ pip3 found: $(pip3 --version)"

# Install the package
echo ""
echo "📦 Installing WorkSmart AI Coach..."
pip3 install -e . --user

if [ $? -eq 0 ]; then
    echo "✅ Installation successful!"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        echo ""
        echo "📝 Creating .env configuration file..."
        cat > .env << EOF
# WorkSmart AI Coach Configuration
# Set your Anthropic API key for AI features (optional)
ANTHROPIC_API_KEY=your_api_key_here

# Set path to crossover files (adjust to your system)
CROSSOVER_FILES_PATH=~/crossoverFiles

# Uncomment to enable debug logging
# DEBUG=true
EOF
        echo "✅ Created .env file - please edit it with your settings"
    else
        echo "✅ Using existing .env file"
    fi
    
    # Test the installation
    echo ""
    echo "🧪 Testing installation..."
    if command -v wsai &> /dev/null; then
        echo "✅ wsai command available"
        wsai --version
    else
        echo "⚠️  wsai command not found in PATH. You may need to:"
        echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "   (Add this to your ~/.bashrc or ~/.zshrc)"
    fi
    
    echo ""
    echo "🎉 Installation complete!"
    echo ""
    echo "Available commands:"
    echo "  wsai                    - Main CLI"
    echo "  worksmart-ai-coach      - Full name CLI"
    echo "  worksmart-enhanced      - Enhanced mode launcher"
    echo ""
    echo "To get started:"
    echo "1. Edit .env file with your settings"
    echo "2. Run: wsai start --mode enhanced"
    echo "3. Or run: worksmart-enhanced"
    
else
    echo "❌ Installation failed!"
    exit 1
fi