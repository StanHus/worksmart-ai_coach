#!/bin/bash
# WorkSmart AI Coach - Universal Quick Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/crossover-io/worksmart-ai-coach/main/quick_install.sh | bash

set -e

echo "🚀 WorkSmart AI Coach - Universal Installer"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

echo -e "${BLUE}🖥️  Detected OS: $OS${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Python if needed
install_python() {
    echo -e "${YELLOW}📦 Installing Python...${NC}"
    if [[ "$OS" == "macos" ]]; then
        if command_exists brew; then
            brew install python3
        else
            echo -e "${RED}❌ Homebrew not found. Please install Python 3 manually from python.org${NC}"
            exit 1
        fi
    elif [[ "$OS" == "linux" ]]; then
        if command_exists apt-get; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip
        elif command_exists yum; then
            sudo yum install -y python3 python3-pip
        elif command_exists pacman; then
            sudo pacman -S python python-pip
        else
            echo -e "${RED}❌ Could not detect package manager. Please install Python 3 manually${NC}"
            exit 1
        fi
    else
        echo -e "${RED}❌ Automatic Python installation not supported on $OS${NC}"
        echo "Please install Python 3 from python.org"
        exit 1
    fi
}

# Check Python installation
echo -e "${BLUE}🐍 Checking Python installation...${NC}"
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✅ Found: $PYTHON_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Python 3 not found${NC}"
    read -p "Install Python 3? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_python
    else
        echo -e "${RED}❌ Python 3 is required. Exiting.${NC}"
        exit 1
    fi
fi

# Check pip
if ! command_exists pip3; then
    echo -e "${YELLOW}⚠️  pip3 not found, installing...${NC}"
    python3 -m ensurepip --upgrade || {
        echo -e "${RED}❌ Could not install pip. Please install manually.${NC}"
        exit 1
    }
fi

# Install WorkSmart AI Coach
echo -e "${BLUE}📦 Installing WorkSmart AI Coach...${NC}"

# Try PyPI first (when published)
if pip3 install worksmart-ai-coach --user 2>/dev/null; then
    echo -e "${GREEN}✅ Installed from PyPI${NC}"
    INSTALL_METHOD="pypi"
else
    echo -e "${YELLOW}⚠️  PyPI install failed, installing from GitHub...${NC}"
    pip3 install git+https://github.com/crossover-io/worksmart-ai-coach.git --user
    INSTALL_METHOD="github"
fi

# Add to PATH if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo -e "${YELLOW}📝 Adding ~/.local/bin to PATH...${NC}"
    SHELL_CONFIG=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_CONFIG="$HOME/.bashrc"
    fi
    
    if [[ -n "$SHELL_CONFIG" ]]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
        echo -e "${GREEN}✅ Added to $SHELL_CONFIG${NC}"
        echo -e "${YELLOW}⚠️  Please run: source $SHELL_CONFIG${NC}"
    fi
fi

# Create config directory
CONFIG_DIR="$HOME/.worksmart-ai-coach"
mkdir -p "$CONFIG_DIR"

# Create .env file
if [[ ! -f "$CONFIG_DIR/.env" ]]; then
    echo -e "${BLUE}📝 Creating configuration file...${NC}"
    
    # Detect CrossOver files path
    CROSSOVER_PATH=""
    if [[ -d "$HOME/crossoverFiles" ]]; then
        CROSSOVER_PATH="$HOME/crossoverFiles"
    elif [[ -d "$HOME/CrossOverFiles" ]]; then
        CROSSOVER_PATH="$HOME/CrossOverFiles"
    else
        echo -e "${YELLOW}❓ CrossOver files directory not found${NC}"
        read -p "Enter path to CrossOver files (or press Enter for default ~/crossoverFiles): " USER_PATH
        if [[ -n "$USER_PATH" ]]; then
            CROSSOVER_PATH="$USER_PATH"
        else
            CROSSOVER_PATH="$HOME/crossoverFiles"
        fi
    fi
    
    cat > "$CONFIG_DIR/.env" << EOF
# WorkSmart AI Coach Configuration
CROSSOVER_FILES_PATH=$CROSSOVER_PATH

# Optional: Add your Anthropic API key for AI features
# ANTHROPIC_API_KEY=your_api_key_here

# Optional: Enable debug mode
# DEBUG=true
EOF
    echo -e "${GREEN}✅ Created config file at $CONFIG_DIR/.env${NC}"
else
    echo -e "${GREEN}✅ Using existing config file${NC}"
fi

# Test installation
echo -e "${BLUE}🧪 Testing installation...${NC}"
export PATH="$HOME/.local/bin:$PATH"

if command_exists wsai; then
    echo -e "${GREEN}✅ wsai command available${NC}"
    wsai --version 2>/dev/null || echo -e "${GREEN}✅ wsai installed${NC}"
else
    echo -e "${YELLOW}⚠️  wsai not found in PATH, you may need to restart your terminal${NC}"
fi

# Create desktop shortcut (macOS)
if [[ "$OS" == "macos" ]]; then
    DESKTOP="$HOME/Desktop"
    if [[ -d "$DESKTOP" ]]; then
        echo -e "${BLUE}🖥️  Creating desktop shortcut...${NC}"
        cat > "$DESKTOP/WorkSmart AI Coach.command" << 'EOF'
#!/bin/bash
cd ~/
export PATH="$HOME/.local/bin:$PATH"
worksmart-enhanced
EOF
        chmod +x "$DESKTOP/WorkSmart AI Coach.command"
        echo -e "${GREEN}✅ Created desktop shortcut${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🎉 Installation Complete!${NC}"
echo ""
echo -e "${BLUE}Available commands:${NC}"
echo "  wsai                    - Main CLI"
echo "  worksmart-ai-coach      - Full CLI"
echo "  worksmart-enhanced      - Start enhanced mode"
echo ""
echo -e "${BLUE}Quick start:${NC}"
echo "  1. worksmart-enhanced"
echo "  2. Or: wsai start --mode enhanced"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Edit: $CONFIG_DIR/.env"
echo ""
if [[ "$OS" == "macos" && -f "$HOME/Desktop/WorkSmart AI Coach.command" ]]; then
    echo -e "${BLUE}🖥️  Double-click 'WorkSmart AI Coach' on your desktop to start!${NC}"
fi

echo -e "${YELLOW}Note: You may need to restart your terminal or run 'source ~/.bashrc' (or ~/.zshrc)${NC}"