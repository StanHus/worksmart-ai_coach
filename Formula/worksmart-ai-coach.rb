class WorksmartAiCoach < Formula
  include Language::Python::Virtualenv

  desc "AI-powered productivity coaching for WorkSmart tracker"
  homepage "https://github.com/crossover-io/worksmart-ai-coach"
  url "https://github.com/crossover-io/worksmart-ai-coach/archive/v1.0.0.tar.gz"
  sha256 "YOUR_SHA256_HASH_HERE"  # Will need to update when published
  license "MIT"

  depends_on "python@3.11"

  resource "anthropic" do
    url "https://files.pythonhosted.org/packages/source/a/anthropic/anthropic-0.35.0.tar.gz"
    sha256 "ANTHROPIC_SHA256_HERE"
  end

  resource "pynput" do
    url "https://files.pythonhosted.org/packages/source/p/pynput/pynput-1.7.6.tar.gz"
    sha256 "PYNPUT_SHA256_HERE"
  end

  resource "psutil" do
    url "https://files.pythonhosted.org/packages/source/p/psutil/psutil-5.9.6.tar.gz"
    sha256 "PSUTIL_SHA256_HERE"
  end

  resource "python-dotenv" do
    url "https://files.pythonhosted.org/packages/source/p/python-dotenv/python_dotenv-1.0.0.tar.gz"
    sha256 "DOTENV_SHA256_HERE"
  end

  resource "aiofiles" do
    url "https://files.pythonhosted.org/packages/source/a/aiofiles/aiofiles-23.2.1.tar.gz"
    sha256 "AIOFILES_SHA256_HERE"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/source/c/click/click-8.1.7.tar.gz"
    sha256 "CLICK_SHA256_HERE"
  end

  resource "numpy" do
    url "https://files.pythonhosted.org/packages/source/n/numpy/numpy-1.24.4.tar.gz"
    sha256 "NUMPY_SHA256_HERE"
  end

  def install
    virtualenv_install_with_resources
    
    # Create config directory
    (var/"worksmart-ai-coach").mkpath
    
    # Install .env template
    (var/"worksmart-ai-coach").install ".env.template"
    
    # Create wrapper script
    (bin/"worksmart-ai-coach-setup").write <<~EOS
      #!/bin/bash
      CONFIG_DIR="$HOME/.worksmart-ai-coach"
      mkdir -p "$CONFIG_DIR"
      
      if [ ! -f "$CONFIG_DIR/.env" ]; then
        echo "🔧 Setting up WorkSmart AI Coach..."
        cp "#{var}/worksmart-ai-coach/.env.template" "$CONFIG_DIR/.env"
        echo "📝 Created config file at $CONFIG_DIR/.env"
        echo "Please edit it with your CrossOver files path"
      fi
      
      echo "✅ WorkSmart AI Coach is ready!"
      echo "Run: worksmart-enhanced"
    EOS
    
    chmod 0755, bin/"worksmart-ai-coach-setup"
  end

  def caveats
    <<~EOS
      WorkSmart AI Coach has been installed!
      
      First-time setup:
        worksmart-ai-coach-setup
      
      Start the coach:
        worksmart-enhanced
      
      Configuration file:
        ~/.worksmart-ai-coach/.env
      
      Available commands:
        wsai                    Main CLI
        worksmart-ai-coach      Full CLI  
        worksmart-enhanced      Enhanced mode
    EOS
  end

  test do
    system "#{bin}/wsai", "--version"
  end
end