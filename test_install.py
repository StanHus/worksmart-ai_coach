#!/usr/bin/env python3
"""
Test script to verify WorkSmart AI Coach installation
"""

import os
import sys
from pathlib import Path

def test_installation():
    print("🧪 Testing WorkSmart AI Coach Installation")
    print("=" * 50)
    
    success = True
    
    # Test 1: Import main modules
    try:
        import worksmart_ai_coach
        print("✅ Main package import: OK")
    except ImportError as e:
        print(f"❌ Main package import: FAILED ({e})")
        success = False
        
    # Test 2: Import core modules
    try:
        from worksmart_ai_coach.core.coach import AICoach
        print("✅ Core coach import: OK")
    except ImportError as e:
        print(f"❌ Core coach import: FAILED ({e})")
        success = False
        
    # Test 3: Import enhanced launcher
    try:
        from worksmart_ai_coach.launchers.enhanced import EnhancedProductionLauncher
        print("✅ Enhanced launcher import: OK")
    except ImportError as e:
        print(f"❌ Enhanced launcher import: FAILED ({e})")
        success = False
        
    # Test 4: Test environment variable configuration
    try:
        from worksmart_ai_coach.core.worksmart_reader import WorkSmartDataReader
        
        # Test default path
        reader1 = WorkSmartDataReader()
        default_path = str(reader1.base_path)
        print(f"✅ Default crossover path: {default_path}")
        
        # Test environment variable
        os.environ['CROSSOVER_FILES_PATH'] = '/tmp/test_crossover'
        reader2 = WorkSmartDataReader()
        env_path = str(reader2.base_path)
        print(f"✅ Environment variable path: {env_path}")
        
        if env_path == '/tmp/test_crossover':
            print("✅ Environment variable configuration: OK")
        else:
            print("❌ Environment variable configuration: FAILED")
            success = False
            
    except Exception as e:
        print(f"❌ Environment configuration test: FAILED ({e})")
        success = False
        
    # Test 5: Check CLI entry points
    cli_commands = ['wsai', 'worksmart-ai-coach', 'worksmart-enhanced']
    for cmd in cli_commands:
        try:
            import subprocess
            result = subprocess.run(['which', cmd], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ CLI command '{cmd}': Available at {result.stdout.strip()}")
            else:
                print(f"⚠️  CLI command '{cmd}': Not found in PATH")
        except Exception as e:
            print(f"❌ CLI command '{cmd}': Error checking ({e})")
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All core tests passed! Installation looks good.")
        print("\nNext steps:")
        print("1. Copy .env.template to .env")
        print("2. Edit .env with your CROSSOVER_FILES_PATH")
        print("3. Run: worksmart-enhanced")
    else:
        print("❌ Some tests failed. Please check the installation.")
        
    return success

if __name__ == "__main__":
    test_installation()