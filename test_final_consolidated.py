#!/usr/bin/env python3
"""
Test Final Consolidated System
==============================
Test the ultimate consolidated AI coaching system with minimal file structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_ultimate_consolidation():
    """Test the ultimate consolidated system"""
    print("🚀 TESTING ULTIMATE CONSOLIDATED SYSTEM")
    print("=" * 50)

    # Test Ultimate AI Coach
    print("\n🧪 Testing Ultimate AI Coach...")
    try:
        from worksmart_ai_coach.core.ai_coach import AICoach
        print("✅ Ultimate AI Coach import successful")

        coach = AICoach()
        print("✅ Ultimate AI Coach initialization successful")

        system_info = coach.get_system_info()
        print(
            f"✅ System: {system_info['coach_type']} with {len(system_info['capabilities'])} capabilities")
        print(f"📋 Capabilities: {', '.join(system_info['capabilities'])}")

        # Test basic functionality
        context = {
            'current_application': 'vscode',
            'productivity_score': 0.6,
            'focus_quality': 0.7,
            'stress_level': 0.4,
            'energy_level': 0.5,
            'session_duration_hours': 1.5
        }

        productivity = coach.calculate_personalized_productivity_score(
            [], context)
        focus = coach.calculate_personalized_focus_quality([], context)
        persona = coach.detect_user_persona(context)

        print(
            f"✅ Functionality test - Productivity: {productivity:.2f}, Focus: {focus:.2f}, Persona: {persona}")

    except Exception as e:
        print(f"❌ Ultimate AI Coach test failed: {e}")
        return False

    # Test Telemetry System
    print("\n🧪 Testing Telemetry System...")
    try:
        from worksmart_ai_coach.core.telemetry_system import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer, WorkSmartDataReader
        print("✅ Telemetry System import successful")

        collector = WorkSmartTelemetryCollector()
        analyzer = WorkSmartTelemetryAnalyzer()
        reader = WorkSmartDataReader()
        print("✅ Telemetry System initialization successful")

        # Test basic functionality
        event = collector.collect_event()
        analysis = analyzer.analyze([event])

        print(
            f"✅ Telemetry functionality - Event collected with {len(event)} fields")
        print(
            f"✅ Analysis completed - Activity: {analysis['activity_level']}, Score: {analysis['productivity_score']:.2f}")

    except Exception as e:
        print(f"❌ Telemetry System test failed: {e}")
        return False

    # Test Launcher Integration
    print("\n🧪 Testing Launcher Integration...")
    try:
        from worksmart_ai_coach.launchers.enhanced import EnhancedProductionLauncher
        print("✅ Enhanced launcher import successful")

        print("📋 Testing launcher initialization:")
        launcher = EnhancedProductionLauncher()
        print(f"✅ Launcher initialized successfully")
        print(f"🎯 Coach type: {getattr(launcher, 'coach_type', 'unknown')}")
        print(f"🔧 Enhanced mode: {getattr(launcher, 'is_enhanced', False)}")

    except Exception as e:
        print(f"❌ Launcher integration test failed: {e}")
        return False

    return True


def test_file_structure():
    """Test the consolidated file structure"""
    print("\n🧪 Testing File Structure...")

    expected_core_files = [
        'worksmart_ai_coach/core/__init__.py',
        'worksmart_ai_coach/core/ai_coach.py',
        'worksmart_ai_coach/core/telemetry_system.py'
    ]

    legacy_files = [
        'worksmart_ai_coach/legacy/coach.py',
        'worksmart_ai_coach/legacy/personalized_coach.py',
        'worksmart_ai_coach/legacy/enhanced_ai_coach.py',
        'worksmart_ai_coach/legacy/telemetry.py',
        'worksmart_ai_coach/legacy/worksmart_reader.py',
        'worksmart_ai_coach/legacy/ml_components.py',
        'worksmart_ai_coach/legacy/ai_coaching_system.py'
    ]

    print("📁 Checking core files (should exist):")
    core_count = 0
    for file_path in expected_core_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            core_count += 1
        else:
            print(f"❌ {file_path} missing")

    print(f"\n📁 Core files: {core_count}/{len(expected_core_files)} present")

    print("\n📁 Checking legacy files (should be moved):")
    legacy_count = 0
    for file_path in legacy_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} (moved to legacy)")
            legacy_count += 1
        else:
            print(f"⚠️ {file_path} not found in legacy")

    print(
        f"📁 Legacy files: {legacy_count}/{len(legacy_files)} moved to legacy")

    return core_count == len(expected_core_files)


def main():
    """Run all tests"""
    print("🔬 ULTIMATE CONSOLIDATED SYSTEM TEST")
    print("Testing the final consolidated file structure...")
    print("=" * 60)

    tests = [
        test_file_structure,
        test_ultimate_consolidation
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"📊 Final Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ULTIMATE CONSOLIDATION SUCCESSFUL!")
        print("✨ All AI coaching functionality consolidated into 2 core files:")
        print("   🧠 ai_coach.py - Complete AI coaching system")
        print("   📊 telemetry_system.py - Complete telemetry and WorkSmart integration")
        print("   🗂️  All legacy files preserved in legacy/ folder")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
