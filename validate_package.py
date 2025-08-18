#!/usr/bin/env python3
"""
WorkSmart AI Coach Package Validation Script
Tests all core functionality to ensure the package is working correctly.
"""

def test_imports():
    """Test all core imports."""
    try:
        from worksmart_ai_coach.core.coach import AICoach
        from worksmart_ai_coach.core.telemetry import WorkSmartTelemetryCollector, WorkSmartTelemetryAnalyzer
        from worksmart_ai_coach.core.personalized_coach import PersonalizedAICoach
        from worksmart_ai_coach.core.adaptive_learning import AdaptiveLearningSystem
        from worksmart_ai_coach.core.context_tracker import ContextHistoryTracker
        from worksmart_ai_coach.core.micro_interventions import MicroInterventionSystem
        from worksmart_ai_coach.cli.main import main
        from worksmart_ai_coach.launchers.enhanced import EnhancedProductionLauncher
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_core_initialization():
    """Test initialization of core components."""
    try:
        from worksmart_ai_coach.core.personalized_coach import PersonalizedAICoach
        from worksmart_ai_coach.core.telemetry import WorkSmartTelemetryCollector
        
        coach = PersonalizedAICoach()
        print(f"✅ PersonalizedAICoach: {len(coach.personal_patterns)} patterns loaded")
        
        telemetry = WorkSmartTelemetryCollector()
        print("✅ Telemetry collector initialized")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def test_intelligence_systems():
    """Test AI intelligence systems."""
    try:
        from worksmart_ai_coach.core.personalized_coach import PersonalizedAICoach
        
        coach = PersonalizedAICoach()
        
        # Test subsystems
        assert hasattr(coach, 'context_tracker')
        assert hasattr(coach, 'micro_interventions') 
        assert hasattr(coach, 'adaptive_learning')
        
        print("✅ Context tracker ready")
        print("✅ Micro-interventions ready")
        print("✅ Adaptive learning ready")
        
        return True
    except Exception as e:
        print(f"❌ Intelligence systems test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 WorkSmart AI Coach Package Validation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Core Initialization", test_core_initialization), 
        ("Intelligence Systems", test_intelligence_systems),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for use.")
        print("\n📖 Available commands:")
        print("   wsai --help              # Main CLI")
        print("   worksmart-enhanced help  # Enhanced launcher") 
        print("   worksmart-bridge --help  # Bridge system")
        return True
    else:
        print("❌ Some tests failed. Package may have issues.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)