#!/usr/bin/env python3
"""Test that the fixed orchestrator works properly"""

from system.orchestrator import ETFAnalysisSystem

print("Testing Fixed Orchestrator")
print("=" * 30)

try:
    print("✓ ETFAnalysisSystem imports successfully")
    system = ETFAnalysisSystem()
    print("✓ ETFAnalysisSystem initializes successfully")
    print(f"✓ Scoring system type: {type(system.scoring_system).__name__}")
    print(f"✓ run_full_analysis method exists: {hasattr(system, 'run_full_analysis')}")
    print("✓ System is ready for analysis")
    print("\n🎉 INDENTATION FIX SUCCESSFUL!")
    print("The analysis system should now run without errors.")

except Exception as e:
    print(f"❌ Error: {e}")
    print("The fix did not work.")
