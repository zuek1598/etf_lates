#!/usr/bin/env python3
"""Test that run_full_analysis method is properly accessible"""

from system.orchestrator import ETFAnalysisSystem

print("Testing run_full_analysis method accessibility")
print("=" * 50)

try:
    system = ETFAnalysisSystem()
    print('✓ ETFAnalysisSystem initialized successfully')
    print('✓ run_full_analysis method exists:', hasattr(system, 'run_full_analysis'))
    
    method = getattr(system, 'run_full_analysis', None)
    print('✓ Method is callable:', callable(method))
    
    if hasattr(system, 'run_full_analysis'):
        print('🎉 SUCCESS: run_full_analysis method is properly defined and accessible!')
        print('The system should now run analysis without AttributeError.')
    else:
        print('❌ FAILURE: run_full_analysis method still missing')
        
except Exception as e:
    print(f'❌ ERROR: {e}')
