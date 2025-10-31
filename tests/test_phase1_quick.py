"""
Quick validation test for Phase 1 changes
Validates code structure without requiring network access
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_parallel_utils_exist():
    """Test that parallel_utils module exists and has required components"""
    print("\n[TEST 1] Parallel Utilities Module Structure")
    print("-" * 60)

    from system.parallel_utils import (
        ParallelConfig,
        ParallelProcessor,
        ThreadParallelProcessor,
        MemoryMonitor
    )

    # Test ParallelConfig
    config = ParallelConfig(max_workers=4)
    assert config.max_workers == 4
    assert config.enable_parallel == True
    print("[PASS] ParallelConfig class exists and works")

    # Test MemoryMonitor
    mem = MemoryMonitor.get_process_memory_mb()
    assert mem > 0
    print(f"[PASS] MemoryMonitor works (using {mem:.1f} MB)")

    # Test ThreadParallelProcessor instantiation
    proc = ThreadParallelProcessor(max_workers=2)
    assert proc.max_workers == 2
    print("[PASS] ThreadParallelProcessor instantiates")

    # Test ParallelProcessor instantiation
    with ParallelProcessor(config) as proc:
        pass
    print("[PASS] ParallelProcessor context manager works")

    return True


def test_classify_etfs_parallel_exists():
    """Test that classify_etfs_parallel method exists in ETFRiskClassifier"""
    print("\n[TEST 2] ETF Risk Classifier - Parallel Method")
    print("-" * 60)

    from analyzers.etf_risk_classifier import ETFRiskClassifier

    classifier = ETFRiskClassifier()

    # Check that both methods exist
    assert hasattr(classifier, 'classify_etfs'), "classify_etfs method missing"
    assert hasattr(classifier, 'classify_etfs_parallel'), "classify_etfs_parallel method missing"

    print("[PASS] classify_etfs method exists")
    print("[PASS] classify_etfs_parallel method exists")

    # Check method signatures
    import inspect

    seq_sig = inspect.signature(classifier.classify_etfs)
    par_sig = inspect.signature(classifier.classify_etfs_parallel)

    assert 'etf_tickers' in seq_sig.parameters
    assert 'etf_tickers' in par_sig.parameters
    assert 'max_workers' in par_sig.parameters

    print("[PASS] classify_etfs_parallel has max_workers parameter")
    print(f"[PASS] Signatures are correct")

    return True


def test_code_quality():
    """Test that code changes don't have obvious syntax errors"""
    print("\n[TEST 3] Code Quality & Syntax")
    print("-" * 60)

    import py_compile

    files_to_check = [
        'system/parallel_utils.py',
        'analyzers/etf_risk_classifier.py',
        'test_phase1_parallelization.py'
    ]

    for file_path in files_to_check:
        full_path = Path(file_path)
        assert full_path.exists(), f"{file_path} does not exist"

        try:
            py_compile.compile(str(full_path), doraise=True)
            print(f"[PASS] {file_path} - valid Python syntax")
        except py_compile.PyCompileError as e:
            print(f"[FAIL] {file_path} - syntax error: {e}")
            return False

    return True


def test_imports():
    """Test that all imports work correctly"""
    print("\n[TEST 4] Module Imports")
    print("-" * 60)

    try:
        from system.parallel_utils import (
            ParallelConfig, ParallelProcessor, ThreadParallelProcessor, MemoryMonitor,
            parallel_map, threaded_map
        )
        print("[PASS] system.parallel_utils imports work")

        from analyzers.etf_risk_classifier import ETFRiskClassifier
        print("[PASS] analyzers.etf_risk_classifier imports work")

        from data_manager.data_manager import ETFDataManager
        print("[PASS] data_manager.data_manager imports work")

        return True
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def main():
    print("=" * 60)
    print("PHASE 1 QUICK VALIDATION TEST")
    print("=" * 60)

    try:
        all_pass = True

        all_pass &= test_code_quality()
        all_pass &= test_imports()
        all_pass &= test_parallel_utils_exist()
        all_pass &= test_classify_etfs_parallel_exists()

        print("\n" + "=" * 60)
        if all_pass:
            print("SUCCESS: All validation tests passed!")
            print("\nPhase 1.1-1.3 Status:")
            print("  - Parallel infrastructure: WORKING")
            print("  - ETF download parallelization: READY FOR TESTING")
            print("  - Code quality: GOOD")
            print("\nNext steps: Test with actual ETF downloads")
        else:
            print("FAILURE: Some tests failed")
            return 1
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
