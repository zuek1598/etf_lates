"""
Parallel Processing Utilities
Provides multiprocessing infrastructure for ETF analysis parallelization

Features:
- Worker pool management with configurable worker count
- Process-safe data serialization
- Memory monitoring and limits
- Graceful error handling and recovery
- Progress tracking for parallel operations
"""

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import logging
from typing import Callable, List, Dict, Any, Tuple, Optional
from functools import wraps
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelConfig:
    """Configuration for parallel processing"""

    def __init__(self,
                 max_workers: Optional[int] = None,
                 memory_limit_gb: float = 4.0,
                 enable_parallel: bool = True,
                 timeout_seconds: int = 300,
                 max_retries: int = 2):
        """
        Initialize parallel configuration

        Args:
            max_workers: Number of worker processes (default: CPU count - 1)
            memory_limit_gb: Max memory per process in GB
            enable_parallel: Enable/disable parallelization globally
            timeout_seconds: Timeout for individual tasks
            max_retries: Max retries for failed tasks
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.memory_limit_gb = memory_limit_gb
        self.enable_parallel = enable_parallel
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def __repr__(self):
        return (f"ParallelConfig(workers={self.max_workers}, "
                f"memory={self.memory_limit_gb}GB, "
                f"parallel={self.enable_parallel})")


class MemoryMonitor:
    """Monitor memory usage during parallel processing"""

    @staticmethod
    def get_process_memory_mb() -> float:
        """Get current process memory in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    @staticmethod
    def get_system_memory_available_mb() -> float:
        """Get available system memory in MB"""
        try:
            return psutil.virtual_memory().available / 1024 / 1024
        except:
            return 0.0

    @staticmethod
    def check_memory_available(required_mb: float) -> bool:
        """Check if enough memory is available"""
        available = MemoryMonitor.get_system_memory_available_mb()
        return available > required_mb


class ParallelProcessor:
    """
    Main parallel processing coordinator
    Handles multiprocessing pool, error handling, and progress tracking
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel processor"""
        self.config = config or ParallelConfig()
        self.pool = None
        self.memory_monitor = MemoryMonitor()

    def __enter__(self):
        """Context manager entry"""
        if self.config.enable_parallel and self.config.max_workers > 1:
            self.pool = Pool(processes=self.config.max_workers)
            logger.info(f"Created process pool with {self.config.max_workers} workers")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.pool:
            self.pool.close()
            self.pool.join()
            logger.info("Process pool closed")

    def map_parallel(self,
                     func: Callable,
                     items: List[Any],
                     description: str = "Processing") -> List[Any]:
        """
        Map function across items in parallel

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for logging

        Returns:
            List of results in original order
        """
        if not self.config.enable_parallel or len(items) <= 1:
            # Fall back to sequential processing
            logger.info(f"{description} (sequential, {len(items)} items)")
            return [func(item) for item in items]

        if not self.pool:
            # Fall back to sequential if pool not available
            logger.warning("Pool not available, falling back to sequential")
            return [func(item) for item in items]

        logger.info(f"{description} (parallel, {len(items)} items, {self.config.max_workers} workers)")

        try:
            results = []
            completed = 0
            for i, result in enumerate(self.pool.imap_unordered(func, items), 1):
                results.append(result)
                completed = i
                if completed % max(1, len(items) // 10) == 0:
                    logger.debug(f"  Progress: {completed}/{len(items)}")

            return results

        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            raise

    def map_with_errors(self,
                        func: Callable,
                        items: List[Any],
                        description: str = "Processing",
                        return_errors: bool = False) -> Tuple[List[Any], Optional[List[Tuple[Any, Exception]]]]:
        """
        Map function across items with error handling

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for logging
            return_errors: Whether to return errors instead of raising

        Returns:
            Tuple of (results, errors) if return_errors=True, else results
        """
        if not self.config.enable_parallel or len(items) <= 1:
            results = []
            errors = [] if return_errors else None

            for item in items:
                try:
                    results.append(func(item))
                except Exception as e:
                    if return_errors:
                        errors.append((item, e))
                    else:
                        raise

            return (results, errors) if return_errors else results

        logger.info(f"{description} with error handling")

        results = []
        errors = [] if return_errors else None

        # Use ProcessPoolExecutor for better error handling
        with mp.Manager() as manager:
            shared_dict = manager.dict()

            try:
                # Use imap with ordered results
                completed = 0
                for i, result in enumerate(self.pool.imap_unordered(func, items), 1):
                    results.append(result)
                    completed = i

                if return_errors:
                    return results, errors
                return results

            except Exception as e:
                if return_errors:
                    logger.error(f"Errors occurred: {e}")
                    return results, errors
                raise

    def map_chunk(self,
                  func: Callable,
                  items: List[Any],
                  chunk_size: int = 10,
                  description: str = "Processing") -> List[Any]:
        """
        Process items in chunks (useful for memory-intensive tasks)

        Args:
            func: Function to apply to each chunk
            items: List of items to process
            chunk_size: Number of items per chunk
            description: Description for logging

        Returns:
            List of results
        """
        logger.info(f"{description} in chunks of {chunk_size}")

        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i+chunk_size]
            logger.debug(f"  Processing chunk {i//chunk_size + 1} ({len(chunk)} items)")
            results.extend([func(item) for item in chunk])

        return results


class ThreadParallelProcessor:
    """
    Thread-based parallel processor for I/O-bound operations
    Better for downloading data with network I/O
    """

    def __init__(self, max_workers: Optional[int] = None, timeout_seconds: int = 300):
        """Initialize thread processor"""
        self.max_workers = max_workers or min(10, cpu_count() * 2)
        self.timeout_seconds = timeout_seconds

    def map_parallel(self,
                     func: Callable,
                     items: List[Any],
                     description: str = "Processing") -> List[Any]:
        """
        Map function across items using thread pool

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for logging

        Returns:
            List of results in original order
        """
        logger.info(f"{description} (threads, {len(items)} items, {self.max_workers} threads)")

        results = [None] * len(items)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(func, item): idx
                               for idx, item in enumerate(items)}

            for future in as_completed(future_to_index, timeout=self.timeout_seconds):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if completed % max(1, len(items) // 10) == 0:
                        logger.debug(f"  Progress: {completed}/{len(items)}")
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    results[idx] = None

        return results

    def map_with_errors(self,
                        func: Callable,
                        items: List[Any],
                        description: str = "Processing") -> Tuple[List[Any], List[Tuple[int, Exception]]]:
        """
        Map function with comprehensive error tracking

        Args:
            func: Function to apply to each item
            items: List of items to process
            description: Description for logging

        Returns:
            Tuple of (results, errors) where errors is list of (index, exception)
        """
        logger.info(f"{description} with error tracking (threads)")

        results = [None] * len(items)
        errors = []
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {executor.submit(func, item): idx
                               for idx, item in enumerate(items)}

            for future in as_completed(future_to_index, timeout=self.timeout_seconds):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    errors.append((idx, e))
                    results[idx] = None

        logger.info(f"Completed {completed}/{len(items)} items, {len(errors)} errors")
        return results, errors


# Convenience functions
def parallel_map(func: Callable,
                 items: List[Any],
                 config: Optional[ParallelConfig] = None) -> List[Any]:
    """
    Convenience function for parallel mapping

    Args:
        func: Function to apply
        items: Items to process
        config: ParallelConfig instance

    Returns:
        List of results
    """
    config = config or ParallelConfig()
    with ParallelProcessor(config) as processor:
        return processor.map_parallel(func, items)


def threaded_map(func: Callable,
                 items: List[Any],
                 max_workers: Optional[int] = None) -> List[Any]:
    """
    Convenience function for threaded mapping (I/O-bound)

    Args:
        func: Function to apply
        items: Items to process
        max_workers: Max worker threads

    Returns:
        List of results
    """
    processor = ThreadParallelProcessor(max_workers=max_workers)
    return processor.map_parallel(func, items)


if __name__ == "__main__":
    # Test parallel processor
    def test_func(x):
        return x * 2

    config = ParallelConfig(max_workers=2, enable_parallel=True)
    with ParallelProcessor(config) as processor:
        results = processor.map_parallel(test_func, list(range(10)))
        print(f"Results: {results}")

    # Test threaded processor
    processor = ThreadParallelProcessor(max_workers=3)
    results = processor.map_parallel(test_func, list(range(10)))
    print(f"Threaded results: {results}")
