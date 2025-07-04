"""Performance monitoring utilities for 3lc_tools."""

import functools
import time
import psutil
import gc
from typing import Any, Callable, Optional
from contextlib import contextmanager


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance.
    
    Tracks execution time and memory usage changes.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get initial state
        start_time = time.perf_counter()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection for more accurate memory measurements
        gc.collect()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Get final state
            end_time = time.perf_counter()
            gc.collect()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"⚡ {func.__name__}:")
            print(f"   Time: {execution_time:.3f}s")
            print(f"   Memory: {memory_delta:+.1f}MB")
            if hasattr(func, '__module__'):
                print(f"   Module: {func.__module__}")
        
        return result
    return wrapper


@contextmanager
def timing_context(name: str, print_result: bool = True):
    """Context manager for timing code blocks.
    
    Args:
        name: Name of the operation being timed
        print_result: Whether to print the timing result
        
    Yields:
        dict: Contains timing information that gets updated during execution
    """
    start_time = time.perf_counter()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = {"name": name, "start_time": start_time}
    
    try:
        yield result
    finally:
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        result.update({
            "execution_time": execution_time,
            "memory_delta": memory_delta,
            "end_time": end_time
        })
        
        if print_result:
            print(f"⏱️  {name}: {execution_time:.3f}s (Memory: {memory_delta:+.1f}MB)")


class PerformanceBenchmark:
    """Class for running performance benchmarks."""
    
    def __init__(self):
        self.results = []
        
    def benchmark_function(self, func: Callable, *args, name: Optional[str] = None, **kwargs) -> Any:
        """Benchmark a function call and store results.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            name: Optional name for the benchmark (defaults to function name)
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
        """
        benchmark_name = name or func.__name__
        
        with timing_context(benchmark_name, print_result=False) as timing:
            result = func(*args, **kwargs)
            
        self.results.append(timing)
        return result
        
    def benchmark_import(self, module_name: str) -> None:
        """Benchmark importing a module.
        
        Args:
            module_name: Name of the module to import
        """
        import importlib
        
        with timing_context(f"import {module_name}", print_result=False) as timing:
            try:
                importlib.import_module(module_name)
                timing["success"] = True
            except ImportError as e:
                timing["success"] = False
                timing["error"] = str(e)
                
        self.results.append(timing)
        
    def print_summary(self):
        """Print a summary of all benchmark results."""
        if not self.results:
            print("No benchmark results to display.")
            return
            
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        total_time = 0
        for result in self.results:
            name = result["name"]
            time_taken = result["execution_time"]
            memory_delta = result["memory_delta"]
            total_time += time_taken
            
            status = "✅" if result.get("success", True) else "❌"
            print(f"{status} {name:<40} {time_taken:>8.3f}s {memory_delta:>+8.1f}MB")
            
            if "error" in result:
                print(f"   Error: {result['error']}")
                
        print("-" * 60)
        print(f"{'TOTAL':<42} {total_time:>8.3f}s")
        print("="*60)
        
    def compare_with_baseline(self, baseline_results: dict):
        """Compare current results with baseline measurements.
        
        Args:
            baseline_results: Dictionary mapping benchmark names to baseline times
        """
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON WITH BASELINE")
        print("="*70)
        
        for result in self.results:
            name = result["name"]
            current_time = result["execution_time"]
            
            if name in baseline_results:
                baseline_time = baseline_results[name]
                improvement = (baseline_time - current_time) / baseline_time * 100
                
                if improvement > 0:
                    symbol = "🚀"
                    status = f"{improvement:+.1f}% faster"
                elif improvement < -5:  # Only show degradation if > 5%
                    symbol = "⚠️"
                    status = f"{abs(improvement):.1f}% slower"
                else:
                    symbol = "➡️"
                    status = "similar"
                    
                print(f"{symbol} {name:<35} {current_time:.3f}s vs {baseline_time:.3f}s ({status})")
            else:
                print(f"ℹ️  {name:<35} {current_time:.3f}s (no baseline)")
                
        print("="*70)


def benchmark_cli_startup():
    """Benchmark CLI startup performance."""
    import subprocess
    import sys
    
    benchmark = PerformanceBenchmark()
    
    # Test different CLI operations
    commands = [
        ["python", "-m", "tlc_tools.cli.main", "--help"],
        ["python", "-m", "tlc_tools.cli.main", "list"],
    ]
    
    for cmd in commands:
        cmd_name = " ".join(cmd[-2:])  # Get last 2 parts of command
        
        with timing_context(f"CLI: {cmd_name}", print_result=False) as timing:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd="."
                )
                timing["success"] = result.returncode == 0
                if result.returncode != 0:
                    timing["error"] = result.stderr
            except subprocess.TimeoutExpired:
                timing["success"] = False
                timing["error"] = "Command timed out"
            except Exception as e:
                timing["success"] = False
                timing["error"] = str(e)
                
        benchmark.results.append(timing)
        
    return benchmark


# Baseline performance measurements (before optimization)
BASELINE_PERFORMANCE = {
    "import tlc_tools": 2.1,  # seconds
    "import tlc_tools.split": 1.8,
    "import tlc_tools.metric_jumps": 1.5,
    "CLI: --help": 3.2,
    "CLI: list": 3.8,
}