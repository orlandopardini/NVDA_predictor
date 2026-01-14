"""
High-precision timing utilities for performance measurement.

This module provides a context manager for accurate timing of code blocks
using perf_counter for microsecond precision.
"""

from __future__ import annotations
import time
from dataclasses import dataclass


@dataclass
class Stopwatch:
    """
    High-precision stopwatch for timing code execution.
    
    Can be used as a context manager or manually started/stopped.
    Uses time.perf_counter() for maximum precision.
    
    Attributes:
        start (float): Start timestamp
        end (float): End timestamp
        running (bool): Whether stopwatch is currently running
        
    Example:
        >>> with Stopwatch() as sw:
        ...     expensive_operation()
        >>> print(f"Took {sw.elapsed:.3f}s")
        
        >>> sw = Stopwatch()
        >>> sw.start = time.perf_counter()
        >>> # do work
        >>> sw.stop()
        >>> print(sw.elapsed)
    """
    
    start: float = 0.0
    end: float = 0.0
    running: bool = False

    def __enter__(self) -> Stopwatch:
        """Start timing when entering context."""
        self.start = time.perf_counter()
        self.running = True
        return self

    def __exit__(self, exc_type, exc, tb):
        """Stop timing when exiting context."""
        self.stop()

    def stop(self) -> float:
        """
        Stop the stopwatch and return elapsed time.
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.running:
            self.end = time.perf_counter()
            self.running = False
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            float: Time elapsed since start (or start to end if stopped)
        """
        return (time.perf_counter() - self.start) if self.running else (self.end - self.start)
