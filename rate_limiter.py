"""
Sliding Window Rate Limiter Implementation

This module provides a production-grade sliding window rate limiter for controlling
request rates in distributed systems. The sliding window algorithm provides more
accurate rate limiting compared to fixed windows by allowing requests that fall
within a moving time window.

Author: ELITE AGENT COLLECTIVE - @APEX
Date: November 30, 2025
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
import threading
import logging

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    A thread-safe sliding window rate limiter that tracks requests within a moving time window.

    This implementation uses a sliding window approach where requests are allowed if the
    number of requests within the current window does not exceed the maximum allowed.

    Time Complexity:
    - allow_request(): O(n) where n is the number of requests in the current window
    - cleanup(): O(n) for periodic cleanup of expired entries

    Space Complexity: O(m * w) where m is number of unique keys and w is window size in requests

    Attributes:
        window_size (float): Size of the sliding window in seconds
        max_requests (int): Maximum number of requests allowed within the window
        cleanup_interval (float): Interval for periodic cleanup in seconds
    """

    def __init__(
        self,
        window_size: float,
        max_requests: int,
        cleanup_interval: float = 60.0
    ):
        """
        Initialize the sliding window rate limiter.

        Args:
            window_size: Size of the sliding window in seconds (e.g., 60.0 for 1 minute)
            max_requests: Maximum number of requests allowed within the window
            cleanup_interval: Interval for periodic cleanup of expired entries (seconds)

        Raises:
            ValueError: If window_size <= 0 or max_requests <= 0
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")

        self.window_size = window_size
        self.max_requests = max_requests
        self.cleanup_interval = cleanup_interval

        # Thread-safe storage: key -> deque of timestamps (sorted)
        self._requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()

        # Start cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

        logger.info(
            f"Initialized SlidingWindowRateLimiter: window={window_size}s, "
            f"max_requests={max_requests}, cleanup_interval={cleanup_interval}s"
        )

    def __del__(self):
        """Cleanup resources on deletion."""
        self._stop_cleanup_thread()

    def allow_request(self, key: str) -> bool:
        """
        Check if a request should be allowed for the given key.

        This method is thread-safe and implements the core sliding window logic:
        1. Remove timestamps outside the current window
        2. Check if current request count is below the limit
        3. Add current timestamp if allowed

        Args:
            key: Identifier for the rate limit (e.g., user ID, IP address)

        Returns:
            bool: True if request is allowed, False if rate limit exceeded

        Example:
            >>> limiter = SlidingWindowRateLimiter(window_size=60.0, max_requests=10)
            >>> limiter.allow_request("user123")  # Returns True
            >>> # Make 10 requests quickly
            >>> all(limiter.allow_request("user123") for _ in range(10))  # Returns True
            >>> limiter.allow_request("user123")  # Returns False (rate limited)
        """
        current_time = time.time()

        with self._lock:
            timestamps = self._requests[key]

            # Remove timestamps outside the current window
            while timestamps and current_time - timestamps[0] > self.window_size:
                timestamps.popleft()

            # Check if under the limit
            if len(timestamps) < self.max_requests:
                timestamps.append(current_time)
                logger.debug(f"Request allowed for key '{key}': {len(timestamps)}/{self.max_requests}")
                return True
            else:
                logger.debug(f"Request denied for key '{key}': {len(timestamps)}/{self.max_requests} (rate limited)")
                return False

    def get_remaining_requests(self, key: str) -> int:
        """
        Get the number of remaining requests allowed for the given key within the current window.

        Args:
            key: Identifier for the rate limit

        Returns:
            int: Number of remaining requests (can be negative if over limit)
        """
        current_time = time.time()

        with self._lock:
            timestamps = self._requests[key]

            # Remove expired timestamps
            while timestamps and current_time - timestamps[0] > self.window_size:
                timestamps.popleft()

            remaining = self.max_requests - len(timestamps)
            return max(0, remaining)  # Don't return negative values

    def get_request_count(self, key: str) -> int:
        """
        Get the current number of requests for the given key within the sliding window.

        Args:
            key: Identifier for the rate limit

        Returns:
            int: Current number of requests in the window
        """
        current_time = time.time()

        with self._lock:
            timestamps = self._requests[key]

            # Remove expired timestamps
            while timestamps and current_time - timestamps[0] > self.window_size:
                timestamps.popleft()

            return len(timestamps)

    def reset_key(self, key: str) -> None:
        """
        Reset the rate limit for a specific key (removes all timestamps).

        Args:
            key: Identifier for the rate limit to reset
        """
        with self._lock:
            if key in self._requests:
                del self._requests[key]
                logger.info(f"Reset rate limit for key '{key}'")

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the rate limiter.

        Returns:
            Dict containing:
            - total_keys: Number of unique keys being tracked
            - total_requests: Total number of requests across all keys
        """
        with self._lock:
            total_keys = len(self._requests)
            total_requests = sum(len(timestamps) for timestamps in self._requests.values())
            return {
                "total_keys": total_keys,
                "total_requests": total_requests
            }

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        if self._cleanup_thread is not None:
            return

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            daemon=True,
            name="RateLimiterCleanup"
        )
        self._cleanup_thread.start()

    def _stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread."""
        if not hasattr(self, '_cleanup_thread') or self._cleanup_thread is None:
            return

        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=1.0)
        self._cleanup_thread = None

    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup of expired entries."""
        while not self._stop_cleanup.wait(self.cleanup_interval):
            try:
                self._cleanup_expired_entries()
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")

    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from all keys."""
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, timestamps in self._requests.items():
                # Remove expired timestamps
                while timestamps and current_time - timestamps[0] > self.window_size:
                    timestamps.popleft()

                # Mark key for removal if no timestamps remain
                if not timestamps:
                    expired_keys.append(key)

            # Remove empty keys
            for key in expired_keys:
                del self._requests[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired keys")


class DistributedSlidingWindowRateLimiter(SlidingWindowRateLimiter):
    """
    Distributed version of the sliding window rate limiter using Redis.

    This implementation extends the base rate limiter to work across multiple
    instances by storing request data in Redis. Requires redis-py library.

    Note: This is a placeholder implementation. In production, you would need:
    - Redis server setup
    - Proper error handling for Redis failures
    - Redis clustering for high availability
    """

    def __init__(
        self,
        window_size: float,
        max_requests: int,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        """
        Initialize the distributed rate limiter.

        Args:
            window_size: Size of the sliding window in seconds
            max_requests: Maximum number of requests allowed within the window
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
        """
        super().__init__(window_size, max_requests, cleanup_interval=30.0)

        try:
            import redis
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True
            )
            self.redis_available = True
            logger.info("Connected to Redis for distributed rate limiting")
        except ImportError:
            logger.warning("redis-py not available, falling back to in-memory implementation")
            self.redis_available = False
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_available = False

    def allow_request(self, key: str) -> bool:
        """
        Check if a request should be allowed using Redis for distributed storage.

        Falls back to in-memory implementation if Redis is unavailable.
        """
        if not self.redis_available:
            return super().allow_request(key)

        current_time = time.time()
        window_start = current_time - self.window_size

        try:
            # Use Redis sorted set to store timestamps
            redis_key = f"ratelimit:{key}"

            # Remove expired timestamps
            self.redis_client.zremrangebyscore(redis_key, 0, window_start)

            # Get current count
            count = self.redis_client.zcard(redis_key)

            if count < self.max_requests:
                # Add current timestamp
                self.redis_client.zadd(redis_key, {str(current_time): current_time})
                # Set expiration on the key (window_size + buffer)
                self.redis_client.expire(redis_key, int(self.window_size) + 60)
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Redis error in allow_request: {e}")
            # Mark Redis as unavailable for future calls
            self.redis_available = False
            # Fallback to in-memory implementation
            current_time = time.time()

            with self._lock:
                timestamps = self._requests[key]

                # Remove timestamps outside the current window
                while timestamps and current_time - timestamps[0] > self.window_size:
                    timestamps.popleft()

                # Check if under the limit
                if len(timestamps) < self.max_requests:
                    timestamps.append(current_time)
                    logger.debug(f"Request allowed for key '{key}': {len(timestamps)}/{self.max_requests}")
                    return True
                else:
                    logger.debug(f"Request denied for key '{key}': {len(timestamps)}/{self.max_requests} (rate limited)")
                    return False


# Convenience functions for common use cases
def create_api_rate_limiter(max_requests: int = 100, window_seconds: int = 60) -> SlidingWindowRateLimiter:
    """
    Create a rate limiter suitable for API endpoints.

    Args:
        max_requests: Maximum requests per window (default: 100)
        window_seconds: Window size in seconds (default: 60)

    Returns:
        Configured SlidingWindowRateLimiter instance
    """
    return SlidingWindowRateLimiter(
        window_size=window_seconds,
        max_requests=max_requests,
        cleanup_interval=300.0  # 5 minutes
    )


def create_user_rate_limiter(max_requests: int = 10, window_seconds: int = 60) -> SlidingWindowRateLimiter:
    """
    Create a rate limiter suitable for user actions (stricter limits).

    Args:
        max_requests: Maximum requests per window (default: 10)
        window_seconds: Window size in seconds (default: 60)

    Returns:
        Configured SlidingWindowRateLimiter instance
    """
    return SlidingWindowRateLimiter(
        window_size=window_seconds,
        max_requests=max_requests,
        cleanup_interval=300.0
    )


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    limiter = SlidingWindowRateLimiter(window_size=10.0, max_requests=5)

    print("Testing Sliding Window Rate Limiter")
    print("=" * 40)

    # Test basic functionality
    key = "test_user"
    for i in range(7):
        allowed = limiter.allow_request(key)
        print(f"Request {i+1}: {'ALLOWED' if allowed else 'DENIED'}")
        time.sleep(1)  # Wait 1 second between requests

    print(f"\nStats: {limiter.get_stats()}")
    print(f"Remaining requests: {limiter.get_remaining_requests(key)}")

    # Wait for window to slide
    print("\nWaiting 6 seconds for window to slide...")
    time.sleep(6)

    # Test again
    for i in range(3):
        allowed = limiter.allow_request(key)
        print(f"Request after wait {i+1}: {'ALLOWED' if allowed else 'DENIED'}")
        time.sleep(1)
