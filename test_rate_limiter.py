"""
Comprehensive Test Suite for Sliding Window Rate Limiter

This module provides extensive testing for the SlidingWindowRateLimiter class,
covering all functionality, edge cases, and performance characteristics.

Author: ELITE AGENT COLLECTIVE - @APEX
Date: November 30, 2025
"""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock
import pytest

from rate_limiter import (
    SlidingWindowRateLimiter,
    DistributedSlidingWindowRateLimiter,
    create_api_rate_limiter,
    create_user_rate_limiter
)


class TestSlidingWindowRateLimiter(unittest.TestCase):
    """Comprehensive test suite for SlidingWindowRateLimiter."""

    def setUp(self):
        """Set up test fixtures."""
        self.limiter = SlidingWindowRateLimiter(window_size=5.0, max_requests=3)

    def tearDown(self):
        """Clean up after tests."""
        # Stop cleanup thread
        if hasattr(self.limiter, '_stop_cleanup_thread'):
            self.limiter._stop_cleanup_thread()

    def test_initialization(self):
        """Test proper initialization of rate limiter."""
        # Valid initialization
        limiter = SlidingWindowRateLimiter(window_size=10.0, max_requests=5)
        self.assertEqual(limiter.window_size, 10.0)
        self.assertEqual(limiter.max_requests, 5)

        # Invalid window_size
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(window_size=0, max_requests=5)
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(window_size=-1, max_requests=5)

        # Invalid max_requests
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(window_size=10.0, max_requests=0)
        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(window_size=10.0, max_requests=-1)

    def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        key = "test_key"

        # First 3 requests should be allowed
        for i in range(3):
            self.assertTrue(self.limiter.allow_request(key),
                          f"Request {i+1} should be allowed")

        # 4th request should be denied
        self.assertFalse(self.limiter.allow_request(key),
                        "4th request should be denied")

    def test_window_sliding(self):
        """Test that the window slides correctly over time."""
        key = "test_key"

        # Fill the window
        for i in range(3):
            self.assertTrue(self.limiter.allow_request(key))

        # Wait for window to slide (more than 5 seconds)
        time.sleep(6)

        # Should be able to make requests again
        self.assertTrue(self.limiter.allow_request(key),
                       "Request should be allowed after window slides")

    def test_multiple_keys(self):
        """Test rate limiting with multiple independent keys."""
        key1 = "user1"
        key2 = "user2"

        # Both users should be able to make max requests
        for i in range(3):
            self.assertTrue(self.limiter.allow_request(key1))
            self.assertTrue(self.limiter.allow_request(key2))

        # Both should be rate limited
        self.assertFalse(self.limiter.allow_request(key1))
        self.assertFalse(self.limiter.allow_request(key2))

    def test_get_remaining_requests(self):
        """Test getting remaining requests count."""
        key = "test_key"

        # Initially should have max requests remaining
        self.assertEqual(self.limiter.get_remaining_requests(key), 3)

        # Make some requests
        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_remaining_requests(key), 2)

        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_remaining_requests(key), 1)

        # Fill the window
        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_remaining_requests(key), 0)

        # Try to exceed limit
        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_remaining_requests(key), 0)

    def test_get_request_count(self):
        """Test getting current request count."""
        key = "test_key"

        self.assertEqual(self.limiter.get_request_count(key), 0)

        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_request_count(key), 1)

        self.limiter.allow_request(key)
        self.assertEqual(self.limiter.get_request_count(key), 2)

    def test_reset_key(self):
        """Test resetting rate limit for a specific key."""
        key = "test_key"

        # Fill the window
        for i in range(3):
            self.limiter.allow_request(key)

        self.assertFalse(self.limiter.allow_request(key))

        # Reset the key
        self.limiter.reset_key(key)

        # Should be able to make requests again
        self.assertTrue(self.limiter.allow_request(key))

    def test_get_stats(self):
        """Test getting statistics."""
        # Initially empty
        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_keys"], 0)
        self.assertEqual(stats["total_requests"], 0)

        # Add some requests
        self.limiter.allow_request("key1")
        self.limiter.allow_request("key2")
        self.limiter.allow_request("key1")

        stats = self.limiter.get_stats()
        self.assertEqual(stats["total_keys"], 2)
        self.assertEqual(stats["total_requests"], 3)

    def test_thread_safety(self):
        """Test thread safety of the rate limiter."""
        results = []
        errors = []

        def worker(key, num_requests):
            try:
                count = 0
                for i in range(num_requests):
                    if self.limiter.allow_request(key):
                        count += 1
                    time.sleep(0.01)  # Small delay to allow interleaving
                results.append((key, count))
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            key = f"thread_{i}"
            t = threading.Thread(target=worker, args=(key, 10))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check results
        self.assertEqual(len(errors), 0, f"Thread errors: {errors}")

        # Each thread should have been limited to max_requests
        for key, count in results:
            self.assertLessEqual(count, self.limiter.max_requests,
                               f"Thread {key} exceeded limit: {count}")

    def test_cleanup_expired_entries(self):
        """Test cleanup of expired entries."""
        key = "test_key"

        # Make some requests
        for i in range(3):
            self.limiter.allow_request(key)

        # Verify requests are tracked
        self.assertEqual(self.limiter.get_request_count(key), 3)

        # Wait for window to expire
        time.sleep(6)

        # Manually trigger cleanup (normally done by background thread)
        self.limiter._cleanup_expired_entries()

        # Should have no requests left
        self.assertEqual(self.limiter.get_request_count(key), 0)

    def test_edge_case_empty_key(self):
        """Test behavior with empty string key."""
        self.assertTrue(self.limiter.allow_request(""))
        self.assertTrue(self.limiter.allow_request(""))
        self.assertTrue(self.limiter.allow_request(""))

        self.assertFalse(self.limiter.allow_request(""))


class TestDistributedSlidingWindowRateLimiter(unittest.TestCase):
    """Test suite for distributed rate limiter."""

    def test_fallback_to_memory_when_redis_unavailable(self):
        """Test fallback to in-memory when Redis is not available."""
        limiter = DistributedSlidingWindowRateLimiter(
            window_size=5.0,
            max_requests=3,
            redis_host="nonexistent"
        )

        # Should work like regular limiter
        key = "test_key"
        for i in range(3):
            self.assertTrue(limiter.allow_request(key))

        self.assertFalse(limiter.allow_request(key))

    @patch('redis.Redis')
    def test_redis_integration(self, mock_redis_class):
        """Test Redis integration when available."""
        mock_redis = MagicMock()
        mock_redis_class.return_value = mock_redis

        # Mock Redis operations
        mock_redis.zremrangebyscore.return_value = 0
        mock_redis.zcard.return_value = 1  # 1 request already
        mock_redis.zadd.return_value = 1
        mock_redis.expire.return_value = True

        limiter = DistributedSlidingWindowRateLimiter(
            window_size=5.0,
            max_requests=3
        )

        # Should allow request (1 < 3)
        result = limiter.allow_request("test_key")
        self.assertTrue(result)

        # Verify Redis calls
        mock_redis.zremrangebyscore.assert_called()
        mock_redis.zcard.assert_called()
        mock_redis.zadd.assert_called()
        mock_redis.expire.assert_called()


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for creating rate limiters."""

    def test_create_api_rate_limiter(self):
        """Test API rate limiter creation."""
        limiter = create_api_rate_limiter(max_requests=50, window_seconds=30)

        self.assertEqual(limiter.window_size, 30.0)
        self.assertEqual(limiter.max_requests, 50)

    def test_create_user_rate_limiter(self):
        """Test user rate limiter creation."""
        limiter = create_user_rate_limiter(max_requests=5, window_seconds=120)

        self.assertEqual(limiter.window_size, 120.0)
        self.assertEqual(limiter.max_requests, 5)


class TestPerformance(unittest.TestCase):
    """Performance tests for the rate limiter."""

    def test_high_concurrency_performance(self):
        """Test performance under high concurrency."""
        limiter = SlidingWindowRateLimiter(window_size=60.0, max_requests=1000)

        start_time = time.time()

        # Simulate high load
        for i in range(100):
            key = f"user_{i % 10}"  # 10 different users
            for j in range(10):
                limiter.allow_request(key)

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (< 1 second)
        self.assertLess(duration, 1.0, f"Performance test took {duration:.2f}s")

    def test_memory_usage(self):
        """Test memory usage with many keys."""
        limiter = SlidingWindowRateLimiter(window_size=300.0, max_requests=10)

        # Create many keys
        for i in range(1000):
            key = f"user_{i}"
            limiter.allow_request(key)

        stats = limiter.get_stats()
        self.assertEqual(stats["total_keys"], 1000)
        self.assertEqual(stats["total_requests"], 1000)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    unittest.main(verbosity=2)
