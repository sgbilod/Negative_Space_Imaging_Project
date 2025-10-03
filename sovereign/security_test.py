#!/usr/bin/env python3
"""
Security Integration Test Script
© 2025 Negative Space Imaging, Inc. - CONFIDENTIAL

This script tests the integration of the security system with the Sovereign Control System.
It verifies:
1. User authentication
2. Access control
3. Encryption and decryption
4. Session management
5. Security middleware
"""

import os
import sys
import json
import time
import random
import string
import hashlib
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('security_test.log')
    ]
)
logger = logging.getLogger('security_test')

# Test configuration
TEST_SERVER = "http://localhost:5000"
DEFAULT_ADMIN = {
    "username": "admin",
    "password": "sovereign_admin_2025"
}
TEST_USER = {
    "username": f"test_user_{random.randint(1000, 9999)}",
    "password": ''.join(random.choices(string.ascii_letters + string.digits, k=12))
}

# Session for maintaining cookies
session = requests.Session()


def run_test_suite():
    """Run the full test suite"""
    test_results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "tests": []
    }

    try:
        # Test authentication
        test_results = run_test(test_authentication, "Authentication", test_results)

        # Test authorization
        test_results = run_test(test_authorization, "Authorization", test_results)

        # Test encryption
        test_results = run_test(test_encryption, "Encryption", test_results)

        # Test session management
        test_results = run_test(test_session_management, "Session Management", test_results)

        # Test security middleware
        test_results = run_test(test_security_middleware, "Security Middleware", test_results)

        # Test route protection
        test_results = run_test(test_route_protection, "Route Protection", test_results)

        # Cleanup
        logout()

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")

    # Print results
    print("\n=== TEST RESULTS ===")
    print(f"Total: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    print("===================")

    for test in test_results['tests']:
        status = "✅" if test['passed'] else "❌"
        print(f"{status} {test['name']}: {test['message']}")

    return test_results


def run_test(test_func, test_name, results):
    """Run a single test and record results"""
    logger.info(f"Running test: {test_name}")
    results['total'] += 1

    try:
        passed, message = test_func()
        if passed:
            results['passed'] += 1
            logger.info(f"Test passed: {test_name}")
        else:
            results['failed'] += 1
            logger.error(f"Test failed: {test_name} - {message}")

        results['tests'].append({
            'name': test_name,
            'passed': passed,
            'message': message
        })
    except Exception as e:
        results['failed'] += 1
        message = f"Exception: {str(e)}"
        logger.error(f"Test error: {test_name} - {message}")
        results['tests'].append({
            'name': test_name,
            'passed': False,
            'message': message
        })

    return results


def login(username=None, password=None):
    """Log in to the system"""
    if username is None:
        username = DEFAULT_ADMIN['username']
    if password is None:
        password = DEFAULT_ADMIN['password']

    response = session.post(
        f"{TEST_SERVER}/security/login",
        data={
            'username': username,
            'password': password
        },
        allow_redirects=False
    )

    return response.status_code == 302  # Redirects on success


def logout():
    """Log out of the system"""
    response = session.get(f"{TEST_SERVER}/security/logout", allow_redirects=False)
    return response.status_code == 302  # Redirects on success


def test_authentication():
    """Test user authentication"""
    # Test valid login
    if not login():
        return False, "Failed to log in with valid credentials"

    # Check that we can access a protected route
    response = session.get(f"{TEST_SERVER}/")
    if response.status_code != 200:
        return False, f"Failed to access protected route after login (status: {response.status_code})"

    # Logout
    if not logout():
        return False, "Failed to log out"

    # Check that protected route redirects to login
    response = session.get(f"{TEST_SERVER}/", allow_redirects=False)
    if response.status_code != 302:
        return False, f"Protected route didn't redirect to login after logout (status: {response.status_code})"

    # Test invalid login
    if login("invalid_user", "invalid_password"):
        return False, "Invalid login succeeded when it should have failed"

    return True, "Authentication tests passed"


def test_authorization():
    """Test user authorization"""
    # Login as admin
    if not login():
        return False, "Failed to log in as admin"

    # Access admin area (security dashboard)
    response = session.get(f"{TEST_SERVER}/security/dashboard")
    if response.status_code != 200:
        return False, f"Admin couldn't access security dashboard (status: {response.status_code})"

    # Create a test user
    response = session.post(
        f"{TEST_SERVER}/security/users/add",
        data={
            'username': TEST_USER['username'],
            'password': TEST_USER['password'],
            'role': 'user'
        }
    )

    if response.status_code != 302:
        return False, f"Failed to create test user (status: {response.status_code})"

    # Logout and login as test user
    logout()
    if not login(TEST_USER['username'], TEST_USER['password']):
        return False, "Failed to log in as test user"

    # Try to access admin area
    response = session.get(f"{TEST_SERVER}/security/dashboard")
    if response.status_code == 200:
        # Check if there's a message saying access denied
        if "You do not have permission" not in response.text:
            return False, "Regular user accessed admin area when they shouldn't be able to"

    # Logout
    logout()

    return True, "Authorization tests passed"


def test_encryption():
    """Test data encryption"""
    # Login as admin
    if not login():
        return False, "Failed to log in as admin"

    # Get security settings
    response = session.get(f"{TEST_SERVER}/security/settings")
    if response.status_code != 200:
        return False, f"Failed to access security settings (status: {response.status_code})"

    # Set security level to ENHANCED
    response = session.post(
        f"{TEST_SERVER}/security/settings/update",
        data={
            'security_level': 'ENHANCED',
            'enable_audit_logging': 'on',
            'session_timeout': '30'
        }
    )

    if response.status_code != 302:
        return False, f"Failed to update security settings (status: {response.status_code})"

    # Create a test file with sensitive data
    test_data = {
        'secret': 'This is confidential information',
        'timestamp': time.time()
    }

    response = session.post(
        f"{TEST_SERVER}/save",
        data={
            'filename': 'encryption_test'
        },
        json=test_data
    )

    if response.status_code != 200:
        return False, f"Failed to save test data (status: {response.status_code})"

    # Verify the file exists
    response = session.get(f"{TEST_SERVER}/status")
    if response.status_code != 200:
        return False, f"Failed to get system status (status: {response.status_code})"

    # Logout
    logout()

    return True, "Encryption tests passed"


def test_session_management():
    """Test session management"""
    # Login
    if not login():
        return False, "Failed to log in"

    # Get security settings
    response = session.get(f"{TEST_SERVER}/security/settings")
    if response.status_code != 200:
        return False, f"Failed to access security settings (status: {response.status_code})"

    # Set short session timeout (5 seconds)
    response = session.post(
        f"{TEST_SERVER}/security/settings/update",
        data={
            'security_level': 'STANDARD',
            'enable_audit_logging': 'on',
            'session_timeout': '0.08'  # 5 seconds in minutes
        }
    )

    if response.status_code != 302:
        return False, f"Failed to update session timeout (status: {response.status_code})"

    # Wait for session to expire
    logger.info("Waiting for session to expire...")
    time.sleep(6)

    # Try to access protected route
    response = session.get(f"{TEST_SERVER}/", allow_redirects=False)
    if response.status_code != 302:
        return False, f"Session didn't expire as expected (status: {response.status_code})"

    # Revert the session timeout
    login()
    response = session.post(
        f"{TEST_SERVER}/security/settings/update",
        data={
            'security_level': 'STANDARD',
            'enable_audit_logging': 'on',
            'session_timeout': '30'  # 30 minutes
        }
    )

    # Logout
    logout()

    return True, "Session management tests passed"


def test_security_middleware():
    """Test security middleware"""
    # Login
    if not login():
        return False, "Failed to log in"

    # Test CSRF protection
    # This should fail because we're not including the CSRF token
    cookies = session.cookies.get_dict()
    headers = {'X-Requested-With': 'XMLHttpRequest'}

    # Create a new session without the CSRF token
    bad_session = requests.Session()
    for cookie, value in cookies.items():
        if cookie != 'csrf_token':
            bad_session.cookies.set(cookie, value)

    response = bad_session.post(
        f"{TEST_SERVER}/execute",
        data={'directive': 'STATUS'},
        headers=headers
    )

    # Should fail with 400 or 403
    if response.status_code < 400:
        return False, f"CSRF protection not working properly (status: {response.status_code})"

    # Test XSS protection
    # Try sending a script tag in a parameter
    response = session.get(
        f"{TEST_SERVER}/directives/<script>alert('XSS')</script>",
        allow_redirects=False
    )

    if response.status_code != 302:
        return False, f"XSS protection might not be working (status: {response.status_code})"

    # Logout
    logout()

    return True, "Security middleware tests passed"


def test_route_protection():
    """Test that all routes are protected"""
    # Define routes to test
    routes = [
        "/",
        "/directives",
        "/optimization",
        "/configuration",
        "/status",
    ]

    # Without login, all should redirect to login
    for route in routes:
        response = session.get(f"{TEST_SERVER}{route}", allow_redirects=False)
        if response.status_code != 302 or 'security/login' not in response.headers.get('Location', ''):
            return False, f"Route {route} is not properly protected (status: {response.status_code})"

    # Login and all should be accessible
    if not login():
        return False, "Failed to log in"

    for route in routes:
        response = session.get(f"{TEST_SERVER}{route}")
        if response.status_code != 200:
            return False, f"Route {route} is not accessible after login (status: {response.status_code})"

    # Logout
    logout()

    return True, "Route protection tests passed"


if __name__ == "__main__":
    print("=== SOVEREIGN SECURITY INTEGRATION TEST ===")
    print("This test suite verifies that the security system")
    print("is properly integrated with the Sovereign Control System.")
    print("=============================================")

    try:
        # Test server connectivity
        response = requests.get(f"{TEST_SERVER}/security/login")
        if response.status_code != 200:
            print(f"ERROR: Cannot connect to test server at {TEST_SERVER}")
            print(f"Status code: {response.status_code}")
            print("Please make sure the server is running.")
            sys.exit(1)
    except requests.RequestException as e:
        print(f"ERROR: Cannot connect to test server at {TEST_SERVER}")
        print(f"Exception: {str(e)}")
        print("Please make sure the server is running.")
        sys.exit(1)

    # Run the tests
    results = run_test_suite()

    # Exit with appropriate code
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
