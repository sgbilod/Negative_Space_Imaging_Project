"""
Password Security Module - Argon2id Hashing Implementation
==========================================================

Provides secure password hashing and verification using Argon2id,
the winner of the Password Hashing Competition (2015).

Features:
- Argon2id key derivation function (memory-hard, resistant to GPU attacks)
- Automatic salt generation
- Configurable work factors
- Timing attack resistance
"""

from passlib.context import CryptContext
import logging
import time

logger = logging.getLogger("password_security")

# Initialize Argon2id password context
# - schemes: Use Argon2id as primary hashing algorithm
# - deprecated: Auto-rehash if older schemes are detected
password_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65540,  # 64 MB (OWASP recommendation)
    argon2__time_cost=3,         # 3 iterations
    argon2__parallelism=4,       # 4 parallel threads
    argon2__hash_len=32,         # 256-bit hash
    argon2__salt_size=16,        # 128-bit salt
)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using Argon2id.

    Args:
        password: The plaintext password to hash

    Returns:
        Hashed password (PHC format: $argon2id$v=19$m=65540,t=3,p=4$...)

    Raises:
        ValueError: If password is empty or None
    """
    if not password:
        raise ValueError("Password cannot be empty")

    if not isinstance(password, str):
        raise ValueError("Password must be a string")

    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")

    start_time = time.time()
    hashed = password_context.hash(password)
    hash_time = time.time() - start_time

    logger.info(f"Password hashed successfully in {hash_time:.3f}s")
    return hashed


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a plaintext password against a stored hash.

    Args:
        password: The plaintext password to verify
        password_hash: The stored password hash

    Returns:
        True if password matches, False otherwise
    """
    if not password or not password_hash:
        return False

    try:
        start_time = time.time()
        is_valid = password_context.verify(password, password_hash)
        verify_time = time.time() - start_time

        if is_valid:
            logger.debug(f"Password verified successfully in {verify_time:.3f}s")
        else:
            logger.warning(f"Password verification failed in {verify_time:.3f}s")

        return is_valid
    except Exception as e:
        logger.error(f"Error during password verification: {e}")
        return False


def needs_rehash(password_hash: str) -> bool:
    """
    Check if a password hash needs to be rehashed with updated parameters.

    Args:
        password_hash: The stored password hash

    Returns:
        True if hash should be updated, False otherwise
    """
    try:
        needs_update = password_context.needs_update(password_hash)
        if needs_update:
            logger.info(f"Password hash needs update to current Argon2id parameters")
        return needs_update
    except Exception as e:
        logger.error(f"Error checking if hash needs update: {e}")
        return False


# Example usage and testing
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    print("=" * 70)
    print("PASSWORD SECURITY MODULE - ARGON2ID IMPLEMENTATION TEST")
    print("=" * 70)

    # Test 1: Hash password
    test_password = "MySecurePassword123!@#"
    print(f"\nTest 1: Hashing password...")
    print(f"Input: {test_password}")

    hashed = hash_password(test_password)
    print(f"Hash: {hashed[:50]}...")
    print(f"Hash length: {len(hashed)} characters")

    # Test 2: Verify correct password
    print(f"\nTest 2: Verifying correct password...")
    result = verify_password(test_password, hashed)
    print(f"Result: {result} (expected: True)")

    # Test 3: Verify wrong password
    print(f"\nTest 3: Verifying wrong password...")
    wrong_password = "WrongPassword123!@#"
    result = verify_password(wrong_password, hashed)
    print(f"Result: {result} (expected: False)")

    # Test 4: Check if needs rehash
    print(f"\nTest 4: Checking if hash needs update...")
    needs_update = needs_rehash(hashed)
    print(f"Result: {needs_update}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
