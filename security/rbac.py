import threading
from .password_security import hash_password, verify_password

class RBAC:
    def __init__(self):
        self.users = {}  # username -> {password_hash: str, roles: set, ...}
        self.roles = set()
        self.lock = threading.Lock()
        self.admin_bootstrapped = False

    def create_user(self, username, password, roles=None):
        """
        Create a new user with securely hashed password.

        Args:
            username: Unique username
            password: Plaintext password (will be hashed with Argon2id)
            roles: List of roles to assign (first user becomes admin)

        Returns:
            True if user created successfully

        Raises:
            ValueError: If user exists or password requirements not met
        """
        with self.lock:
            if username in self.users:
                raise ValueError("User already exists")

            # Hash password using Argon2id
            try:
                password_hash = hash_password(password)
            except ValueError as e:
                raise ValueError(f"Password security error: {e}")

            # Bootstrap: first user is admin
            if not self.admin_bootstrapped:
                assigned_roles = {"admin"}
                self.admin_bootstrapped = True
            else:
                assigned_roles = set(roles) if roles else set()

            self.users[username] = {
                "password_hash": password_hash,
                "roles": assigned_roles
            }
            self.roles.update(assigned_roles)
            return True

    def authenticate(self, username, password):
        """
        Authenticate a user with plaintext password.

        Args:
            username: Username to authenticate
            password: Plaintext password to verify

        Returns:
            True if authentication succeeds, False otherwise
        """
        user = self.users.get(username)
        if not user:
            return False

        # Verify password against stored hash
        return verify_password(password, user["password_hash"])

    def has_role(self, username, role):
        user = self.users.get(username)
        if not user:
            return False
        return role in user["roles"]

    def assign_role(self, username, role):
        with self.lock:
            if username not in self.users:
                raise ValueError("User not found")
            self.users[username]["roles"].add(role)
            self.roles.add(role)

    def remove_role(self, username, role):
        with self.lock:
            if username not in self.users:
                raise ValueError("User not found")
            self.users[username]["roles"].discard(role)

    def get_user_roles(self, username):
        user = self.users.get(username)
        if not user:
            return set()
        return user["roles"]

    def is_admin(self, username):
        return self.has_role(username, "admin")

# Singleton instance
rbac = RBAC()
