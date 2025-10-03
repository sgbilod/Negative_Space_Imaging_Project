import threading

class RBAC:
    def __init__(self):
        self.users = {}  # username -> {roles: set, ...}
        self.roles = set()
        self.lock = threading.Lock()
        self.admin_bootstrapped = False

    def create_user(self, username, password, roles=None):
        with self.lock:
            if username in self.users:
                raise ValueError("User already exists")
            # Bootstrap: first user is admin
            if not self.admin_bootstrapped:
                assigned_roles = {"admin"}
                self.admin_bootstrapped = True
            else:
                assigned_roles = set(roles) if roles else set()
            self.users[username] = {
                "password": password,
                "roles": assigned_roles
            }
            self.roles.update(assigned_roles)
            return True

    def authenticate(self, username, password):
        user = self.users.get(username)
        if not user or user["password"] != password:
            return False
        return True

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