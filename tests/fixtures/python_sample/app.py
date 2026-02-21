"""Sample Python application for code-agent fixture testing."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class User:
    """Represents a user in the system."""
    name: str
    email: str
    roles: List[str] = field(default_factory=list)

    def has_role(self, role: str) -> bool:
        """Check if the user has a specific role."""
        return role in self.roles

    def add_role(self, role: str) -> None:
        """Add a role to the user."""
        if role not in self.roles:
            self.roles.append(role)


class UserService:
    """Service for managing users."""

    def __init__(self):
        self._users: dict[str, User] = {}

    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(name=name, email=email)
        self._users[email] = user
        return user

    def get_user(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self._users.get(email)

    def list_users(self) -> List[User]:
        """List all users."""
        return list(self._users.values())

    def delete_user(self, email: str) -> bool:
        """Delete a user by email. Returns True if found and deleted."""
        if email in self._users:
            del self._users[email]
            return True
        return False
