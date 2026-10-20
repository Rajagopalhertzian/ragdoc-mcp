"""
Example Python module demonstrating various language features.

This module includes:
- Classes with type hints
- Functions with decorators
- Async functions
- Context managers
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime


@dataclass
class User:
    """User data model."""
    id: int
    name: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            email=data["email"],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


class UserRepository:
    """Repository for user operations."""
    
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1
    
    def create(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(id=self._next_id, name=name, email=email)
        self._users[self._next_id] = user
        self._next_id += 1
        return user
    
    def get(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def list_all(self) -> List[User]:
        """List all users."""
        return list(self._users.values())
    
    def update(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user fields."""
        user = self._users.get(user_id)
        if user is None:
            return None
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        return user
    
    def delete(self, user_id: int) -> bool:
        """Delete a user."""
        if user_id in self._users:
            del self._users[user_id]
            return True
        return False


async def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """Simulate async data fetching."""
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}


@asynccontextmanager
async def transaction():
    """Simulate database transaction context manager."""
    print("Transaction started")
    try:
        yield
        print("Transaction committed")
    except Exception:
        print("Transaction rolled back")
        raise


async def process_users(user_ids: List[int]) -> List[Dict[str, Any]]:
    """Process multiple users concurrently."""
    tasks = [fetch_user_data(uid) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results


def main():
    """Main entry point."""
    repo = UserRepository()
    
    # Create users
    alice = repo.create("Alice", "alice@example.com")
    bob = repo.create("Bob", "bob@example.com")
    
    # List users
    print("Users:", [u.name for u in repo.list_all()])
    
    # Async operations
    results = asyncio.run(process_users([1, 2, 3]))
    print("Async results:", results)
    
    # Transaction
    asyncio.run(run_with_transaction(repo))
    
    print("Done!")


async def run_with_transaction(repo: UserRepository):
    """Run operations in a transaction."""
    async with transaction():
        user = repo.create("Charlie", "charlie@example.com")
        repo.update(user.id, name="Charlie Updated")


if __name__ == "__main__":
    main()