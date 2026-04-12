"""
User Management Module for Aviation Intelligence Platform.
Stores users in a JSON file with hashed passwords.
Supports persistent session tokens for login across page reloads.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

USERS_FILE = Path(__file__).resolve().parent.parent / "data" / "users.json"
TOKENS_FILE = Path(__file__).resolve().parent.parent / "data" / "sessions.json"
SESSION_EXPIRY_HOURS = 24

ROLES = ["admin", "viewer"]


def _hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> Dict:
    """Load users from JSON file."""
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def _save_users(users: Dict) -> None:
    """Save users to JSON file."""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def initialize_default_users() -> None:
    """Create default admin/viewer users if users.json doesn't exist."""
    if USERS_FILE.exists():
        return
    users = {
        "admin": {
            "password_hash": _hash_password("aviation2024"),
            "role": "admin",
            "created": datetime.now().isoformat(),
            "created_by": "system",
        },
        "viewer": {
            "password_hash": _hash_password("readonly2024"),
            "role": "viewer",
            "created": datetime.now().isoformat(),
            "created_by": "system",
        },
    }
    _save_users(users)


def authenticate(username: str, password: str) -> Optional[str]:
    """
    Authenticate a user. Returns role if valid, None otherwise.
    Falls back to secrets.toml if users.json doesn't exist yet.
    """
    initialize_default_users()
    users = _load_users()
    user = users.get(username)
    if user and user["password_hash"] == _hash_password(password):
        return user["role"]
    return None


def list_users() -> List[Dict]:
    """Return list of all users (without password hashes)."""
    initialize_default_users()
    users = _load_users()
    return [
        {
            "username": uname,
            "role": info["role"],
            "created": info.get("created", "N/A"),
            "created_by": info.get("created_by", "N/A"),
        }
        for uname, info in users.items()
    ]


def add_user(username: str, password: str, role: str, created_by: str) -> tuple[bool, str]:
    """Add a new user. Returns (success, message)."""
    if role not in ROLES:
        return False, f"Invalid role. Must be one of: {ROLES}"
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username in users:
        return False, f"User '{username}' already exists."

    users[username] = {
        "password_hash": _hash_password(password),
        "role": role,
        "created": datetime.now().isoformat(),
        "created_by": created_by,
    }
    _save_users(users)
    return True, f"User '{username}' created with role '{role}'."


def remove_user(username: str) -> tuple[bool, str]:
    """Remove a user. Cannot remove the last admin."""
    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."

    # Prevent removing last admin
    if users[username]["role"] == "admin":
        admin_count = sum(1 for u in users.values() if u["role"] == "admin")
        if admin_count <= 1:
            return False, "Cannot remove the last admin user."

    del users[username]
    _save_users(users)
    return True, f"User '{username}' removed."


def change_password(username: str, new_password: str) -> tuple[bool, str]:
    """Change a user's password."""
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."

    users[username]["password_hash"] = _hash_password(new_password)
    _save_users(users)
    return True, f"Password updated for '{username}'."


def change_role(username: str, new_role: str) -> tuple[bool, str]:
    """Change a user's role."""
    if new_role not in ROLES:
        return False, f"Invalid role. Must be one of: {ROLES}"

    users = _load_users()
    if username not in users:
        return False, f"User '{username}' not found."

    # Prevent demoting last admin
    if users[username]["role"] == "admin" and new_role != "admin":
        admin_count = sum(1 for u in users.values() if u["role"] == "admin")
        if admin_count <= 1:
            return False, "Cannot demote the last admin user."

    users[username]["role"] = new_role
    _save_users(users)
    return True, f"Role for '{username}' changed to '{new_role}'."


# ---------------------------------------------------------------------------
# Session token management (persistent login across page reloads)
# ---------------------------------------------------------------------------
def _load_tokens() -> Dict:
    if not TOKENS_FILE.exists():
        return {}
    with open(TOKENS_FILE, "r") as f:
        return json.load(f)


def _save_tokens(tokens: Dict) -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def generate_token(username: str, role: str) -> str:
    """Generate a session token after successful login."""
    token = secrets.token_hex(32)
    tokens = _load_tokens()
    expiry = (datetime.now() + timedelta(hours=SESSION_EXPIRY_HOURS)).isoformat()
    tokens[token] = {"username": username, "role": role, "expiry": expiry}
    _save_tokens(tokens)
    return token


def validate_token(token: str) -> Optional[Dict]:
    """Validate a session token. Returns {"username", "role"} or None."""
    if not token:
        return None
    tokens = _load_tokens()
    session = tokens.get(token)
    if not session:
        return None
    if datetime.fromisoformat(session["expiry"]) < datetime.now():
        del tokens[token]
        _save_tokens(tokens)
        return None
    return {"username": session["username"], "role": session["role"]}


def revoke_token(token: str) -> None:
    """Revoke a session token (used on logout)."""
    tokens = _load_tokens()
    tokens.pop(token, None)
    _save_tokens(tokens)
