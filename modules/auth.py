"""
FASALSAARTHI - Authentication Module
Simple file-based user management with hashed passwords.
"""

import hashlib
import json
import os
import re
from datetime import datetime

USERS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.json')

# ── Helpers ────────────────────────────────────────────────────────────────────
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _load() -> dict:
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    if not os.path.exists(USERS_FILE):
        # seed demo account
        demo = {
            "demo": {
                "password": _hash("demo123"),
                "name": "Demo Farmer",
                "state": "Maharashtra",
                "district": "Pune",
                "joined": datetime.now().strftime("%Y-%m-%d"),
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(demo, f, indent=2)
    with open(USERS_FILE) as f:
        return json.load(f)

def _save(users: dict):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> tuple[bool, str, dict]:
    """Returns (success, message, user_info)."""
    users = _load()
    u = username.strip().lower()
    if u not in users:
        return False, "❌ Username not found.", {}
    if users[u]['password'] != _hash(password):
        return False, "❌ Incorrect password.", {}
    return True, "✅ Login successful!", users[u]


def register(username: str, password: str, name: str,
             state: str = "Maharashtra", district: str = "Pune") -> tuple[bool, str]:
    """Returns (success, message)."""
    users = _load()
    u = username.strip().lower()

    if len(u) < 3:
        return False, "Username must be at least 3 characters."
    if not re.match(r'^[a-z0-9_]+$', u):
        return False, "Username can only contain letters, numbers and underscore."
    if u in users:
        return False, "❌ Username already exists. Please choose another."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users[u] = {
        "password": _hash(password),
        "name": name.strip(),
        "state": state,
        "district": district,
        "joined": datetime.now().strftime("%Y-%m-%d"),
    }
    _save(users)
    return True, "✅ Account created! You can now log in."


def get_user(username: str) -> dict:
    users = _load()
    return users.get(username.strip().lower(), {})
