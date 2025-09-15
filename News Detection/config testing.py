# backend/config.py
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data files
USERS_FILE = BASE_DIR / "users.txt"
SEARCH_FILE = BASE_DIR / "search.txt"

# Ensure data directory exists
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
