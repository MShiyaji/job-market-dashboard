"""
Configuration settings loaded from environment variables.

Do not commit real secrets. Create a .env file in the project root for local dev.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    openai_api_key: Optional[str] = None

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
