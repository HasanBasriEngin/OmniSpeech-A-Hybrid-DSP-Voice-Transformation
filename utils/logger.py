from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionLogger:
    entries: list[dict] = field(default_factory=list)

    def add(self, event: str, **payload) -> None:
        self.entries.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event": event,
                "payload": payload,
            }
        )

