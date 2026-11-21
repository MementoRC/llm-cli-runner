"""Quota and metrics helpers shared by CLI-based providers."""

import itertools
import threading
from datetime import datetime
from typing import Any


class QuotaManager:
    """Simple daily request quota manager."""

    def __init__(self, daily_limit: int = 1000) -> None:
        """Initialize quota manager with daily request limit."""
        self.daily_limit = daily_limit
        self.requests_today = 0
        self.last_reset = datetime.now().date()
        # Threading lock: critical sections never await and the manager may be
        # constructed outside a running event loop.
        self._lock = threading.Lock()
        self._token_ids = itertools.count(1)
        self._outstanding: dict[int, int] = {}

    def check_quota(self) -> bool:
        """Check if quota is available for requests."""
        with self._lock:
            self._reset_if_needed()
            return self.requests_today < self.daily_limit

    def reserve(self, amount: int = 1) -> int | None:
        """Atomically reserve quota; return a unique reservation token or None.

        Each reservation is tracked individually, so releasing the same token
        twice, or releasing after a daily reset (which clears all outstanding
        reservations), never frees another request's slot.

        Raises:
            ValueError: If amount is not positive.

        """
        if amount <= 0:
            raise ValueError(f"amount must be positive, got {amount}")
        with self._lock:
            self._reset_if_needed()
            if self.requests_today + amount > self.daily_limit:
                return None
            self.requests_today += amount
            token = next(self._token_ids)
            self._outstanding[token] = amount
            return token

    def release(self, token: int) -> None:
        """Return an outstanding reservation; unknown or repeated tokens are no-ops."""
        with self._lock:
            self._reset_if_needed()
            amount = self._outstanding.pop(token, None)
            if amount is not None:
                self.requests_today = max(0, self.requests_today - amount)

    def consume_quota(self, amount: int = 1) -> bool:
        """Consume quota and return True if successful."""
        return self.reserve(amount) is not None

    def get_remaining_quota(self) -> int:
        """Get remaining quota for today."""
        with self._lock:
            self._reset_if_needed()
            return max(0, self.daily_limit - self.requests_today)

    def get_current_usage(self) -> int:
        """Get current usage count for today."""
        with self._lock:
            self._reset_if_needed()
            return self.requests_today

    def get_quota_limit(self) -> int:
        """Get the daily quota limit."""
        return self.daily_limit

    def _reset_if_needed(self) -> None:
        """Reset counters on a new day. Caller must hold self._lock."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.requests_today = 0
            self._outstanding.clear()
            self.last_reset = today


class SimpleMetrics:
    """Simple metrics class for compatibility."""

    def __init__(self) -> None:
        """Initialize metrics with zeroed counters."""
        self.request_timestamps = []
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_response_time = 0.0

    def record_request(
        self,
        response_time: float,
        token_count: int,
        cost: float,
        error: str | None = None,
    ) -> None:
        """Record metrics for a completed request."""
        self.request_timestamps.append(datetime.now())
        self.request_count += 1
        self.total_tokens += token_count
        self.total_cost += cost
        self.total_response_time += response_time

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_response_time": (
                self.total_response_time / self.request_count
                if self.request_count > 0
                else 0.0
            ),
            "request_timestamps": [
                ts.isoformat() for ts in self.request_timestamps[-10:]
            ],
        }
