import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Allows bursts up to `capacity`, refilling at `rate` tokens per second."""

    capacity: float
    rate: float
    tokens: float = field(init=False)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.tokens = self.capacity

    def try_acquire(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.rate)
        self.updated_at = now
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True


buckets: dict[str, TokenBucket] = {}


def allow(client_id: str) -> bool:
    bucket = buckets.setdefault(client_id, TokenBucket(capacity=20, rate=5))
    return bucket.try_acquire()
