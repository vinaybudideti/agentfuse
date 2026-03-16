"""
RequestDeduplicator — coalesces identical in-flight requests.

When multiple threads/tasks make the same LLM request simultaneously,
only ONE actual API call is made. The others wait for the result and
share it. This is called "request coalescing" — used by Cloudflare,
Varnish, and Portkey.

This saves money on duplicate requests that happen when:
- Multiple agent loops hit the same prompt simultaneously
- Retry storms send the same request multiple times
- Concurrent users ask the same question

Usage:
    dedup = RequestDeduplicator()

    # Thread A calls this first
    result = dedup.execute("hash_of_request", lambda: call_llm(...))

    # Thread B calls this while A is in-flight
    result = dedup.execute("hash_of_request", lambda: call_llm(...))
    # Thread B gets Thread A's result WITHOUT making a second API call
"""

import hashlib
import json
import threading
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """
    Coalesces identical in-flight requests to avoid duplicate API calls.

    Thread-safe. Uses per-key locks to allow concurrent requests for
    DIFFERENT keys while serializing requests for the SAME key.
    """

    def __init__(self, max_inflight: int = 10000):
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._results: dict[str, Any] = {}
        self._errors: dict[str, Exception] = {}
        self._max_inflight = max_inflight
        self._dedup_count = 0  # number of deduplicated requests

    def execute(self, request_key: str, call_fn: Callable[[], Any]) -> Any:
        """
        Execute call_fn, deduplicating by request_key.

        If the same key is already in-flight, wait for it and share the result.
        If not, execute call_fn and store the result for others.
        """
        with self._lock:
            if request_key in self._inflight:
                # Another thread is already processing this request
                event = self._inflight[request_key]
                self._dedup_count += 1
                logger.debug("Request deduplicated (key=%s...)", request_key[:16])
            else:
                # We're the first — register and proceed
                event = threading.Event()
                self._inflight[request_key] = event
                event = None  # signal that WE should execute

        if event is not None:
            # Wait for the first thread's result
            event.wait(timeout=120)  # 2 minute max wait

            with self._lock:
                if request_key in self._errors:
                    raise self._errors[request_key]
                if request_key in self._results:
                    return self._results[request_key]

            # Timeout or result already cleaned up — execute ourselves
            return call_fn()

        # We're the executor
        try:
            result = call_fn()
            with self._lock:
                self._results[request_key] = result
            return result
        except Exception as e:
            with self._lock:
                self._errors[request_key] = e
            raise
        finally:
            with self._lock:
                evt = self._inflight.pop(request_key, None)
                if evt:
                    evt.set()  # wake up waiting threads
                # Clean up results after a short delay
                # (waiting threads need to read them first)
                threading.Timer(1.0, self._cleanup, args=(request_key,)).start()

    def _cleanup(self, key: str):
        """Remove stored result/error after waiting threads have read it."""
        with self._lock:
            self._results.pop(key, None)
            self._errors.pop(key, None)

    @staticmethod
    def make_key(model: str, messages: list[dict], temperature: float = 0.0) -> str:
        """Create a dedup key from request parameters."""
        raw = json.dumps({"m": model, "msgs": messages, "t": temperature},
                         sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @property
    def dedup_count(self) -> int:
        """Number of requests deduplicated (not sent to API)."""
        return self._dedup_count
