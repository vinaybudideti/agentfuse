"""
AsyncSpendRecorder — non-blocking cost recording via asyncio.Queue.

Based on research file 4 (Block 12): "asyncio.Queue as the primary path.
For an SDK embedded in user apps, zero additional dependencies is critical.
Cost events are fire-and-forget telemetry."

This decouples cost recording from the LLM hot path:
- record() is non-blocking (puts event in queue, never waits)
- Background task batches events and flushes to SpendLedger
- Drops events if queue is full (never blocks the caller)

Usage:
    recorder = AsyncSpendRecorder()
    await recorder.start()

    # In the hot path (non-blocking):
    recorder.record(model="gpt-4o", cost_usd=0.05, ...)

    # Background task flushes to SpendLedger every 5 seconds
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpendEvent:
    """A single cost event for async recording."""
    timestamp: float = field(default_factory=time.time)
    model: str = ""
    provider: str = ""
    run_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    metadata: dict = field(default_factory=dict)


class AsyncSpendRecorder:
    """
    Non-blocking cost recorder using asyncio.Queue.

    Events are queued instantly and flushed in batches to SpendLedger
    by a background asyncio.Task. Never blocks the LLM call path.
    """

    def __init__(
        self,
        flush_interval: float = 5.0,
        max_batch_size: int = 10_000,
        max_queue_size: int = 100_000,
        ledger_path: Optional[str] = None,
    ):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._flush_interval = flush_interval
        self._max_batch_size = max_batch_size
        self._ledger_path = ledger_path
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._total_recorded = 0
        self._total_dropped = 0
        self._total_flushed = 0

    def record(self, **kwargs) -> bool:
        """Non-blocking enqueue. Returns True if queued, False if dropped."""
        event = SpendEvent(**kwargs)
        try:
            self._queue.put_nowait(event)
            self._total_recorded += 1
            return True
        except asyncio.QueueFull:
            self._total_dropped += 1
            return False

    async def start(self):
        """Start the background flush task."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("AsyncSpendRecorder started (flush every %.1fs)", self._flush_interval)

    async def stop(self):
        """Stop the recorder and flush remaining events."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_batch(drain=True)

    async def _flush_loop(self):
        """Background loop: batch events and flush periodically."""
        while self._running:
            batch = []
            deadline = time.monotonic() + self._flush_interval

            while len(batch) < self._max_batch_size:
                timeout = max(0.01, deadline - time.monotonic())
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch.append(event)
                except asyncio.TimeoutError:
                    break

            if batch:
                await self._write_batch(batch)

    async def _flush_batch(self, drain: bool = False):
        """Flush all queued events."""
        batch = []
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if batch:
            await self._write_batch(batch)

    async def _write_batch(self, batch: list[SpendEvent]):
        """Write a batch of events to SpendLedger."""
        try:
            from agentfuse.storage.spend_ledger import SpendLedger
            ledger = SpendLedger(self._ledger_path or "~/.agentfuse/spend.jsonl")
            for event in batch:
                ledger.record(
                    run_id=event.run_id,
                    model=event.model,
                    cost_usd=event.cost_usd,
                    provider=event.provider,
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                    cached=event.cached,
                )
            self._total_flushed += len(batch)
        except Exception as e:
            logger.warning("AsyncSpendRecorder flush failed: %s", e)

    def get_stats(self) -> dict:
        """Get recorder statistics."""
        return {
            "total_recorded": self._total_recorded,
            "total_dropped": self._total_dropped,
            "total_flushed": self._total_flushed,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }
