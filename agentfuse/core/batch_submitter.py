"""
BatchSubmitter — actual batch API execution for OpenAI and Anthropic.

BatchEligibilityDetector detects opportunities. This module EXECUTES them.

OpenAI Batch API: 50% discount, 50K requests/file, 24h SLA
Anthropic Message Batches: 50% discount, 100K requests/batch, stacks with caching

Usage:
    submitter = BatchSubmitter(provider="openai")
    batch_id = submitter.submit([
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Q1"}]},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "Q2"}]},
    ])
    results = submitter.get_results(batch_id)
"""

import json
import logging
import time
import tempfile
import os
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    """Tracks a submitted batch job."""
    batch_id: str
    provider: str
    request_count: int
    status: str = "pending"
    submitted_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    results: Optional[list] = None
    error: Optional[str] = None


class BatchSubmitter:
    """
    Submits batch requests to OpenAI and Anthropic batch APIs.

    Both providers offer 50% discount for batch processing.
    Anthropic stacks with prompt caching for up to 95% savings.
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self._provider = provider
        self._api_key = api_key
        self._jobs: dict[str, BatchJob] = {}

    def submit(
        self,
        requests: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 4096,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Submit a batch of requests.

        Args:
            requests: List of {"messages": [...], "model": "..."} dicts
            model: Default model (can be overridden per request)
            max_tokens: Default max tokens per response
            metadata: Optional metadata for tracking

        Returns:
            batch_id string for tracking
        """
        if self._provider == "openai":
            return self._submit_openai(requests, model or "gpt-4o", max_tokens, metadata)
        elif self._provider == "anthropic":
            return self._submit_anthropic(requests, model or "claude-sonnet-4-6", max_tokens, metadata)
        else:
            raise ValueError(f"Unsupported batch provider: {self._provider}")

    def _submit_openai(self, requests, model, max_tokens, metadata) -> str:
        """Submit via OpenAI Batch API."""
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        client_kwargs = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        client = openai.OpenAI(**client_kwargs)

        # Create JSONL input file
        jsonl_lines = []
        for i, req in enumerate(requests):
            entry = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req.get("model", model),
                    "messages": req["messages"],
                    "max_tokens": req.get("max_tokens", max_tokens),
                },
            }
            jsonl_lines.append(json.dumps(entry))

        # Write to temp file and upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_lines))
            temp_path = f.name

        try:
            batch_file = client.files.create(
                file=open(temp_path, "rb"), purpose="batch"
            )
            batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata or {},
            )

            job = BatchJob(
                batch_id=batch_job.id,
                provider="openai",
                request_count=len(requests),
                status="submitted",
            )
            self._jobs[batch_job.id] = job
            logger.info("OpenAI batch submitted: %s (%d requests)", batch_job.id, len(requests))
            return batch_job.id
        finally:
            os.unlink(temp_path)

    def _submit_anthropic(self, requests, model, max_tokens, metadata) -> str:
        """Submit via Anthropic Message Batches API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

        client_kwargs = {}
        if self._api_key:
            client_kwargs["api_key"] = self._api_key
        client = anthropic.Anthropic(**client_kwargs)

        batch_requests = []
        for i, req in enumerate(requests):
            messages = req["messages"]
            # Extract system messages (Anthropic handles separately)
            system_msgs = [m["content"] for m in messages if m.get("role") == "system"]
            chat_msgs = [m for m in messages if m.get("role") != "system"]

            params = {
                "model": req.get("model", model),
                "max_tokens": req.get("max_tokens", max_tokens),
                "messages": chat_msgs,
            }
            if system_msgs:
                params["system"] = "\n".join(system_msgs)

            batch_requests.append({
                "custom_id": f"task-{i}",
                "params": params,
            })

        batch = client.messages.batches.create(requests=batch_requests)

        job = BatchJob(
            batch_id=batch.id,
            provider="anthropic",
            request_count=len(requests),
            status="submitted",
        )
        self._jobs[batch.id] = job
        logger.info("Anthropic batch submitted: %s (%d requests)", batch.id, len(requests))
        return batch.id

    def check_status(self, batch_id: str) -> str:
        """Check the status of a batch job."""
        job = self._jobs.get(batch_id)
        if not job:
            return "unknown"

        if job.provider == "openai":
            return self._check_openai_status(batch_id, job)
        elif job.provider == "anthropic":
            return self._check_anthropic_status(batch_id, job)
        return job.status

    def _check_openai_status(self, batch_id, job) -> str:
        try:
            import openai
            client = openai.OpenAI(**({"api_key": self._api_key} if self._api_key else {}))
            batch = client.batches.retrieve(batch_id)
            job.status = batch.status
            if batch.status == "completed":
                job.completed_at = time.time()
            return batch.status
        except Exception as e:
            logger.warning("Failed to check OpenAI batch status: %s", e)
            return job.status

    def _check_anthropic_status(self, batch_id, job) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(**({"api_key": self._api_key} if self._api_key else {}))
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            job.status = "completed" if status == "ended" else status
            if job.status == "completed":
                job.completed_at = time.time()
            return job.status
        except Exception as e:
            logger.warning("Failed to check Anthropic batch status: %s", e)
            return job.status

    def get_results(self, batch_id: str) -> Optional[list[dict]]:
        """Get results of a completed batch job."""
        job = self._jobs.get(batch_id)
        if not job or job.status != "completed":
            return None

        if job.results is not None:
            return job.results

        if job.provider == "openai":
            return self._get_openai_results(batch_id, job)
        elif job.provider == "anthropic":
            return self._get_anthropic_results(batch_id, job)
        return None

    def _get_openai_results(self, batch_id, job) -> Optional[list[dict]]:
        try:
            import openai
            client = openai.OpenAI(**({"api_key": self._api_key} if self._api_key else {}))
            batch = client.batches.retrieve(batch_id)
            if not batch.output_file_id:
                return None
            result_file = client.files.content(batch.output_file_id)
            results = []
            for line in result_file.text.strip().split("\n"):
                entry = json.loads(line)
                content = entry["response"]["body"]["choices"][0]["message"]["content"]
                results.append({"custom_id": entry["custom_id"], "content": content})
            job.results = results
            return results
        except Exception as e:
            logger.warning("Failed to get OpenAI batch results: %s", e)
            return None

    def _get_anthropic_results(self, batch_id, job) -> Optional[list[dict]]:
        try:
            import anthropic
            client = anthropic.Anthropic(**({"api_key": self._api_key} if self._api_key else {}))
            results = []
            for entry in client.messages.batches.results(batch_id):
                if entry.result.type == "succeeded":
                    text = "".join(
                        b.text for b in entry.result.message.content if b.type == "text"
                    )
                    results.append({"custom_id": entry.custom_id, "content": text})
            job.results = results
            return results
        except Exception as e:
            logger.warning("Failed to get Anthropic batch results: %s", e)
            return None

    def get_job(self, batch_id: str) -> Optional[BatchJob]:
        """Get a batch job by ID."""
        return self._jobs.get(batch_id)

    def list_jobs(self) -> list[BatchJob]:
        """List all batch jobs."""
        return list(self._jobs.values())

    def estimate_savings(self, requests: list[dict], model: str = "gpt-4o") -> dict:
        """Estimate savings from batching vs real-time."""
        from agentfuse.providers.pricing import ModelPricingEngine
        pricing = ModelPricingEngine()

        total_realtime = 0.0
        for req in requests:
            est = pricing.estimate_cost(req.get("model", model), req["messages"])
            total_realtime += est

        return {
            "request_count": len(requests),
            "realtime_cost_usd": round(total_realtime, 6),
            "batch_cost_usd": round(total_realtime * 0.50, 6),  # 50% discount
            "savings_usd": round(total_realtime * 0.50, 6),
            "savings_pct": 50.0,
        }
