"""In-memory job store and GPU-bound worker pool.

Jobs are addressed by an unguessable token so the status URL is capability-style:
holding it is what authorises reading the result.

State is process-local by design -- a restart drops pending work. Workers
default to one because a single consumer GPU serialises inference anyway;
more workers would only contend for VRAM.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable

log = logging.getLogger("jobs")


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED}


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    request: dict[str, Any]
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    error_kind: str | None = None

    def view(self, position: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.id,
            "status": self.status.value,
            "request": self.request,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
        if self.status is JobStatus.QUEUED and position is not None:
            out["queue_position"] = position
        if self.finished_at and self.started_at:
            out["processing_seconds"] = round(self.finished_at - self.started_at, 3)
        if self.status is JobStatus.READY:
            out["result"] = self.result
        elif self.status in (JobStatus.FAILED, JobStatus.CANCELLED):
            out["error"] = {"kind": self.error_kind, "detail": self.error}
        return out


Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class JobRunner:
    def __init__(
        self,
        handler: Handler,
        workers: int = 1,
        queue_max: int = 64,
        ttl_seconds: int = 3600,
        poll_seconds: float = 2.0,
    ):
        self.handler = handler
        self.workers = max(1, workers)
        self.queue_max = queue_max
        self.ttl_seconds = ttl_seconds
        self.poll_seconds = poll_seconds

        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []

    # --- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        for i in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker(i), name=f"worker-{i}"))
        self._tasks.append(asyncio.create_task(self._reaper(), name="reaper"))
        log.info("started %d summarisation worker(s)", self.workers)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # --- submission --------------------------------------------------------
    async def submit(self, request: dict[str, Any]) -> Job:
        async with self._lock:
            pending = sum(
                1 for j in self._jobs.values() if j.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
            )
            if pending >= self.queue_max:
                raise QueueFull(
                    f"{pending} jobs already queued or running (limit {self.queue_max})"
                )
            job = Job(id=uuid.uuid4().hex, request=request)
            self._jobs[job.id] = job
            self._order.append(job.id)
        await self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Only queued work can be withdrawn; an in-flight GPU generation is
            # not interruptible mid-token.
            if job.status is JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.finished_at = time.time()
                job.error_kind = "cancelled"
                job.error = "cancelled before processing started"
            return job

    def position(self, job: Job) -> int | None:
        if job.status is not JobStatus.QUEUED:
            return None
        ahead = 0
        for jid in self._order:
            j = self._jobs.get(jid)
            if j is None or j is job:
                break
            if j.status is JobStatus.QUEUED:
                ahead += 1
        return ahead

    def stats(self) -> dict[str, int]:
        counts = {s.value: 0 for s in JobStatus}
        for j in self._jobs.values():
            counts[j.status.value] += 1
        counts["queue_capacity"] = self.queue_max
        counts["workers"] = self.workers
        return counts

    # --- internals ---------------------------------------------------------
    async def _worker(self, index: int) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.status is not JobStatus.QUEUED:
                    continue  # cancelled while waiting
                job.status = JobStatus.PROCESSING
                job.started_at = time.time()
                log.info("worker-%d processing %s %s", index, job_id[:8], job.request)
                try:
                    job.result = await self.handler(job.request)
                    job.status = JobStatus.READY
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - reported to the poller
                    job.status = JobStatus.FAILED
                    job.error_kind = type(exc).__name__
                    job.error = str(exc)
                    log.warning("job %s failed: %s: %s", job_id[:8], job.error_kind, job.error)
                finally:
                    job.finished_at = time.time()
                    log.info(
                        "job %s -> %s in %.1fs",
                        job_id[:8], job.status.value, job.finished_at - job.started_at,
                    )
            finally:
                self._queue.task_done()

    async def _reaper(self) -> None:
        """Drop terminal jobs past TTL so a long-lived process cannot grow."""
        while True:
            await asyncio.sleep(max(30.0, self.poll_seconds))
            cutoff = time.time() - self.ttl_seconds
            async with self._lock:
                expired = [
                    jid for jid in self._order
                    if (j := self._jobs.get(jid))
                    and j.status in TERMINAL
                    and (j.finished_at or j.created_at) < cutoff
                ]
                for jid in expired:
                    del self._jobs[jid]
                    self._order.remove(jid)
            if expired:
                log.info("reaped %d expired job(s)", len(expired))
