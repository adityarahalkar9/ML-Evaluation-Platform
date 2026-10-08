from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from app.core.exceptions import NotFoundError

logger = logging.getLogger("app.jobs")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    type: str
    description: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    current_step: str = "waiting for a worker"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=400))
    result: Any = None
    error: str | None = None

    def log(self, message: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')} | {message}")

    def set_progress(self, fraction: float, step: str) -> None:
        self.progress = max(self.progress, min(float(fraction), 1.0))
        self.current_step = step

    def to_dict(self) -> dict:
        def iso(ts: float | None) -> str | None:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds") if ts else None

        duration = None
        if self.started_at:
            duration = (self.finished_at or time.time()) - self.started_at
        return {
            "job_id": self.id,
            "type": self.type,
            "description": self.description,
            "status": self.status.value,
            "progress": round(self.progress, 3),
            "current_step": self.current_step,
            "created_at": iso(self.created_at),
            "started_at": iso(self.started_at),
            "finished_at": iso(self.finished_at),
            "duration_sec": round(duration, 2) if duration is not None else None,
            "error": self.error,
            "result": self.result,
            "logs": list(self.logs),
        }


class JobManager:
    """Background jobs on a small thread pool with an in-memory job table.

    max_workers=1 serializes training jobs -> predictable CPU/RAM usage and
    reproducible results. Swap for Celery/Redis if you need distribution.
    """

    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="mlep")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, job_type: str, description: str, fn: Callable[[Job], Any]) -> str:
        job = Job(id=f"job_{uuid.uuid4().hex[:10]}", type=job_type, description=description)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job, fn)
        return job.id

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.current_step = "starting"
        try:
            job.result = fn(job)
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.current_step = "completed"
        except Exception as exc:  # noqa: BLE001 - error is surfaced to the client
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.current_step = "failed"
            logger.exception("job %s failed", job.id)
            job.log(f"FAILED: {job.error}")
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("Job", job_id)
        return job

    def list_jobs(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)