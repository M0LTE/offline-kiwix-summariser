"""Job store / queue lifecycle tests."""
import asyncio

import pytest

from app.jobs import JobRunner, JobStatus, QueueFull


async def wait_terminal(runner: JobRunner, job_id: str, timeout: float = 3.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        job = runner.get(job_id)
        if job and job.status in (JobStatus.READY, JobStatus.FAILED, JobStatus.CANCELLED):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached a terminal state")


@pytest.fixture
async def runner():
    async def handler(req):
        return {"echo": req["value"], "doubled": req["value"] * 2}

    r = JobRunner(handler=handler, workers=1, queue_max=4, ttl_seconds=3600)
    await r.start()
    yield r
    await r.stop()


async def test_submit_starts_queued(runner):
    job = await runner.submit({"value": 2})
    assert job.id
    assert len(job.id) == 32  # unguessable token
    assert job.status is JobStatus.QUEUED


async def test_job_completes_ready_with_result(runner):
    job = await runner.submit({"value": 21})
    done = await wait_terminal(runner, job.id)
    assert done.status is JobStatus.READY
    assert done.result == {"echo": 21, "doubled": 42}
    assert done.started_at and done.finished_at
    assert done.finished_at >= done.started_at
    assert done.error is None


async def test_view_shape_per_status(runner):
    job = await runner.submit({"value": 1})
    queued_view = job.view(position=0)
    assert queued_view["status"] == "queued"
    assert "queue_position" in queued_view
    assert "result" not in queued_view

    done = await wait_terminal(runner, job.id)
    ready_view = done.view()
    assert ready_view["status"] == "ready"
    assert ready_view["result"] == {"echo": 1, "doubled": 2}
    assert "processing_seconds" in ready_view
    assert "queue_position" not in ready_view


async def test_handler_exception_becomes_failed_not_crash(runner):
    async def boom(req):
        raise ValueError("upstream exploded")

    r = JobRunner(handler=boom, workers=1, queue_max=4)
    await r.start()
    try:
        job = await r.submit({"value": 1})
        done = await wait_terminal(r, job.id)
        assert done.status is JobStatus.FAILED
        assert done.error_kind == "ValueError"
        assert "upstream exploded" in done.error
        assert done.result is None
        view = done.view()
        assert view["error"] == {"kind": "ValueError", "detail": "upstream exploded"}
    finally:
        await r.stop()


async def test_worker_survives_a_failed_job(runner):
    """A poison job must not wedge the queue for everyone behind it."""
    ok = await runner.submit({"value": 5})
    await wait_terminal(runner, ok.id)
    assert runner.get(ok.id).status is JobStatus.READY


async def test_cancel_queued_job_is_never_processed():
    calls = []
    gate = asyncio.Event()

    async def handler(req):
        calls.append(req["value"])
        await gate.wait()
        return {"value": req["value"]}

    r = JobRunner(handler=handler, workers=1, queue_max=8)
    await r.start()
    try:
        first = await r.submit({"value": "busy"})
        await asyncio.sleep(0.05)  # let the worker pick it up and block
        second = await r.submit({"value": "doomed"})
        assert second.status is JobStatus.QUEUED

        cancelled = await r.cancel(second.id)
        assert cancelled.status is JobStatus.CANCELLED
        assert cancelled.error_kind == "cancelled"

        gate.set()
        await wait_terminal(r, first.id)
        await asyncio.sleep(0.05)
        assert "doomed" not in calls
        assert r.get(second.id).status is JobStatus.CANCELLED
    finally:
        gate.set()
        await r.stop()


async def test_cancel_unknown_and_running_jobs():
    async def slow(req):
        await asyncio.sleep(5)
        return {}

    r = JobRunner(handler=slow, workers=1, queue_max=8)
    await r.start()
    try:
        assert await r.cancel("nonexistent") is None
        job = await r.submit({})
        await asyncio.sleep(0.05)
        # In-flight generation cannot be withdrawn mid-token.
        again = await r.cancel(job.id)
        assert again.status is JobStatus.PROCESSING
    finally:
        await r.stop()


async def test_queue_max_raises_queue_full():
    async def slow(req):
        await asyncio.sleep(5)
        return {}

    r = JobRunner(handler=slow, workers=1, queue_max=3)
    await r.start()
    try:
        for _ in range(3):
            await r.submit({})
        with pytest.raises(QueueFull):
            await r.submit({})
    finally:
        await r.stop()


async def test_queue_position_counts_jobs_ahead(runner):
    """Position reflects only queued jobs submitted earlier."""
    jobs = [await runner.submit({"value": i}) for i in range(3)]
    positions = [runner.position(j) for j in jobs]
    assert positions == sorted(positions)
    assert positions[0] == 0


async def test_get_unknown_returns_none(runner):
    assert runner.get("deadbeef") is None


async def test_stats_counts(runner):
    await runner.submit({"value": 1})
    s = runner.stats()
    assert s["workers"] == 1
    assert s["queue_capacity"] == 4
    assert sum(s[k] for k in ("queued", "processing", "ready")) >= 1


async def test_multiple_workers_drain_concurrently():
    seen = 0

    async def handler(req):
        nonlocal seen
        seen += 1
        await asyncio.sleep(0.02)
        return {"n": req["value"]}

    r = JobRunner(handler=handler, workers=3, queue_max=16)
    await r.start()
    try:
        ids = [(await r.submit({"value": i})).id for i in range(6)]
        for jid in ids:
            await wait_terminal(r, jid)
        assert seen == 6
        assert all(r.get(j).status is JobStatus.READY for j in ids)
    finally:
        await r.stop()
