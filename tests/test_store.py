from concurrent.futures import ThreadPoolExecutor

from navid.store import QueueFull, Store


def test_concurrent_admission_and_claim_are_atomic(tmp_path):
    store = Store(tmp_path)

    def submit(_):
        try:
            return store.enqueue({"prompt": "p", "duration": 5, "seed": 0, "references": []}, limit=8)
        except QueueFull:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        accepted = [job for job in pool.map(submit, range(40)) if job]
        claimed = [job for job in pool.map(lambda _: store.claim(), range(16)) if job]
    assert len(accepted) == 8
    assert {job["id"] for job in claimed} == set(accepted)
    assert len(claimed) == 8
    store.fail_pending("restarted")
    assert all(store.get(job)["status"] == "failed" for job in accepted)


def test_late_completion_cannot_revive_failed_job(tmp_path):
    store = Store(tmp_path)
    job = store.enqueue({"duration": 5, "seed": 0}, limit=1)
    store.claim()
    store.fail_pending("stopped")
    store.finish(job, output="late.mp4")
    assert store.get(job)["status"] == "failed"
