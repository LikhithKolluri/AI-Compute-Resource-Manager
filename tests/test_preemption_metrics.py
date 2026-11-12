from src.scheduler import GPUScheduler, Job, Status


def test_successful_preemption_increments_counter():
    scheduler = GPUScheduler()

    try:
        training = Job(
            name="flexible-training",
            vram_gb=7.0,
            priority=2,
            preemptible=True,
        )

        scheduler.submit(training)

        assert training.status == Status.RUNNING
        assert training.borrowed_gb == 1.0
        assert scheduler.preemption_count == 0

        inference = Job(
            name="critical-inference",
            vram_gb=2.0,
            priority=1,
        )

        scheduler.submit(inference)

        assert scheduler.preemption_count == 1
        assert training.status == Status.PAUSED
        assert inference.status == Status.RUNNING

    finally:
        scheduler.stop_heartbeat()