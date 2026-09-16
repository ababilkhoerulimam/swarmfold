import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from threading import BoundedSemaphore, Event, Lock, Thread
from time import monotonic, process_time

import psutil


class TrainingResourcePool:
    def __init__(self, max_concurrent: int = 1) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.max_concurrent = max_concurrent
        self._semaphore = BoundedSemaphore(max_concurrent)
        self._counter_lock = Lock()
        self._active = 0
        self.peak_active = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._semaphore:
            with self._counter_lock:
                self._active += 1
                self.peak_active = max(self.peak_active, self._active)
            try:
                yield
            finally:
                with self._counter_lock:
                    self._active -= 1


LOCAL_TRAINING_POOL = TrainingResourcePool(
    max_concurrent=int(os.getenv("SWARMFOLD_MAX_CONCURRENT_TRAINING", "2"))
)


class ProcessResourceMeter:
    def __init__(self, sample_interval_seconds: float = 0.05) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._process = psutil.Process()
        self._stop = Event()
        self._thread: Thread | None = None
        self._started_wall = 0.0
        self._started_cpu = 0.0
        self._peak_rss = 0
        self._measurement: dict[str, float] | None = None

    def __enter__(self) -> "ProcessResourceMeter":
        self._started_wall = monotonic()
        self._started_cpu = process_time()
        try:
            self._peak_rss = self._process.memory_info().rss
        except psutil.Error:
            self._peak_rss = 0
        self._thread = Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        with suppress(psutil.Error):
            self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
        self._measurement = {
            # Some Windows clocks can report the same tick for very short work.
            # A completed measurement must still represent a positive duration.
            "wall_seconds": max(monotonic() - self._started_wall, 1e-9),
            "cpu_seconds": process_time() - self._started_cpu,
            "peak_rss_mb": self._peak_rss / (1024 * 1024),
        }

    def result(self) -> dict[str, float]:
        if self._measurement is None:
            raise RuntimeError("Resource measurement is not complete")
        return dict(self._measurement)

    def _sample(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            try:
                self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
            except psutil.Error:
                return
