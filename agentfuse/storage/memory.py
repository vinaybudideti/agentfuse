# InMemoryStore: default zero-dependency run state storage

import threading


class InMemoryStore:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def set(self, run_id, key, value):
        with self._lock:
            if run_id not in self._store:
                self._store[run_id] = {}
            self._store[run_id][key] = value

    def get(self, run_id, key, default=None):
        with self._lock:
            return self._store.get(run_id, {}).get(key, default)

    def delete(self, run_id):
        with self._lock:
            self._store.pop(run_id, None)

    def get_all(self, run_id):
        with self._lock:
            return dict(self._store.get(run_id, {}))

    def list_runs(self):
        with self._lock:
            return list(self._store.keys())
