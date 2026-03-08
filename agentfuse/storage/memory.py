# InMemoryStore: default zero-dependency run state storage


class InMemoryStore:
    def __init__(self):
        self._store = {}

    def set(self, run_id, key, value):
        if run_id not in self._store:
            self._store[run_id] = {}
        self._store[run_id][key] = value

    def get(self, run_id, key, default=None):
        return self._store.get(run_id, {}).get(key, default)

    def delete(self, run_id):
        self._store.pop(run_id, None)

    def get_all(self, run_id):
        return dict(self._store.get(run_id, {}))

    def list_runs(self):
        return list(self._store.keys())
