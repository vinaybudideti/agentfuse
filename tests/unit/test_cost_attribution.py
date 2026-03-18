"""Tests for CostAttribution."""

from agentfuse.core.cost_attribution import CostAttribution


def test_record_by_user():
    attr = CostAttribution()
    attr.record(cost=0.05, user_id="user_1")
    attr.record(cost=0.10, user_id="user_1")
    by_user = attr.get_by_user()
    assert by_user["user_1"]["total_cost"] == 0.15
    assert by_user["user_1"]["calls"] == 2


def test_record_by_team():
    attr = CostAttribution()
    attr.record(cost=0.05, team="engineering")
    attr.record(cost=0.10, team="sales")
    by_team = attr.get_by_team()
    assert by_team["engineering"]["total_cost"] == 0.05
    assert by_team["sales"]["total_cost"] == 0.10


def test_record_by_feature():
    attr = CostAttribution()
    attr.record(cost=0.05, feature="chat")
    attr.record(cost=0.20, feature="search")
    by_feature = attr.get_by_feature()
    assert by_feature["chat"]["total_cost"] == 0.05
    assert by_feature["search"]["total_cost"] == 0.20


def test_top_users():
    attr = CostAttribution()
    attr.record(cost=0.50, user_id="big_spender")
    attr.record(cost=0.01, user_id="small_user")
    top = attr.get_top_users(n=2)
    assert top[0][0] == "big_spender"


def test_summary():
    attr = CostAttribution()
    attr.record(cost=0.05, user_id="u1", team="t1", feature="f1", session_id="s1")
    summary = attr.get_summary()
    assert summary["unique_users"] == 1
    assert summary["unique_teams"] == 1
    assert summary["unique_features"] == 1
    assert summary["unique_sessions"] == 1


def test_reset():
    attr = CostAttribution()
    attr.record(cost=0.05, user_id="u1")
    attr.reset()
    assert attr.get_summary()["unique_users"] == 0


def test_thread_safety():
    import threading
    attr = CostAttribution()
    errors = []
    def worker():
        try:
            for _ in range(100):
                attr.record(cost=0.001, user_id="concurrent", team="team")
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(errors) == 0
    assert attr.get_by_user()["concurrent"]["calls"] == 500
