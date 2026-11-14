# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``assay.primitives.tool_cache``."""

from __future__ import annotations

import threading


from assay import ToolCallCache, canonical_args


def test_canonical_args_sorts_keys():
    a = canonical_args({"b": 1, "a": 2})
    b = canonical_args({"a": 2, "b": 1})
    assert a == b == '{"a": 2, "b": 1}'


def test_canonical_args_default_str_collapses_paths():
    """Path("foo") and "foo" should canonicalize to the same key."""
    from pathlib import Path
    assert canonical_args({"p": Path("foo")}) == canonical_args({"p": "foo"})


def test_canonical_args_handles_nested_and_unicode():
    out = canonical_args({"data": {"name": "café", "ids": [3, 1, 2]}})
    assert "café" in out
    # ids list preserves its original order (sort_keys only affects dict keys).
    assert '"ids": [3, 1, 2]' in out


def test_cache_miss_returns_none():
    cache = ToolCallCache()
    assert cache.get("read_source", {"path": "x.py"}) is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_hit_after_put():
    cache = ToolCallCache()
    cache.put("read_source", {"path": "x.py"}, "result-string")
    assert cache.get("read_source", {"path": "x.py"}) == "result-string"
    assert cache.hits == 1


def test_cache_distinguishes_args():
    cache = ToolCallCache()
    cache.put("read_source", {"path": "a.py"}, "A")
    cache.put("read_source", {"path": "b.py"}, "B")
    assert cache.get("read_source", {"path": "a.py"}) == "A"
    assert cache.get("read_source", {"path": "b.py"}) == "B"


def test_cache_collapses_path_arguments():
    """Path("a.py") and "a.py" hash to the same key (default=str)."""
    from pathlib import Path
    cache = ToolCallCache()
    cache.put("read_source", {"path": Path("a.py")}, "OK")
    assert cache.get("read_source", {"path": "a.py"}) == "OK"


def test_cache_key_independent_of_dict_insertion_order():
    cache = ToolCallCache()
    cache.put("grep", {"path": "a", "pattern": "foo"}, "match")
    # Different insertion order — same canonical key.
    assert cache.get("grep", {"pattern": "foo", "path": "a"}) == "match"


def test_cache_overwrite_on_repeat_put():
    cache = ToolCallCache()
    cache.put("read_source", {"path": "x"}, "old")
    cache.put("read_source", {"path": "x"}, "new")
    assert cache.get("read_source", {"path": "x"}) == "new"


def test_cache_clear():
    cache = ToolCallCache()
    cache.put("a", {}, "1")
    cache.get("a", {})
    cache.get("b", {})
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_stats_summary():
    cache = ToolCallCache()
    cache.put("a", {}, "A")
    cache.get("a", {})  # hit
    cache.get("b", {})  # miss
    stats = cache.stats()
    assert stats == {"hits": 1, "misses": 1, "size": 1}


def test_cache_size_tracks_entries():
    cache = ToolCallCache()
    assert cache.size == 0
    cache.put("a", {}, "1")
    cache.put("a", {"x": 1}, "2")
    cache.put("b", {}, "3")
    assert cache.size == 3


def test_cache_shared_by_reference_across_phases():
    """One cache instance can be shared by reference across phases."""
    shared = ToolCallCache()

    def phase_rca():
        shared.put("read_source", {"path": "kit/main.cpp"}, "<bytes>")

    def phase_solutioning():
        # Solutioning phase reads the same file; should get an instant hit.
        return shared.get("read_source", {"path": "kit/main.cpp"})

    phase_rca()
    result = phase_solutioning()
    assert result == "<bytes>"
    assert shared.hits == 1
    assert shared.misses == 0


def test_cache_is_thread_safe_for_stats():
    """Concurrent gets shouldn't lose hit-counter increments to races."""
    cache = ToolCallCache()
    cache.put("a", {}, "A")

    N = 200
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        for _ in range(N):
            cache.get("a", {})

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cache.hits == 2 * N
