# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial breach tests for assay.corpus.store.Corpus.

Targets the confirmed high-severity defects from the prior judge feedback,
plus tries harder on the dismissed get_artifact hash-verification concern.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from assay.corpus import Corpus
from assay.corpus import store as store_module
from assay.corpus.store import _atomic_write


# ---------------------------------------------------------------------------
# Attack 1: put_artifact NOT idempotent across name_hint extensions
# ---------------------------------------------------------------------------
def test_put_artifact_duplicates_across_extensions(tmp_path: Path) -> None:
    """Same bytes + different name_hint extensions => two on-disk files.

    The content hash is identical (it's derived from the bytes), but the
    extension is mixed into the on-disk filename. The existence check only
    looks at one extension at a time, so we get a duplicate.
    """
    c = Corpus(tmp_path / "store")
    blob = b'{"x": 1}'  # valid JSON, valid binary
    h1 = c.put_artifact(blob, kind="diffs", name_hint="a.json")
    h2 = c.put_artifact(blob, kind="diffs", name_hint="b.bin")
    # Same content hash returned for both:
    assert h1 == h2, "content hashes diverged: not even hash-stable"
    files = sorted(
        p.name for p in (c.spec_dir / "artifacts" / "diffs").iterdir()
    )
    # BUG: content-addressed store now has two files for one logical artifact.
    assert len(files) == 1, (
        f"content-addressed invariant broken: {files!r} — "
        "same bytes stored under two filenames"
    )


def test_list_artifacts_yields_duplicate_hash_when_extension_split(
    tmp_path: Path,
) -> None:
    """list_artifacts emits the same ContentHash twice if both .json and
    .bin variants exist for the same content. Downstream callers iterating
    expecting uniqueness will double-process the artifact.
    """
    c = Corpus(tmp_path / "store")
    blob = b"identical-payload"
    c.put_artifact(blob, kind="diffs", name_hint="x.json")
    c.put_artifact(blob, kind="diffs", name_hint="y.bin")
    keys = list(c.list_artifacts(kind="diffs"))
    # BUG: same hash yielded twice.
    assert len(keys) == len(set(keys)), (
        f"list_artifacts yielded duplicates: {keys!r}"
    )


# ---------------------------------------------------------------------------
# Attack 2: list_artifacts leaks in-flight .tmp_* tempfiles
# ---------------------------------------------------------------------------
def test_list_artifacts_does_not_emit_tempfiles(tmp_path: Path) -> None:
    """If a writer crashed mid-write, _atomic_write tempfiles remain. Even
    without a crash, a concurrent writer's tempfile is visible to readers.

    We simulate the leftover-tempfile case by dropping an obvious ``.tmp_*``
    file into the artifacts/<kind>/ directory (exactly what _atomic_write
    would leave on SIGKILL between mkstemp and os.replace).
    """
    c = Corpus(tmp_path / "store")
    kind_dir = c.spec_dir / "artifacts" / "diffs"
    kind_dir.mkdir(parents=True, exist_ok=True)
    # Simulate a crashed writer leaving a tempfile behind:
    leftover = kind_dir / ".tmp_abcdef.bin"
    leftover.write_bytes(b"partial")
    # Also put a legit artifact alongside:
    c.put_artifact(b"good", kind="diffs", name_hint="good.bin")

    keys = list(c.list_artifacts(kind="diffs"))
    # BUG: the tempfile is yielded as a "sha256:.tmp_abcdef" key.
    bad = [k for k in keys if ".tmp_" in k or len(k) != len("sha256:") + 64]
    assert not bad, f"list_artifacts leaked tempfile/non-hash keys: {bad!r}"


def test_list_artifacts_during_concurrent_write_no_tmp_leak(
    tmp_path: Path,
) -> None:
    """Race: readers iterating list_artifacts while writers are in their
    _atomic_write window observe transient .tmp_* siblings.

    We construct the race directly by interposing in the parent dir — drop
    a tempfile in mid-test (mimicking another writer's mkstemp).
    """
    c = Corpus(tmp_path / "store")
    kind_dir = c.spec_dir / "artifacts" / "diffs"
    kind_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    seen_bad: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            for k in c.list_artifacts(kind="diffs"):
                if ".tmp_" in k:
                    seen_bad.append(k)
                    return

    def writer() -> None:
        # Hammer _atomic_write to create+rename tempfiles in the same dir.
        for i in range(200):
            _atomic_write(kind_dir / f"file_{i}.bin", b"payload-%d" % i)

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    w.join(timeout=10)
    stop.set()
    r.join(timeout=5)

    assert not seen_bad, (
        f"reader observed .tmp_* keys leaked through list_artifacts: "
        f"{seen_bad[:3]!r}"
    )


# ---------------------------------------------------------------------------
# Attack 3: open_run tag collision (2-byte secrets.token_hex => 65536 space)
# ---------------------------------------------------------------------------
def test_open_run_raises_clear_error_when_tag_space_truly_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACOR-6 (closed): open_run now uses a 32-bit tag AND retries fresh
    tags on collision. If the tag space is TRULY exhausted — token_hex
    pinned to a constant so every retry collides — it degrades to a clear
    RuntimeError after its retry budget rather than leaking a bare
    FileExistsError from mkdir(exist_ok=False). The realistic concurrent
    case no longer collides at all (see test_open_run_threaded_no_collision).
    """
    import secrets as _secrets

    # Constant tag → every retry produces the same dir → exhaustion.
    monkeypatch.setattr(_secrets, "token_hex", lambda n: "deadbeef")
    # Freeze the timestamp so the directory name is fully determined:
    import datetime as _dt

    class _Frozen(_dt.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return _dt.datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(_dt, "datetime", _Frozen)

    c = Corpus(tmp_path / "store")
    c.open_run("plan")
    # Second call: same ts+recipe+tag, all retries collide → a clear,
    # diagnosable RuntimeError, NOT a raw FileExistsError crash.
    with pytest.raises(RuntimeError, match="could not allocate a unique run dir"):
        c.open_run("plan")


def test_open_run_threaded_no_collision(tmp_path: Path) -> None:
    """Under threaded same-second open_run("plan") calls, mkdir(exist_ok=False)
    with a 16-bit tag will eventually collide. No retry loop exists.
    """
    c = Corpus(tmp_path / "store")
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            c.open_run("plan")
        except BaseException as e:  # pragma: no cover - we want to see it
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Birthday: P(collision among 64 in a 65536 space) ≈ 1 - exp(-64*63/(2*65536))
    # ≈ 3%. Not certain in one trial, but if it fires the bug is real and
    # there is no retry. We assert no errors; if this *does* fire it
    # documents the defect.
    assert not errors, (
        f"open_run raised under concurrent calls (no mkdir retry loop): "
        f"{errors[:3]!r}"
    )


# ---------------------------------------------------------------------------
# Attack 4: get_artifact hash-stem ambiguity / no verification
# ---------------------------------------------------------------------------
def test_get_artifact_rejects_tampered_bytes(tmp_path: Path) -> None:
    """get_artifact must verify content matches the requested hash before
    returning. Tampered/corrupted bytes must NOT be returned silently —
    the content-addressed contract requires re-hashing on read.

    (Originally authored as
    ``test_get_artifact_does_not_verify_content_matches_hash`` —
    inverted assertion to confirm the Defender's hash-verification
    fix in store.py:get_artifact lines ~150-180.)
    """
    c = Corpus(tmp_path / "store")
    blob = b"original-payload"
    h = c.put_artifact(blob, kind="diffs", name_hint="a.bin")
    files = list((c.spec_dir / "artifacts" / "diffs").iterdir())
    assert len(files) == 1
    files[0].write_bytes(b"TAMPERED")

    with pytest.raises(FileNotFoundError):
        c.get_artifact(h)


def test_get_artifact_stem_collision_picks_arbitrary_file(
    tmp_path: Path,
) -> None:
    """Because get_artifact does ``p.stem == hex_`` over rglob, a file
    planted under a *different* kind with the same hex stem is
    indistinguishable from the legitimate one. First-match-wins ordering
    is filesystem-dependent (non-deterministic across platforms).
    """
    c = Corpus(tmp_path / "store")
    blob = b"legit-bytes"
    h = c.put_artifact(blob, kind="diffs", name_hint="a.bin")
    hex_ = h.split(":", 1)[1]

    # Plant a same-stem file in a different kind directory with wrong bytes.
    other = c.spec_dir / "artifacts" / "rogue"
    other.mkdir(parents=True, exist_ok=True)
    (other / f"{hex_}.bin").write_bytes(b"WRONG")

    got = c.get_artifact(h)
    assert got == blob, (
        "get_artifact returned bytes from a same-stem rogue file in a "
        "different kind dir — content-addressed lookup is order-dependent"
    )


def test_get_artifact_bounded_does_not_follow_directory_alias(
    tmp_path: Path,
) -> None:
    """A valid payload outside the held artifact tree grants no authority."""

    corpus = Corpus(tmp_path / "store")
    payload = b"outside-alias-payload"
    ref = corpus.put_artifact(payload, kind="diffs")
    digest = str(ref).split(":", 1)[1]
    next((corpus.spec_dir / "artifacts" / "diffs").iterdir()).unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / f"{digest}.bin").write_bytes(payload)
    (corpus.spec_dir / "artifacts" / "alias").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(FileNotFoundError):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))


def test_get_artifact_bounded_does_not_follow_candidate_symlink(
    tmp_path: Path,
) -> None:
    """A matching-stem final-component alias is not opened or hashed."""

    corpus = Corpus(tmp_path / "store")
    payload = b"symlink-candidate"
    ref = corpus.put_artifact(payload, kind="diffs")
    artifact = next((corpus.spec_dir / "artifacts" / "diffs").iterdir())
    outside = tmp_path / "outside-candidate.bin"
    artifact.rename(outside)
    artifact.symlink_to(outside)

    with pytest.raises(FileNotFoundError):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))


def test_get_artifact_bounded_does_not_open_matching_fifo(
    tmp_path: Path,
) -> None:
    """A planted FIFO is rejected from metadata without a blocking open."""

    corpus = Corpus(tmp_path / "store")
    ref = corpus.put_artifact(b"fifo-ref", kind="diffs")
    artifact = next((corpus.spec_dir / "artifacts" / "diffs").iterdir())
    artifact.unlink()
    os.mkfifo(artifact)

    with pytest.raises(FileNotFoundError):
        corpus.get_artifact_bounded(ref, max_bytes=1024)


def test_get_artifact_bounded_accepts_empty_payload_at_zero_ceiling(
    tmp_path: Path,
) -> None:
    """The inclusive ceiling preserves the valid zero-byte boundary."""

    corpus = Corpus(tmp_path / "store")
    ref = corpus.put_artifact(b"", kind="diffs")

    assert corpus.get_artifact_bounded(ref, max_bytes=0) == b""


def test_get_artifact_bounded_rejects_multiply_linked_candidate(
    tmp_path: Path,
) -> None:
    """A second pathname must not make a mutable inode an authority blob."""

    corpus = Corpus(tmp_path / "store")
    payload = b"single-link-only"
    ref = corpus.put_artifact(payload, kind="diffs")
    digest = str(ref).split(":", 1)[1]
    source = next((corpus.spec_dir / "artifacts" / "diffs").iterdir())
    other = corpus.spec_dir / "artifacts" / "other"
    other.mkdir()
    os.link(source, other / f"{digest}.bin")

    with pytest.raises(FileNotFoundError):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))


def test_get_artifact_bounded_stops_if_candidate_grows_while_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming ceiling remains authoritative after the metadata check."""

    corpus = Corpus(tmp_path / "store")
    payload = b"small-before-open"
    ref = corpus.put_artifact(payload, kind="diffs")
    artifact = next((corpus.spec_dir / "artifacts" / "diffs").iterdir())
    identity = (artifact.stat().st_dev, artifact.stat().st_ino)
    original_read = store_module.os.read
    injected = False

    def growing_read(descriptor: int, count: int) -> bytes:
        nonlocal injected
        metadata = os.fstat(descriptor)
        if not injected and (metadata.st_dev, metadata.st_ino) == identity:
            injected = True
            return b"x" * (len(payload) + 1)
        return original_read(descriptor, count)

    monkeypatch.setattr(store_module.os, "read", growing_read)
    with pytest.raises(ValueError, match="exceeds"):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))
    assert injected


def test_get_artifact_bounded_rejects_path_identity_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid opened inode cannot authenticate a replacement pathname."""

    corpus = Corpus(tmp_path / "store")
    payload = b"opened-before-substitution"
    ref = corpus.put_artifact(payload, kind="diffs")
    artifact = next((corpus.spec_dir / "artifacts" / "diffs").iterdir())
    identity = (artifact.stat().st_dev, artifact.stat().st_ino)
    parked = tmp_path / "parked-original.bin"
    original_read = store_module.os.read
    substituted = False

    def substituting_read(descriptor: int, count: int) -> bytes:
        nonlocal substituted
        metadata = os.fstat(descriptor)
        if not substituted and (metadata.st_dev, metadata.st_ino) == identity:
            artifact.rename(parked)
            artifact.write_bytes(payload)
            substituted = True
        return original_read(descriptor, count)

    monkeypatch.setattr(store_module.os, "read", substituting_read)
    with pytest.raises(FileNotFoundError):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))
    assert substituted


def test_get_artifact_bounded_enforces_entry_ceiling_while_scanning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traversal stops at the ceiling without first materializing the tree."""

    corpus = Corpus(tmp_path / "store")
    ref = corpus.put_artifact(b"removed", kind="diffs")
    next((corpus.spec_dir / "artifacts" / "diffs").iterdir()).unlink()
    for index in range(4):
        (corpus.spec_dir / "artifacts" / f"decoy-{index}").mkdir()
    monkeypatch.setattr(store_module, "_BOUNDED_ARTIFACT_MAX_ENTRIES", 2)

    with pytest.raises(ValueError, match="entry ceiling"):
        corpus.get_artifact_bounded(ref, max_bytes=16)


def test_get_artifact_bounded_enforces_directory_depth_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alias-safe recursion retains a finite descriptor/depth budget."""

    corpus = Corpus(tmp_path / "store")
    payload = b"too-deep"
    ref = corpus.put_artifact(payload, kind="diffs")
    digest = str(ref).split(":", 1)[1]
    next((corpus.spec_dir / "artifacts" / "diffs").iterdir()).unlink()
    deep = corpus.spec_dir / "artifacts" / "one" / "two"
    deep.mkdir(parents=True)
    (deep / f"{digest}.bin").write_bytes(payload)
    monkeypatch.setattr(store_module, "_BOUNDED_ARTIFACT_MAX_DEPTH", 1)

    with pytest.raises(ValueError, match="depth ceiling"):
        corpus.get_artifact_bounded(ref, max_bytes=len(payload))
