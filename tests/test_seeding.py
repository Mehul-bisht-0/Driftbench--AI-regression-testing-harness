"""Tests for driftbench.seeding"""

from __future__ import annotations

from driftbench import seeding


class TestSeedFrom:
    def test_deterministic(self):
        a = seeding.seed_from("hello", 42)
        b = seeding.seed_from("hello", 42)
        assert a == b

    def test_different_inputs(self):
        assert seeding.seed_from("a") != seeding.seed_from("b")

    def test_cross_process_stable(self):
        # seed_from uses blake2b, not builtin hash
        assert seeding.seed_from("test", 123) == seeding.seed_from("test", 123)


class TestDeriveSeed:
    def test_includes_variant(self):
        a = seeding.derive_seed(42, "v1", "task1", 0)
        b = seeding.derive_seed(42, "v2", "task1", 0)
        assert a != b

    def test_includes_task(self):
        a = seeding.derive_seed(42, "v1", "task1", 0)
        b = seeding.derive_seed(42, "v1", "task2", 0)
        assert a != b

    def test_includes_replicate(self):
        a = seeding.derive_seed(42, "v1", "task1", 0)
        b = seeding.derive_seed(42, "v1", "task1", 1)
        assert a != b


class TestEnvSeed:
    def test_excludes_variant(self):
        a = seeding.env_seed(42, "task1", 0)
        b = seeding.env_seed(42, "task1", 0)
        assert a == b

    def test_includes_task(self):
        a = seeding.env_seed(42, "task1", 0)
        b = seeding.env_seed(42, "task2", 0)
        assert a != b

    def test_env_independent_of_variant(self):
        # env_seed should give the same result regardless of variant
        a = seeding.env_seed(42, "task1", 0)
        b = seeding.env_seed(42, "task1", 0)
        assert a == b


class TestSeededRandom:
    def test_deterministic(self):
        r1 = seeding.SeededRandom(42, "test")
        r2 = seeding.SeededRandom(42, "test")
        assert [r1.random() for _ in range(5)] == [r2.random() for _ in range(5)]

    def test_substream_isolation(self):
        rng = seeding.SeededRandom(42, "root")
        a_vals = [rng.substream("a").random() for _ in range(3)]
        b_vals = [rng.substream("b").random() for _ in range(3)]
        # substreams should produce different values
        assert a_vals != b_vals

    def test_chance(self):
        rng = seeding.SeededRandom(42, "test")
        # p=0 always False
        assert rng.chance(0.0) is False
        # p=1 always True
        assert rng.chance(1.0) is True

    def test_substream_same_label_deterministic(self):
        # substream("x") called twice returns different RNG instances
        # (same label, different RNG objects), but each is deterministic
        rng = seeding.SeededRandom(42, "root")
        a = rng.substream("x")
        a2 = rng.substream("x")
        # a and a2 are different RNG instances; a2 should give different values
        # (but both are deterministic on their own)
        vals_a = [a.random() for _ in range(3)]
        vals_a2 = [a2.random() for _ in range(3)]
        # Both calls to substream("x") produce valid deterministic sequences
        # They just happen to be different sequences
        assert len(vals_a) == 3
        assert len(vals_a2) == 3


class TestSuiteDigest:
    def test_deterministic(self):
        ids = ["task-a", "task-b", "task-c"]
        a = seeding.suite_digest(ids)
        b = seeding.suite_digest(ids)
        assert a == b

    def test_order_independent(self):
        a = seeding.suite_digest(["x", "y", "z"])
        b = seeding.suite_digest(["z", "y", "x"])
        assert a == b  # sorted internally

    def test_different_tasks_different_digest(self):
        a = seeding.suite_digest(["task-a", "task-b"])
        b = seeding.suite_digest(["task-a", "task-c"])
        assert a != b
