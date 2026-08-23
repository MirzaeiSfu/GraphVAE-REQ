"""Gate 2 tests against an explicitly supplied disposable PostgreSQL endpoint.

These tests create and delete only UUID-named Optuna studies. They never drop a
database, schema, table, or study not created by the running test.
"""

import multiprocessing as mp
import copy
import json
import os
import signal
import time
import uuid
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from scripts.graphvae_attr_bo_distributed import (
    BUDGET_INDEX_ATTR,
    LIFECYCLE_ATTR,
    LIFECYCLE_INITIALIZING,
    RESERVED_ATTR,
    UNRESERVED_GUARD_ATTR,
    ControllerLocks,
    DistributedContractError,
    build_study_definition,
    build_worker_sampler,
    canonical_contract_hash,
    create_or_load_distributed_study,
    create_portable_snapshot,
    create_postgresql_storage,
    guard_reserved_trial,
    initialize_reserved_study,
    atomic_write_json,
    reservation_audit,
    sampler_seed,
    sha256_file,
    trial_semantic_fingerprint,
)


pytestmark = pytest.mark.postgres
REPO_ROOT = Path(__file__).resolve().parents[1]
MICRO_PYTHON = Path("/local-scratch2/mirzaei/miniconda3/envs/micro/bin/python")


def _allow_insecure(url):
    return urlsplit(url).hostname in {"localhost", "127.0.0.1", "::1"}


@pytest.fixture
def postgres_url():
    value = os.environ.get("GRAPHVAE_BO_TEST_STORAGE_URL")
    if not value:
        pytest.skip("GRAPHVAE_BO_TEST_STORAGE_URL is not set")
    return value


def _storage(url, heartbeat=60, grace=600):
    return create_postgresql_storage(
        url,
        heartbeat_interval=heartbeat,
        grace_period=grace,
        connect_timeout=5,
        allow_insecure_local_test=_allow_insecure(url),
    )


def _definition(name, trials, max_parallel=2, base_config=None, base_config_sha256="base"):
    return build_study_definition(
        study_name=name,
        study_uuid=str(uuid.uuid4()),
        base_config=base_config or {"epoch_number": 1},
        base_config_sha256=base_config_sha256,
        ranges={
            "alpha_node_feat": {"low": 1e-3, "high": 1e2, "log": True},
            "alpha_edge_feat": {"low": 1e-4, "high": 1e1, "log": True},
            "alpha_motif_loss": None,
        },
        reserved_trials=trials,
        seeds={
            "study_seed": 9,
            "split_seed": 123,
            "training_seed": 0,
            "generation_seed": 123,
            "evaluator_seed": 0,
        },
        evaluator={"mode": "decoded_node_edge", "split": "validation", "test_access": False},
        training={"epoch_number": 1},
        source={"tree_sha256": "source"},
        environment={"sha256": "environment"},
        dataset_cache={"sha256": "cache", "split_fingerprint": "split"},
        feature_schemas={"node_sha256": "node", "edge_sha256": "edge"},
        hardware_policy={"attr_f1pr_abs_tolerance": 0.02},
        heartbeat_interval=60,
        grace_period=600,
        max_parallel=max_parallel,
    )


class IsolatedStudy:
    def __init__(
        self, url, trials, tmp_path, max_parallel=2, heartbeat=60, grace=600,
        worker_ready=False, cache_manifest=None,
    ):
        self.url = url
        self.name = f"graphvae_bo_pytest_{uuid.uuid4().hex}"
        self.storage = _storage(url, heartbeat, grace)
        self.study = create_or_load_distributed_study(
            self.storage,
            study_name=self.name,
            sampler_seed_value=9,
            create=True,
        )
        self.root = tmp_path / self.name
        self.root.mkdir()
        self.config_path = self.root / "base.yaml"
        if worker_ready:
            self.config_path.write_text("epoch_number: 1\n", encoding="utf-8")
            base_hash = __import__("hashlib").sha256(self.config_path.read_bytes()).hexdigest()
        else:
            base_hash = "base"
        self.definition = _definition(
            self.name,
            trials,
            max_parallel=max_parallel,
            base_config={"epoch_number": 1},
            base_config_sha256=base_hash,
        )
        if worker_ready:
            self.definition["source"] = {}
            self.definition["environment"] = {}
            self.definition["training"]["mock"] = True
            self.definition["storage"]["tls_policy"] = (
                "localhost-test-exception" if _allow_insecure(url) else "verify-full"
            )
        if cache_manifest is not None:
            self.definition["dataset_cache"] = copy.deepcopy(cache_manifest)
            self.definition["feature_schemas"] = {
                "node_sha256": cache_manifest["node_schema_fingerprint"],
                "edge_sha256": cache_manifest["edge_schema_fingerprint"],
                "node": copy.deepcopy(cache_manifest["node_schema"]),
                "edge": copy.deepcopy(cache_manifest["edge_schema"]),
            }
        self.controller_uuid = str(uuid.uuid4())
        atomic_write_json(
            self.root / "controller_identity.json",
            {
                "schema_version": "graphvae-attr-f1pr-controller-v1",
                "controller_uuid": self.controller_uuid,
            },
        )
        self.contract = initialize_reserved_study(
            self.study,
            self.definition,
            controller_uuid=self.controller_uuid,
            output_root=self.root,
        )
        self.next_dispatch_sequence = 0

    def cleanup(self):
        import optuna

        optuna.delete_study(study_name=self.name, storage=self.storage)


def _one_process_worker(url, name, contract, seed, barrier, delay, queue):
    try:
        storage = _storage(url)
        study = create_or_load_distributed_study(
            storage, study_name=name, sampler_seed_value=seed, create=False
        )
        barrier.wait(timeout=15)

        def objective(trial):
            index = guard_reserved_trial(trial, study, expected_contract_hash=contract)
            node = trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
            edge = trial.suggest_float("alpha_edge_feat", 1e-4, 1e1, log=True)
            time.sleep(delay)
            trial.set_user_attr("process_budget_index", index)
            return node + edge

        study.optimize(objective, n_trials=1, catch=(Exception,))
        completed = [
            trial for trial in study.get_trials(deepcopy=False)
            if trial.user_attrs.get("process_budget_index") is not None
        ]
        queue.put(("ok", sorted(trial.number for trial in completed)))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _heartbeat_worker(url, name, contract):
    storage = _storage(url, heartbeat=1, grace=2)
    study = create_or_load_distributed_study(
        storage, study_name=name, sampler_seed_value=1, create=False
    )

    def objective(trial):
        guard_reserved_trial(trial, study, expected_contract_hash=contract)
        trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
        time.sleep(60)
        return 0.5

    study.optimize(objective, n_trials=1, catch=(Exception,))


def _artifactless_heartbeat_worker(url, name, contract, worker_run_id):
    storage = _storage(url, heartbeat=1, grace=2)
    study = create_or_load_distributed_study(
        storage, study_name=name, sampler_seed_value=1, create=False
    )

    def objective(trial):
        trial.set_user_attr("worker_id", "heartbeat-worker")
        trial.set_user_attr("worker_run_id", worker_run_id)
        guard_reserved_trial(trial, study, expected_contract_hash=contract)
        trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
        time.sleep(60)
        return 0.5

    study.optimize(objective, n_trials=1, catch=(Exception,))


def _run_mock_worker(isolated, postgres_url, run_id, *extra):
    config_path = isolated.config_path
    assert config_path.is_file()
    atomic_write_json(isolated.root / "study_definition.json", isolated.definition)
    environment = os.environ.copy()
    environment["GRAPHVAE_BO_TEST_STORAGE_URL"] = postgres_url
    dispatch_sequence = isolated.next_dispatch_sequence
    isolated.next_dispatch_sequence += 1
    worker_seed = sampler_seed(
        int(isolated.definition["sampler"]["study_seed"]), dispatch_sequence
    )
    command = [
        str(MICRO_PYTHON),
        str(REPO_ROOT / "scripts" / "run_graphvae_attr_bo_worker.py"),
        "--study-name", isolated.name,
        "--base-config", str(config_path),
        "--artifact-root", str(isolated.root),
        "--study-contract-sha256", isolated.contract,
        "--worker-id", "local-worker",
        "--worker-run-id", run_id,
        "--sampler-seed", str(worker_seed),
        "--dispatch-sequence", str(dispatch_sequence),
        "--device", "cpu",
        "--storage-env", "GRAPHVAE_BO_TEST_STORAGE_URL",
        "--mock",
    ]
    if _allow_insecure(postgres_url):
        command.append("--allow-insecure-local-postgres")
    command.extend(extra)
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_l01_actual_one_trial_worker_writes_structural_artifacts(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1, worker_ready=True)
    try:
        result = _run_mock_worker(isolated, postgres_url, "local-worker-run-1")
        assert result.returncode == 0, result.stderr
        trial = isolated.study.get_trials(deepcopy=False)[0]
        assert trial.state.name == "COMPLETE"
        trial_root = isolated.root / "trials" / "trial_00000"
        record = json.loads((trial_root / "trial_result.json").read_text(encoding="utf-8"))
        assert record["objective_json_path"] == "evaluation.modes.decoded_node_edge.summary.f1_pr.mean"
        assert record["status"] == "COMPLETE"
        assert record["budget_index"] == 0
        assert record["checkpoint_sha256"]
        assert record["evaluator_output_sha256"]
        assert (isolated.root / "workers" / "local-worker-run-1" / "COMPLETED").is_file()
    finally:
        isolated.cleanup()


def test_l02_failure_consumes_one_slot_and_later_reservation_runs(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 2, tmp_path, worker_ready=True)
    try:
        first = _run_mock_worker(
            isolated, postgres_url, "local-worker-fail", "--mock-fail-trial", "0"
        )
        assert first.returncode == 0, first.stderr
        assert isolated.study.get_trials(deepcopy=False)[0].state.name == "FAIL"
        second = _run_mock_worker(isolated, postgres_url, "local-worker-success")
        assert second.returncode == 0, second.stderr
        states = [trial.state.name for trial in isolated.study.get_trials(deepcopy=False)]
        assert states == ["FAIL", "COMPLETE"]
    finally:
        isolated.cleanup()


def test_l04_unauthorized_worker_never_creates_training_artifacts(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1, worker_ready=True)
    try:
        assert _run_mock_worker(isolated, postgres_url, "authorized").returncode == 0
        guard_result = _run_mock_worker(isolated, postgres_url, "unauthorized")
        assert guard_result.returncode == 0, guard_result.stderr
        trials = isolated.study.get_trials(deepcopy=False)
        assert len(trials) == 2
        assert trials[1].state.name == "FAIL"
        assert trials[1].params == {}
        assert trials[1].user_attrs[UNRESERVED_GUARD_ATTR] is True
        assert not (isolated.root / "trials" / "trial_00001").exists()
    finally:
        isolated.cleanup()


def test_p01_empty_reservation_is_claimed_and_samples_missing_parameters(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1)
    try:
        waiting = isolated.study.trials[0]
        assert waiting.state.name == "WAITING"
        assert waiting.params == {}
        assert waiting.user_attrs[RESERVED_ATTR] is True
        assert waiting.user_attrs[BUDGET_INDEX_ATTR] == 0

        def objective(trial):
            guard_reserved_trial(trial, isolated.study, expected_contract_hash=isolated.contract)
            node = trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
            edge = trial.suggest_float("alpha_edge_feat", 1e-4, 1e1, log=True)
            return node + edge

        isolated.study.optimize(objective, n_trials=1)
        trial = isolated.study.trials[0]
        assert trial.state.name == "COMPLETE"
        assert set(trial.params) == {"alpha_node_feat", "alpha_edge_feat"}
        assert trial.user_attrs[BUDGET_INDEX_ATTR] == 0
    finally:
        isolated.cleanup()


def test_p01b_fixed_qualification_parameters_survive_postgres_claim(postgres_url, tmp_path):
    name = f"graphvae_bo_pytest_{uuid.uuid4().hex}"
    storage = _storage(postgres_url)
    study = create_or_load_distributed_study(
        storage, study_name=name, sampler_seed_value=9, create=True
    )
    definition = _definition(name, 2, max_parallel=2)
    fixed = {"alpha_node_feat": 2.0, "alpha_edge_feat": 3.0}
    definition["search_space"]["fixed_parameters"] = fixed
    root = tmp_path / name
    root.mkdir()
    try:
        contract = initialize_reserved_study(
            study,
            definition,
            controller_uuid=str(uuid.uuid4()),
            output_root=root,
        )

        def objective(trial):
            guard_reserved_trial(trial, study, expected_contract_hash=contract)
            node = trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
            edge = trial.suggest_float("alpha_edge_feat", 1e-4, 1e1, log=True)
            return node + edge

        study.optimize(objective, n_trials=2)
        assert [trial.params for trial in study.trials] == [fixed, fixed]
        assert all(trial.state.name == "COMPLETE" for trial in study.trials)
    finally:
        import optuna

        optuna.delete_study(study_name=name, storage=storage)


def test_p02_two_true_processes_claim_distinct_reservations(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 2, tmp_path)
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [
        context.Process(
            target=_one_process_worker,
            args=(postgres_url, isolated.name, isolated.contract, sampler_seed(9, index), barrier, delay, queue),
        )
        for index, delay in enumerate((0.3, 0.05))
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        messages = [queue.get(timeout=5) for _ in processes]
        assert all(message[0] == "ok" for message in messages), messages
        trials = isolated.study.get_trials(deepcopy=False)
        assert {trial.number for trial in trials if trial.state.name == "COMPLETE"} == {0, 1}
        assert {trial.user_attrs[BUDGET_INDEX_ATTR] for trial in trials} == {0, 1}
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
        isolated.cleanup()


def test_p03_interrupted_reservation_initialization_resumes_exact_indexes(postgres_url, tmp_path):
    name = f"graphvae_bo_pytest_{uuid.uuid4().hex}"
    storage = _storage(postgres_url)
    study = create_or_load_distributed_study(
        storage, study_name=name, sampler_seed_value=3, create=True
    )
    definition = _definition(name, 8)
    root = tmp_path / name
    root.mkdir()
    try:
        with pytest.raises(RuntimeError, match="interruption"):
            initialize_reserved_study(
                study,
                definition,
                controller_uuid=str(uuid.uuid4()),
                output_root=root,
                interrupt_after=3,
            )
        assert study.user_attrs[LIFECYCLE_ATTR] == LIFECYCLE_INITIALIZING
        contract = initialize_reserved_study(
            study,
            definition,
            controller_uuid=study.user_attrs["graphvae_bo_controller_uuid"],
            output_root=root,
        )
        audit = reservation_audit(study, 8)
        assert sorted(audit["reserved_by_index"]) == list(range(8))
        assert len(study.trials) == 8
        assert canonical_contract_hash(definition) == contract
    finally:
        import optuna

        optuna.delete_study(study_name=name, storage=storage)


def test_p04_oversubscription_creates_only_parameter_free_fail_guard(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1)
    expensive_starts = []

    def objective(trial):
        guard_reserved_trial(trial, isolated.study, expected_contract_hash=isolated.contract)
        expensive_starts.append(trial.number)
        return trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)

    try:
        isolated.study.optimize(objective, n_trials=1, catch=(Exception,))
        isolated.study.optimize(objective, n_trials=1, catch=(Exception,))
        assert expensive_starts == [0]
        guard = isolated.study.trials[1]
        assert guard.state.name == "FAIL"
        assert guard.params == {}
        assert guard.user_attrs[UNRESERVED_GUARD_ATTR] is True
    finally:
        isolated.cleanup()


def test_p05_sampler_settings_and_dispatch_seeds_are_recordable(monkeypatch):
    import scripts.graphvae_attr_bo_distributed as module

    calls = []

    class RecordingSampler:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module.optuna.samplers, "TPESampler", RecordingSampler)
    seeds = [sampler_seed(5, index) for index in range(3)]
    for seed in seeds:
        build_worker_sampler(seed, startup_trials=5)
    assert [call["seed"] for call in calls] == seeds
    assert all(call["constant_liar"] is True for call in calls)
    assert all(call["n_startup_trials"] == 5 for call in calls)


def test_p06_parallel_contract_explicitly_disclaims_proposal_replay(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 2, tmp_path, max_parallel=2)
    try:
        assert isolated.definition["scheduler"]["study_path_replay"] is False
        assert isolated.definition["scheduler"]["mode"] == "bounded_synchronous_waves"
        assert isolated.definition["seeds"]["study_seed"] == 9
    finally:
        isolated.cleanup()


def test_p07_sigkill_running_worker_is_failed_by_native_stale_sweep(postgres_url, tmp_path):
    import optuna

    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1, heartbeat=1, grace=2)
    context = mp.get_context("spawn")
    process = context.Process(
        target=_heartbeat_worker,
        args=(postgres_url, isolated.name, isolated.contract),
    )
    try:
        process.start()
        deadline = time.time() + 15
        while time.time() < deadline:
            state = isolated.study.get_trials(deepcopy=False)[0].state.name
            if state == "RUNNING":
                break
            time.sleep(0.1)
        assert state == "RUNNING"
        process.kill()
        process.join(timeout=5)
        deadline = time.time() + 15
        while time.time() < deadline:
            optuna.storages.fail_stale_trials(isolated.study)
            state = isolated.study.get_trials(deepcopy=False)[0].state.name
            if state == "FAIL":
                break
            time.sleep(0.5)
        assert state == "FAIL"
    finally:
        if process.is_alive():
            process.kill()
        isolated.cleanup()


def test_p08_preclaim_database_outage_leaves_reservation_waiting(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1)
    try:
        parsed = urlsplit(postgres_url)
        dead_url = postgres_url.replace(
            f":{parsed.port or 5432}", ":1", 1
        ) if parsed.port else postgres_url.replace(parsed.hostname, f"{parsed.hostname}:1", 1)
        with pytest.raises(RuntimeError):
            create_postgresql_storage(
                dead_url,
                heartbeat_interval=1,
                grace_period=2,
                connect_timeout=1,
                allow_insecure_local_test=_allow_insecure(dead_url),
            )
        trials = isolated.study.get_trials(deepcopy=False)
        assert len(trials) == 1
        assert trials[0].state.name == "WAITING"
        assert trials[0].params == {}
    finally:
        isolated.cleanup()


def test_p09_postgres_advisory_and_filesystem_locks_allow_one_controller(postgres_url, tmp_path):
    name = f"graphvae_bo_pytest_{uuid.uuid4().hex}"
    root = tmp_path / name
    with ControllerLocks(root, postgres_url, name):
        with pytest.raises(DistributedContractError):
            with ControllerLocks(root, postgres_url, name):
                pass


def test_p10_portable_snapshot_is_semantic_and_idempotent(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 1, tmp_path, max_parallel=1)
    try:
        isolated.study.optimize(
            lambda trial: (
                guard_reserved_trial(trial, isolated.study, expected_contract_hash=isolated.contract),
                trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True),
            )[1],
            n_trials=1,
        )
        snapshot = create_portable_snapshot(
            isolated.study,
            source_storage=isolated.storage,
            snapshot_path=isolated.root / "study_snapshot.sqlite3",
        )
        before = snapshot.read_bytes()
        assert create_portable_snapshot(
            isolated.study,
            source_storage=isolated.storage,
            snapshot_path=snapshot,
        ) == snapshot
        assert snapshot.read_bytes() == before
        import optuna

        copied = optuna.load_study(study_name=isolated.name, storage="sqlite:///" + snapshot.as_posix())
        assert trial_semantic_fingerprint(copied) == trial_semantic_fingerprint(isolated.study)
    finally:
        isolated.cleanup()


def test_p11_fail_does_not_count_as_usable_startup_observation(postgres_url, tmp_path):
    isolated = IsolatedStudy(postgres_url, 3, tmp_path)
    try:
        def objective(trial):
            guard_reserved_trial(trial, isolated.study, expected_contract_hash=isolated.contract)
            trial.suggest_float("alpha_node_feat", 1e-3, 1e2, log=True)
            if trial.number == 0:
                raise RuntimeError("startup failure")
            return 0.5

        isolated.study.optimize(objective, n_trials=3, catch=(RuntimeError,))
        usable = [trial for trial in isolated.study.trials if trial.state.name == "COMPLETE"]
        failed = [trial for trial in isolated.study.trials if trial.state.name == "FAIL"]
        assert len(usable) == 2
        assert len(failed) == 1
        assert len(usable) < 5
    finally:
        isolated.cleanup()


def test_d05_read_only_cache_survives_actual_mock_worker(postgres_url, tmp_path):
    fixture_root = REPO_ROOT / "tests" / "fixtures" / "distributed_attr_f1pr_bo"
    cache_path = fixture_root / "qm9_tiny_cache.pkl"
    manifest = json.loads(
        (fixture_root / "dataset_cache_manifest.json").read_text(encoding="utf-8")
    )
    isolated = IsolatedStudy(
        postgres_url,
        1,
        tmp_path,
        max_parallel=1,
        worker_ready=True,
        cache_manifest=manifest,
    )
    original_mode = cache_path.stat().st_mode
    before = (cache_path.stat().st_mtime_ns, sha256_file(cache_path))
    try:
        cache_path.chmod(original_mode & ~0o222)
        result = _run_mock_worker(
            isolated, postgres_url, "read-only-cache-worker"
        )
        assert result.returncode == 0, result.stderr
        assert isolated.study.get_trials(deepcopy=False)[0].state.name == "COMPLETE"
        assert (cache_path.stat().st_mtime_ns, sha256_file(cache_path)) == before
    finally:
        cache_path.chmod(original_mode)
        isolated.cleanup()


def test_d08_all_fail_finalizer_reconciles_artifactless_heartbeat(
    postgres_url, tmp_path
):
    import optuna

    isolated = IsolatedStudy(
        postgres_url,
        2,
        tmp_path,
        max_parallel=1,
        heartbeat=1,
        grace=2,
        worker_ready=True,
    )
    process = None
    try:
        first = _run_mock_worker(
            isolated, postgres_url, "ordinary-fail", "--mock-fail-trial", "0"
        )
        assert first.returncode == 0, first.stderr
        assert isolated.study.get_trials(deepcopy=False)[0].state.name == "FAIL"

        heartbeat_run_id = "artifactless-heartbeat"
        heartbeat_dir = isolated.root / "workers" / heartbeat_run_id
        heartbeat_dir.mkdir(parents=True)
        atomic_write_json(
            heartbeat_dir / "RUN_INFO.json",
            {"worker_id": "heartbeat-worker", "worker_run_id": heartbeat_run_id},
        )
        atomic_write_json(
            heartbeat_dir / "HEARTBEAT.json",
            {"worker_run_id": heartbeat_run_id, "updated_at_unix": time.time()},
        )
        context = mp.get_context("spawn")
        process = context.Process(
            target=_artifactless_heartbeat_worker,
            args=(
                postgres_url,
                isolated.name,
                isolated.contract,
                heartbeat_run_id,
            ),
        )
        process.start()
        deadline = time.time() + 15
        state = "WAITING"
        while time.time() < deadline:
            state = isolated.study.get_trials(deepcopy=False)[1].state.name
            if state == "RUNNING":
                break
            time.sleep(0.1)
        assert state == "RUNNING"
        process.kill()
        process.join(timeout=5)
        deadline = time.time() + 15
        while time.time() < deadline:
            optuna.storages.fail_stale_trials(isolated.study)
            state = isolated.study.get_trials(deepcopy=False)[1].state.name
            if state == "FAIL":
                break
            time.sleep(0.5)
        assert state == "FAIL"
        assert not (isolated.root / "trials" / "trial_00001").exists()

        environment = os.environ.copy()
        environment["GRAPHVAE_BO_TEST_STORAGE_URL"] = postgres_url
        command = [
            str(MICRO_PYTHON),
            str(REPO_ROOT / "scripts" / "run_distributed_graphvae_attr_bo.py"),
            "finalize",
            "--study-name", isolated.name,
            "--output-dir", str(isolated.root),
            "--storage-env", "GRAPHVAE_BO_TEST_STORAGE_URL",
        ]
        if _allow_insecure(postgres_url):
            command.append("--allow-insecure-local-postgres")
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 2, result.stderr
        tombstone_path = (
            isolated.root
            / "trials"
            / "trial_00001"
            / "trial_failure_tombstone.json"
        )
        tombstone = json.loads(tombstone_path.read_text(encoding="utf-8"))
        assert tombstone["db_state"] == "FAIL"
        assert tombstone["budget_index"] == 1
        assert tombstone["missing_artifacts"] == [
            {
                "kind": "trial_result",
                "path": "trials/trial_00001/trial_result.json",
                "verified_absent": True,
            }
        ]
        assert (heartbeat_dir / "RECONCILED_FAIL").is_file()
        assert "All reserved scientific trials failed" in (
            isolated.root / "SUMMARY.md"
        ).read_text(encoding="utf-8")
        assert not (isolated.root / "best_trial.json").exists()
        frozen = json.loads(
            (isolated.root / "FROZEN.json").read_text(encoding="utf-8")
        )
        assert frozen["best_trial_number"] is None
        snapshot = optuna.load_study(
            study_name=isolated.name,
            storage="sqlite:///" + (
                isolated.root / "study_snapshot.sqlite3"
            ).as_posix(),
        )
        assert [trial.state.name for trial in snapshot.trials] == ["FAIL", "FAIL"]
    finally:
        if process is not None and process.is_alive():
            process.kill()
        isolated.cleanup()


def test_r01_actual_controller_multi_host_dry_run_is_secret_and_test_safe(
    postgres_url, tmp_path
):
    name = f"graphvae_bo_pytest_{uuid.uuid4().hex}"
    output_root = tmp_path / name
    repo_paths = tmp_path / "repos.txt"
    python_paths = tmp_path / "pythons.txt"
    slots = tmp_path / "slots.txt"
    repo_paths.write_text(
        "worker-a /srv/graphvae-a\nworker-b /srv/graphvae-b\n",
        encoding="utf-8",
    )
    python_paths.write_text(
        "worker-a /env/a/bin/python\nworker-b /env/b/bin/python\n",
        encoding="utf-8",
    )
    slots.write_text(
        "worker-a 3 worker-a-gpu3\nworker-b 7 worker-b-gpu7\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["GRAPHVAE_BO_R01_STORAGE_URL"] = postgres_url
    controller = REPO_ROOT / "scripts" / "run_distributed_graphvae_attr_bo.py"
    common = [
        "--study-name", name,
        "--output-dir", str(output_root),
        "--storage-env", "GRAPHVAE_BO_R01_STORAGE_URL",
    ]
    if _allow_insecure(postgres_url):
        common.append("--allow-insecure-local-postgres")
    storage = None
    try:
        init = subprocess.run(
            [
                str(MICRO_PYTHON),
                str(controller),
                "init",
                *common,
                "--base-config",
                "configs/bayesian_optimization/qm9_graphvae_attr_f1pr_smoke.yaml",
                "--dataset-cache-manifest",
                str(
                    REPO_ROOT
                    / "tests"
                    / "fixtures"
                    / "distributed_attr_f1pr_bo"
                    / "dataset_cache_manifest.json"
                ),
                "--trials", "2",
                "--max-parallel", "2",
                "--sampler-seed", "11",
                "--mock",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert init.returncode == 0, init.stderr
        storage = _storage(postgres_url)

        preflight = subprocess.run(
            [
                str(MICRO_PYTHON),
                str(controller),
                "preflight",
                *common,
                "--repo-paths", str(repo_paths),
                "--python-paths", str(python_paths),
                "--slots", str(slots),
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert preflight.returncode == 0, preflight.stderr

        dry_run = subprocess.run(
            [
                str(MICRO_PYTHON),
                str(controller),
                "run",
                *common,
                "--base-config",
                "configs/bayesian_optimization/qm9_graphvae_attr_f1pr_smoke.yaml",
                "--repo-paths", str(repo_paths),
                "--python-paths", str(python_paths),
                "--slots", str(slots),
                "--max-parallel", "2",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert dry_run.returncode == 0, dry_run.stderr
        manifest = json.loads(
            (
                output_root / "launch_manifests" / "wave_0001.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["dry_run"] is True
        assert len(manifest["launches"]) == 2
        assert {launch["physical_gpu"] for launch in manifest["launches"]} == {3, 7}
        for launch in manifest["launches"]:
            command = launch["remote_command"]
            assert f"CUDA_VISIBLE_DEVICES={launch['physical_gpu']}" in command
            assert f"--physical-gpu {launch['physical_gpu']}" in command
            assert "--device cuda:0" in command
            assert "--mock" in command
            assert "--storage-env GRAPHVAE_BO_R01_STORAGE_URL" in command
            assert "--split" not in command
            assert "--execute-remote" not in command
            assert "final_test" not in command
            assert "PGPASS" not in command
            assert "password" not in command.lower()
            assert "postgresql" not in command.lower()
            assert postgres_url not in command
    finally:
        if storage is None:
            try:
                storage = _storage(postgres_url)
            except Exception:
                storage = None
        if storage is not None:
            try:
                import optuna

                optuna.delete_study(study_name=name, storage=storage)
            except KeyError:
                pass
