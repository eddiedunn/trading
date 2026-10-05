"""Unit tests for the attempt ledger (SQLite under TRADING_RESULTS_DIR)."""

import json
import sqlite3

import pytest

from backtest_api import ledger


@pytest.fixture(autouse=True)
def _results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADING_RESULTS_DIR", str(tmp_path / "results"))


def test_file_is_created_on_first_use(tmp_path):
    assert not (tmp_path / "results").exists()
    assert ledger.attempt_count("2026-04-06") == 0
    assert ledger.ledger_path() == tmp_path / "results" / "ledger.sqlite"
    assert ledger.ledger_path().exists()


class TestHash:
    def test_trailing_whitespace_and_blank_lines_do_not_change_it(self):
        assert ledger.code_sha256("a = 1\nb = 2\n") == ledger.code_sha256("a = 1   \nb = 2\t\n\n\n")

    def test_any_real_change_does(self):
        base = ledger.code_sha256("a = 1\nb = 2\n")
        assert ledger.code_sha256("a = 1\nb = 3\n") != base
        assert ledger.code_sha256("a  = 1\nb = 2\n") != base  # inner whitespace is a change
        assert ledger.code_sha256("    a = 1\nb = 2\n") != base  # so is indentation


class TestAttempts:
    def test_counts_distinct_code_reaching_phase1_per_campaign(self):
        ledger.record_attempt("c1", "A", "sha1", 1)
        ledger.record_attempt("c1", "A", "sha1", 1, passed=False)  # same code again
        ledger.record_attempt("c1", "B", "sha2", 1, passed=True)
        ledger.record_attempt("c1", "B", "sha3", 2, passed=True)  # Phase 2 only doesn't count
        ledger.record_attempt("c2", "C", "sha4", 1)
        assert ledger.attempt_count("c1") == 2
        assert ledger.attempt_count("c2") == 1
        assert ledger.attempt_count("c3") == 0

    def test_finish_attempt_sets_passed(self):
        row = ledger.record_attempt("c1", "A", "sha1", 2)
        assert not ledger.phase2_passed("c1", "sha1")
        ledger.finish_attempt(row, True)
        assert ledger.phase2_passed("c1", "sha1")

    def test_phase2_passed_is_per_code_and_campaign(self):
        ledger.record_attempt("c1", "A", "sha1", 2, passed=False)
        ledger.record_attempt("c1", "A", "sha2", 2, passed=True)
        ledger.record_attempt("c1", "A", "sha3", 1, passed=True)
        assert not ledger.phase2_passed("c1", "sha1")
        assert ledger.phase2_passed("c1", "sha2")
        assert not ledger.phase2_passed("c1", "sha3")
        assert not ledger.phase2_passed("c2", "sha2")


class TestFinalTests:
    def test_done_by_code_or_by_name(self):
        ledger.record_final_test("c1", "sha1", "A", True, {"passed": True})
        assert ledger.final_test_done("c1", sha="sha1")
        assert ledger.final_test_done("c1", name="A")
        assert ledger.final_test_done("c1", sha="other", name="A")
        assert not ledger.final_test_done("c1", sha="sha2", name="B")
        assert not ledger.final_test_done("c2", sha="sha1", name="A")

    def test_unique_on_code_and_on_name(self):
        ledger.record_final_test("c1", "sha1", "A", False, {})
        with pytest.raises(ledger.FinalTestAlreadyRun):
            ledger.record_final_test("c1", "sha1", "B", True, {})  # renamed copy of the same code
        with pytest.raises(ledger.FinalTestAlreadyRun):
            ledger.record_final_test("c1", "sha2", "A", True, {})  # new code under a used name
        ledger.record_final_test("c2", "sha1", "A", True, {})  # a new campaign starts fresh

    def test_claim_then_record_fills_the_claim(self):
        ledger.claim_final_test("c1", "sha1", "A")
        assert ledger.final_test_done("c1", sha="sha1")
        with pytest.raises(ledger.FinalTestAlreadyRun):
            ledger.claim_final_test("c1", "sha1", "A")
        ledger.record_final_test("c1", "sha1", "A", True, {"passed": True, "stats": {"sharpe": 1.2}})
        rows = sqlite3.connect(ledger.ledger_path()).execute(
            "SELECT strategy_name, passed, result_json FROM final_tests").fetchall()
        assert len(rows) == 1
        assert rows[0][:2] == ("A", 1) and json.loads(rows[0][2])["stats"]["sharpe"] == 1.2

    def test_release_drops_only_an_unfinished_claim(self):
        ledger.claim_final_test("c1", "sha1", "A")
        ledger.release_final_test("c1", "sha1")
        assert not ledger.final_test_done("c1", sha="sha1", name="A")
        ledger.record_final_test("c1", "sha1", "A", False, {})
        ledger.release_final_test("c1", "sha1")
        assert ledger.final_test_done("c1", sha="sha1")
