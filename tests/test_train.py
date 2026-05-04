"""
test_train.py — Unit tests cho các hàm trong train.py
Chạy: pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Thêm src vào path để import train module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import train  # noqa: E402


# ── Test downcast ────────────────────────────────────────────────────────────
class TestDowncast:
    def test_downcast_reduces_int_memory(self):
        """downcast phải giảm memory cho cột integer."""
        df = pd.DataFrame({"a": np.array([1, 2, 3], dtype="int64")})
        before = df.memory_usage(deep=True).sum()
        df = train.downcast(df)
        after = df.memory_usage(deep=True).sum()
        assert after <= before, "Memory should decrease or stay the same after downcast"

    def test_downcast_reduces_float_memory(self):
        """downcast phải giảm memory cho cột float."""
        df = pd.DataFrame({"a": np.array([1.0, 2.5, 3.7], dtype="float64")})
        before = df.memory_usage(deep=True).sum()
        df = train.downcast(df)
        after = df.memory_usage(deep=True).sum()
        assert after <= before

    def test_downcast_preserves_values(self):
        """downcast không được thay đổi giá trị."""
        original = [1, 200, 30000]
        df = pd.DataFrame({"a": np.array(original, dtype="int64")})
        df = train.downcast(df)
        assert df["a"].tolist() == original


# ── Test parse_args ──────────────────────────────────────────────────────────
class TestParseArgs:
    def test_default_values(self, monkeypatch):
        """parse_args phải trả về giá trị mặc định đúng."""
        monkeypatch.setattr("sys.argv", ["train.py"])
        args = train.parse_args()
        assert args.n_estimators == 1500
        assert args.learning_rate == 0.05
        assert args.max_depth == 8
        assert args.num_leaves == 64
        assert args.random_state == 42

    def test_custom_values(self, monkeypatch):
        """parse_args phải parse tham số tuỳ chỉnh đúng."""
        monkeypatch.setattr("sys.argv", [
            "train.py",
            "--n_estimators", "500",
            "--learning_rate", "0.01",
            "--data_dir", "/tmp/data",
        ])
        args = train.parse_args()
        assert args.n_estimators == 500
        assert args.learning_rate == 0.01
        assert args.data_dir == "/tmp/data"


# ── Test evaluate ────────────────────────────────────────────────────────────
class TestEvaluate:
    def test_evaluate_returns_all_metrics(self):
        """evaluate phải trả về dict chứa mae và rmse."""
        # Mock model đơn giản
        class MockModel:
            def predict(self, X):
                return np.zeros(len(X))

        X_test = pd.DataFrame({col: [0.0] * 10 for col in train.FEAT_COLS})
        y_test = pd.Series(np.ones(10))  # actual = 1, predict = 0 → MAE = 1

        metrics = train.evaluate(MockModel(), X_test, y_test)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["mae"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)

    def test_evaluate_perfect_prediction(self):
        """Khi predict chính xác, tất cả metrics = 0."""
        class PerfectModel:
            def predict(self, X):
                return np.array([1.0, 2.0, 3.0])

        X_test = pd.DataFrame({col: [0.0] * 3 for col in train.FEAT_COLS})
        y_test = pd.Series([1.0, 2.0, 3.0])

        metrics = train.evaluate(PerfectModel(), X_test, y_test)

        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)


# ── Test feature columns consistency ─────────────────────────────────────────
class TestFeatureConsistency:
    def test_feat_cols_match_score(self):
        """FEAT_COLS trong train.py phải khớp với score.py."""
        # Import score module
        import score

        assert train.FEAT_COLS == score.FEAT_COLS, (
            f"FEAT_COLS mismatch!\n"
            f"  train.py: {train.FEAT_COLS}\n"
            f"  score.py: {score.FEAT_COLS}"
        )

    def test_cat_cols_match_score(self):
        """CAT_COLS trong train.py phải khớp với score.py."""
        import score

        assert train.CAT_COLS == score.CAT_COLS
