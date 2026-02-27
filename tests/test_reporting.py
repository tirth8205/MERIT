"""Tests for reporting module."""
import pytest
import json
import csv
from pathlib import Path
from merit.reporting.tables import generate_results_table, generate_correlation_table
from merit.reporting.plots import radar_chart, scaling_plot, correlation_heatmap, cost_comparison_bar
from merit.reporting.export import export_json, export_csv


class TestResultsTable:
    def test_basic_table(self):
        data = {
            "tinyllama-1b": {"arc": {"consistency": 0.75, "factual": 0.60}},
            "phi-2": {"arc": {"consistency": 0.85, "factual": 0.72}},
        }
        latex = generate_results_table(data)
        assert r"\begin{table}" in latex
        assert "tinyllama" in latex
        assert "0.75" in latex
        assert "0.85" in latex
        assert r"\end{table}" in latex

    def test_empty_results(self):
        assert generate_results_table({}) == ""

    def test_custom_caption(self):
        data = {"model": {"ds": {"m": 0.5}}}
        latex = generate_results_table(data, caption="Custom Caption")
        assert "Custom Caption" in latex


class TestCorrelationTable:
    def test_basic_correlation(self):
        data = {
            "merit_heuristic": [0.8, 0.7, 0.9, 0.6, 0.85],
            "bertscore": [0.75, 0.65, 0.82, 0.55, 0.80],
        }
        annotations = [0.9, 0.7, 0.95, 0.5, 0.88]
        latex = generate_correlation_table(data, annotations)
        assert r"\begin{table}" in latex
        assert "merit" in latex
        assert r"$\rho$" in latex


class TestPlots:
    def test_radar_chart(self, tmp_path):
        scores = {
            "model_a": {"consistency": 0.8, "factual": 0.7, "reasoning": 0.9, "alignment": 0.85},
            "model_b": {"consistency": 0.6, "factual": 0.8, "reasoning": 0.7, "alignment": 0.9},
        }
        out = str(tmp_path / "radar.png")
        radar_chart(scores, out)
        assert Path(out).exists()

    def test_scaling_plot(self, tmp_path):
        sizes = [0.5, 1.1, 2.7, 7.0]
        scores = {
            "consistency": [0.5, 0.6, 0.7, 0.85],
            "factual": [0.4, 0.55, 0.65, 0.8],
        }
        out = str(tmp_path / "scaling.png")
        scaling_plot(sizes, scores, out)
        assert Path(out).exists()

    def test_correlation_heatmap(self, tmp_path):
        corr = {
            "consistency": {"consistency": 1.0, "factual": 0.6},
            "factual": {"consistency": 0.6, "factual": 1.0},
        }
        out = str(tmp_path / "heatmap.png")
        correlation_heatmap(corr, out)
        assert Path(out).exists()

    def test_cost_comparison(self, tmp_path):
        out = str(tmp_path / "cost.png")
        cost_comparison_bar(
            methods=["Heuristic", "LLM-Judge"],
            times=[0.1, 2.5],
            costs=[0.0, 0.01],
            output_path=out,
        )
        assert Path(out).exists()


class TestExport:
    def test_export_json(self, tmp_path):
        data = {"model": "test", "score": 0.85}
        path = str(tmp_path / "out.json")
        export_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["score"] == 0.85

    def test_export_csv(self, tmp_path):
        results = {
            "model_a": {"ds1": {"metric1": 0.8, "metric2": 0.7}},
            "model_b": {"ds1": {"metric1": 0.6, "metric2": 0.9}},
        }
        path = str(tmp_path / "out.csv")
        export_csv(results, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["model"] == "model_a"

    def test_export_csv_empty(self, tmp_path):
        path = str(tmp_path / "empty.csv")
        export_csv({}, path)
        assert not Path(path).exists()  # Empty results don't create a file
