import pandas as pd
import numpy as np
from typing import Dict, Any


class DataQualityValidationAgent:
    """
    Validates dataset quality before downstream processing.
    Checks for missing values, type consistency, and anomalies.
    Produces a structured validation report with severity flags.
    """

    def __init__(
        self,
        missing_warn_threshold: float = 0.10,
        missing_critical_threshold: float = 0.30,
        outlier_iqr_multiplier: float = 1.5,
    ):
        self.missing_warn_threshold = missing_warn_threshold
        self.missing_critical_threshold = missing_critical_threshold
        self.outlier_iqr_multiplier = outlier_iqr_multiplier

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return self._empty_dataset_report()

        report = {
            "dataset_summary": self._dataset_summary(df),
            "missing_values": self._check_missing_values(df),
            "type_issues": self._check_type_consistency(df),
            "anomalies": self._detect_outliers(df),
        }

        severity, blocking_issues = self._compute_severity(report)
        report["severity"] = severity
        report["blocking_issues"] = blocking_issues

        return report

    # ----------------------- Checks -----------------------

    def _dataset_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "columns": list(df.columns),
        }

    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            col: round(df[col].isnull().mean(), 4)
            for col in df.columns
            if df[col].isnull().any()
        }

    def _check_type_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        issues = {}

        for col in df.columns:
            series = df[col]

            # Numeric columns with non-numeric values
            if pd.api.types.is_numeric_dtype(series):
                non_numeric_count = series.apply(
                    lambda x: not isinstance(x, (int, float, np.number)) and not pd.isnull(x)
                ).sum()
                if non_numeric_count > 0:
                    issues[col] = "Numeric column contains non-numeric values"

            # Object columns that look numeric but aren't
            elif pd.api.types.is_object_dtype(series):
                try:
                    pd.to_numeric(series.dropna())
                except Exception:
                    unique_types = series.dropna().map(type).nunique()
                    if unique_types > 1:
                        issues[col] = "Mixed data types in column"

        return issues

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        outliers = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - self.outlier_iqr_multiplier * iqr
            upper = q3 + self.outlier_iqr_multiplier * iqr

            count = int(((series < lower) | (series > upper)).sum())
            if count > 0:
                outliers[col] = {
                    "method": "IQR",
                    "count": count,
                    "percentage": round(count / len(series), 4),
                }

        return outliers

    # ----------------------- Severity -----------------------

    def _compute_severity(self, report: Dict[str, Any]):
        blocking_issues = []

        for col, pct in report["missing_values"].items():
            if pct >= self.missing_critical_threshold:
                blocking_issues.append(
                    f"Column '{col}' missing {pct*100:.1f}% of values"
                )

        if blocking_issues:
            return "HIGH", blocking_issues

        if report["type_issues"] or report["anomalies"]:
            return "MEDIUM", []

        return "LOW", []

    # ----------------------- Edge Case -----------------------

    def _empty_dataset_report(self):
        return {
            "dataset_summary": {
                "num_rows": 0,
                "num_columns": 0,
                "columns": [],
            },
            "missing_values": {},
            "type_issues": {},
            "anomalies": {},
            "severity": "HIGH",
            "blocking_issues": ["Dataset is empty"],
        }
    
    # ----------------------- Formatting -----------------------

    def format_report(self, raw_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the raw validation output into a human-readable,
        presentation-friendly structure.
        """
        return {
            "status": self._format_status(raw_report),
            "dataset_overview": self._format_dataset_overview(raw_report),
            "data_quality": self._format_data_quality(raw_report),
            "anomaly_analysis": self._format_anomalies(raw_report),
            "blocking_issues": raw_report["blocking_issues"],
            "recommendations": self._generate_recommendations(raw_report),
        }

    def _format_status(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "severity": report["severity"],
            "blocking": len(report["blocking_issues"]) > 0,
            "summary": self._severity_summary(report),
        }

    def _format_dataset_overview(self, report: Dict[str, Any]) -> Dict[str, Any]:
        summary = report["dataset_summary"]
        return {
            "rows": summary["num_rows"],
            "columns": summary["num_columns"],
            "column_names": summary["columns"],
        }

    def _format_data_quality(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "missing_values": {
                "columns_with_missing": len(report["missing_values"]),
                "details": report["missing_values"],
            },
            "type_issues": {
                "count": len(report["type_issues"]),
                "details": report["type_issues"],
            },
        }

    def _format_anomalies(self, report: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "method": "IQR",
            "summary": (
                "Minor statistical outliers detected"
                if report["anomalies"]
                else "No anomalies detected"
            ),
            "details": report["anomalies"],
        }

    def _severity_summary(self, report: Dict[str, Any]) -> str:
        if report["severity"] == "HIGH":
            return "Critical data quality issues detected. Dataset is not safe for modeling."
        if report["severity"] == "MEDIUM":
            return "Dataset is usable with minor anomalies that should be reviewed."
        return "Dataset meets all quality checks."

    def _generate_recommendations(self, report: Dict[str, Any]):
        recommendations = []

        if report["blocking_issues"]:
            recommendations.append("Resolve blocking data quality issues before modeling.")
        else:
            recommendations.append("Proceed with downstream modeling.")

        if report["anomalies"]:
            recommendations.append(
                "Review anomaly-prone numerical features for business validity."
            )

        if not report["missing_values"]:
            recommendations.append("No missing-value preprocessing required.")

        if not report["type_issues"]:
            recommendations.append("No schema or type corrections needed.")

        return recommendations

