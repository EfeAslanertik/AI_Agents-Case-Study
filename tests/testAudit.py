from agents.audit_trail import AuditTrailAgent

# -------------------------------
# Initialize the Audit Trail Agent
# -------------------------------
audit_agent = AuditTrailAgent("test_audit_log.json")

# -------------------------------
# Log sample actions from different agents
# -------------------------------

# Example 1: Model Output Summarizer Agent
audit_agent.log_action(
    agent_name="ModelOutputSummarizer",
    action_type="summary",
    inputs={
        "predictions": [0.85, 0.30, 0.60],
        "shap_values": [
            {"age": 0.3, "chol": 0.5, "sex": -0.1},
            {"age": -0.1, "chol": -0.2, "sex": 0.05},
            {"age": 0.2, "chol": 0.25, "sex": -0.05}
        ]
    },
    outputs=[
        "Prediction 1: High risk (0.85)...",
        "Prediction 2: Low risk (0.30)...",
        "Prediction 3: Moderate risk (0.60)..."
    ],
    notes="Test run with sample SHAP values"
)

# Example 2: Data Quality Validation Agent
audit_agent.log_action(
    agent_name="DataQualityValidator",
    action_type="validation",
    inputs={"dataset": "heart_disease_uci.csv"},
    outputs={
        "severity": "HIGH",
        "blocking_issues": ["ca missing 66%", "thal missing 52%"]
    },
    notes="Initial data validation for heart disease dataset"
)

# Example 3: Another sample action
audit_agent.log_action(
    agent_name="RandomAgent",
    action_type="custom_action",
    inputs={"param": 42},
    outputs={"result": "Success"},
    notes="Testing custom action logging"
)

# -------------------------------
# Query and print logs
# -------------------------------

print("---- All Logs ----")
all_logs = audit_agent.query_logs()
for entry in all_logs:
    print(entry)
    print("-" * 50)

print("---- Logs for ModelOutputSummarizer ----")
summarizer_logs = audit_agent.query_logs(agent_name="ModelOutputSummarizer")
for entry in summarizer_logs:
    print(entry)
    print("-" * 50)

print("---- Logs for action_type='validation' ----")
validation_logs = audit_agent.query_logs(action_type="validation")
for entry in validation_logs:
    print(entry)
    print("-" * 50)
