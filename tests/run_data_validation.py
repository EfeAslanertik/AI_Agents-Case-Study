import json
import pandas as pd
from agents.data_quality_validation import DataQualityValidationAgent

# Load dataset
df = pd.read_csv("datasets/Amazon.csv")

agent = DataQualityValidationAgent()

# Run agent
raw_report = agent.run(df)
formatted_report = agent.format_report(raw_report)

# Save outputs
with open("outputs/validation_report_raw.json", "w") as f:
    json.dump(raw_report, f, indent=4)

with open("outputs/validation_report_formatted.json", "w") as f:
    json.dump(formatted_report, f, indent=4)

print("Validation completed. Reports saved to outputs/")
