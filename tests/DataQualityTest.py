import json
import pandas as pd
from agents.data_quality_validation import DataQualityValidationAgent


df = pd.read_csv("datasets/heart_disease_uci.csv")
df.replace("?", pd.NA, inplace=True)

agent = DataQualityValidationAgent()
raw_report = agent.run(df)
formatted_report = agent.format_report(raw_report)

from pprint import pprint
pprint(raw_report)

# Can also print out the formatted_report if desired for better readability
# pprint(formatted_report)