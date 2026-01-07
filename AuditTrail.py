import json
from datetime import datetime
import os

class AuditTrailAgent:
    def __init__(self, log_file="audit_log.json"):
        """
        Initializes the Audit Trail Agent.
        :param log_file: Path to the JSON log file.
        """
        self.log_file = log_file
        # Create log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f, indent=4)

    def log_action(self, agent_name, action_type, inputs=None, outputs=None, notes=None):
        """
        Logs an agent's action.
        :param agent_name: Name of the agent performing the action.
        :param action_type: Type of action (e.g., "prediction", "validation", "summary").
        :param inputs: Inputs to the agent (can be dict, list, etc.).
        :param outputs: Outputs from the agent.
        :param notes: Optional notes or context.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "action_type": action_type,
            "inputs": inputs,
            "outputs": outputs,
            "notes": notes
        }

        # Append log entry to JSON file
        with open(self.log_file, "r+") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)

    def query_logs(self, agent_name=None, action_type=None):
        """
        Query logs for a specific agent or action type.
        :param agent_name: Filter by agent name.
        :param action_type: Filter by action type.
        :return: List of matching log entries.
        """
        with open(self.log_file, "r") as f:
            data = json.load(f)

        # Apply filters
        if agent_name:
            data = [d for d in data if d["agent_name"] == agent_name]
        if action_type:
            data = [d for d in data if d["action_type"] == action_type]

        return data
