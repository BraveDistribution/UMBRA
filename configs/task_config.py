import json


class CONFIG:
    """Main configuration class that can be inherited later."""

    def __init__(self, task_config=None):
        self.devices: list[int] = [0]
        self.max_epochs: int = 300

        self.task_config = task_config
        if task_config:
            self.load_from_task_config(task_config)

    def load_from_task_config(self, task_config):
        """Load configuration from task_config file or dict."""
        if isinstance(task_config, str):
            with open(task_config, "r") as f:
                config_data = json.load(f)
        elif isinstance(task_config, dict):
            config_data = task_config
        else:
            raise ValueError("task_config must be a file path or dictionary")

        for key, value in config_data.items():
            setattr(self, key, value)
