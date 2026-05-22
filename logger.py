import jax
from typing import Dict, Any, Optional, Sequence


class Logger:
    def __init__(self, log_dir: Optional[str] = None, backend: str = "tensorboard"):
        self.backend = backend
        self.writer = None

        if self.backend == "tensorboard":
            from tensorboardX import SummaryWriter
            if log_dir is not None:
                self.writer = SummaryWriter(log_dir)
        else:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")

    def log_scalar(self, key: str, value: Any, step: int):
        if self.writer is not None:
            self.writer.add_scalar(key, float(value), int(step))

    def log_dict(self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = None):
        """Logs a dictionary of scalar metrics."""
        if self.writer is None:
            return

        for k, v in metrics.items():
            log_key = f"{prefix}/{k}" if prefix else k
            self.writer.add_scalar(log_key, float(v), int(step))

    def log_sequence(self, metrics: Dict[str, Sequence[Any]], start_step: int, interval: int, prefix: Optional[str] = None):
        """Logs lists/arrays of metrics over a sequence of steps."""
        if self.writer is None:
            return

        for k, values in metrics.items():
            log_key = f"{prefix}/{k}" if prefix else k
            for i, val in enumerate(values):
                step = start_step + (i + 1) * interval
                self.log_scalar(log_key, val, step)

    def print_summary(
        self,
        step: int,
        metrics: Dict[str, Any],
        signature: str = "Train",
        group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        use_jax_debug: bool = True
    ):
        """
        Dynamically prints metrics in a JAX-safe way based on grouping by keys.

        Args:
            step: Current training/eval step.
            metrics: Dictionary of metric names and values.
            signature: Main header signature (e.g., "Train", "Eval").
            group_configs: Nested dict defining custom headlines. A headline can be a string,
                           or a dict with "format" and "params".
        """
        grouped_metrics = {}
        print_kwargs = dict(metrics)
        print_kwargs["step"] = step

        # Grouping
        for k in metrics.keys():
            parts = k.split("/", 1)
            if len(parts) == 2:
                group, name = parts
            else:
                group, name = "General", k

            if group not in grouped_metrics:
                grouped_metrics[group] = []
            grouped_metrics[group].append((name, k))

        # Formatting
        lines = [f"\n=== Step {{step}}: {signature} ==="]

        for group, items in grouped_metrics.items():
            headline = f"--- {group.capitalize()} ---"

            # Apply custom config if provided
            if group_configs and group in group_configs:
                group_config = group_configs[group]

                if "headline" in group_config:
                    headline_data = group_config["headline"]

                    if isinstance(headline_data, dict):
                        headline = headline_data.get("format", headline)
                        if "params" in headline_data:
                            print_kwargs.update(headline_data["params"])

                    elif isinstance(headline_data, str):
                        headline = headline_data

            lines.append(f"\n{headline}")

            max_name_len = max([len(name) for name, _ in items]) if items else 0

            for name, k in items:
                lines.append(f"    {name + ':':<{max_name_len + 1}}  {{{k}}}")

        fmt_string = "\n".join(lines) + "\n"

        if use_jax_debug:
            jax.debug.print(fmt_string, **print_kwargs)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
