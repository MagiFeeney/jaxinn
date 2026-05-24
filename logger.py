import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, Sequence, Literal

import equinox as eqx


default_group_configs = {
    "eval": {
        "headline": {
            "format": "--- Evaluation ({n} episodes) ---",
            "params": {"n": "unknown"} # Default fallback placeholder
        }
    },
    "ac": {
        "headline": {
            "format": "--- Actor and Critic Loss ---"
        }
    },
    "model": {
        "headline": {
            "format": "--- Model Loss ---"
        }
    },
    "aux": {
        "headline": {
            "format": "--- Auxiliary ---"
        }
    }
}


class HostLogger:
    """
    A standard Python logger, handling I/O
    """

    def __init__(
            self,
            log_dir: Optional[str] = None,
            backend: str = "tensorboard",
            aggregate_keywords: tuple = ("eval", "test", "reward", "return"),
            shaded_method: Literal["std", "se", "ci", "iqr"] = "std"
    ):
        if backend == "tensorboard":
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir) if log_dir is not None else SummaryWriter()
        else:
            raise NotImplementedError(f"Backend '{backend}' is not supported.")

        self.aggregate_keywords = aggregate_keywords
        self.shaded_method = shaded_method.lower()

        self.layout = {"Aggregated Metrics (Shaded)": {}}
        self.known_keys = set()

    def compute_bounds(self, x: np.ndarray) -> tuple[float, float, float]:
        mean = np.mean(x)

        if self.shaded_method == "std":
            # Standard deviation
            offset = np.std(x)
            return mean, mean - offset, mean + offset
        elif self.shaded_method == "se":
            # Standard error
            offset = np.std(x, ddof=1) / np.sqrt(x.size)
            return mean, mean - offset, mean + offset
        elif self.shaded_method == "iqr":
            # Interquartile range
            return mean, np.percentile(x, 25), np.percentile(x, 75)
        elif self.shaded_method == "ci":
            # 95% Confidence Interval
            stderr = np.std(x, ddof=1) / np.sqrt(x.size)
            offset = 1.96 * stderr
            return mean, mean - offset, mean + offset
        else:
            raise ValueError(f"Unknown shaded method: {self.shaded_method}")

    def register_layout(self, k: str):
        """Update layout if a new key is first time seen."""
        if k not in self.known_keys:
            self.known_keys.add(k)

            self.layout["Aggregated Metrics (Shaded)"][k] = [
                "Margin",
                [f"shaded/{k}/mean", f"shaded/{k}/lower", f"shaded/{k}/upper"]
            ]
            self.writer.add_custom_scalars(self.layout)

    def log_dict(self, metrics: Dict[str, Any], step: Any):
        if self.writer is None:
            return

        s_arr = np.asarray(step).ravel()
        if s_arr.size > 1 and not np.all(s_arr == s_arr[0]):
            raise ValueError(f"log: 'step' must be identical across seeds, got {s_arr}.")
        s = int(s_arr[0])

        for k, v in metrics.items():
            val = np.asarray(v)
            if val.ndim == 0:
                self.writer.add_scalar(k, float(val), s)
            else:
                self.writer.add_scalars(
                    k,
                    {f"seed_{'_'.join(map(str, idx))}": float(x) for idx, x in np.ndenumerate(val)},
                    s
                )

                should_aggregate = (
                    val.size > 1 and
                    any(keyword in k.lower() for keyword in self.aggregate_keywords)
                )

                if should_aggregate:
                    self.register_layout(k)

                    mean, lower, upper = self.compute_bounds(val)

                    self.writer.add_scalar(f"shaded/{k}/mean", float(mean), s)
                    self.writer.add_scalar(f"shaded/{k}/lower", float(lower), s)
                    self.writer.add_scalar(f"shaded/{k}/upper", float(upper), s)

    def log_sequence(self, metrics: Dict[str, Any], start_step: Any, interval: Any):
        if self.writer is None:
            return

        st = int(np.asarray(start_step).ravel()[0])
        inv = int(np.asarray(interval).ravel()[0])

        for k, values in metrics.items():
            arr = np.asarray(values)
            if arr.ndim == 1:
                for i, val in enumerate(arr):
                    self.log_dict({k: val}, st + (i + 1) * inv)
            elif arr.ndim == 2:
                for i, val in enumerate(arr.T): # Transpose to time axis so that the logging is correct
                    self.log_dict({k: val}, st + (i + 1) * inv)
            else:
                raise ValueError(f"log_sequence: '{k}' shape must be (T,) or (N, T).")

    def print_summary(
            self,
            step: int,
            metrics: Dict[str, Any],
            signature: str = "Train",
            headline_params: Optional[Dict[str, Any]] = None,
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        fmt_string, static_params = self.get_summary_template(
            metric_keys=list(metrics.keys()),
            signature=signature,
            group_configs=group_configs
        )

        print_kwargs = dict(metrics)
        print_kwargs["step"] = int(np.asarray(step).ravel()[0])
        print_kwargs.update(static_params)
        if headline_params:
            print_kwargs.update(headline_params)

        print(fmt_string.format(**print_kwargs))

    @classmethod
    def get_summary_template(
            cls,
            metric_keys: Sequence[str],
            signature: str = "Train",
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Pure Python formatter. Generates the template string and static parameters.
        Runs ONCE at trace time for JAX, or at runtime for Eager mode.
        """
        grouped_metrics = {}
        static_params = {}

        # Grouping
        for k in metric_keys:
            parts = k.split("/", 1)
            group, name = parts if len(parts) == 2 else ("General", k)
            if group not in grouped_metrics:
                grouped_metrics[group] = []
            grouped_metrics[group].append((name, k))

        lines = [f"\n=== Step {{step}}: {signature} ==="]
        merged_group_configs = cls.merge_configs(default_group_configs, group_configs)

        # Formatting
        for group, items in grouped_metrics.items():
            fallback_headline = f"--- {group.capitalize()} ---"
            headline, print_kwargs_update = cls.get_headline(merged_group_configs, group, fallback_headline)
            static_params.update(print_kwargs_update)
            lines.append(f"\n{headline}")

            max_name_len = max([len(name) for name, _ in items]) if items else 0
            for name, k in items:
                lines.append(f"    {name + ':':<{max_name_len + 1}}  {{{k}}}")

        fmt_string = "\n".join(lines) + "\n"
        return fmt_string, static_params

    @staticmethod
    def get_headline(configs: Dict[str, Any], current_group: str, fallback: str) -> Tuple[str, Dict[str, Any]]:
        print_kwargs_update = {}
        headline = fallback

        if configs and current_group in configs:
            group_config = configs[current_group]
            if "headline" in group_config:
                headline_data = group_config["headline"]
                if isinstance(headline_data, dict):
                    headline = headline_data.get("format", headline)
                    if "params" in headline_data:
                        print_kwargs_update = headline_data["params"]
                elif isinstance(headline_data, str):
                    headline = headline_data

        return headline, print_kwargs_update

    @staticmethod
    def merge_configs(base: Dict[str, Any], custom: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = {k: v.copy() if isinstance(v, dict) else v for k, v in base.items()}
        if custom:
            for group, config in custom.items():
                if group in merged and isinstance(merged[group], dict) and isinstance(config, dict):
                    merged[group] = {**merged[group], **config}
                else:
                    merged[group] = config
        return merged

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


class LoggerJaxConverter(eqx.Module):
    """
    A bridge from JAX to Python via `jax.debug.callback`.
    """

    host_logger: HostLogger = eqx.field(static=True)

    def __init__(self, host_logger: HostLogger):
        self.host_logger = host_logger

    def _log_dict(self, metrics: Dict[str, Any], step: Any):
        return jax.debug.callback(self.host_logger.log_dict, metrics, step, ordered=True)

    def _log_sequence(self, metrics: Dict[str, Any], start_step: Any, interval: Any):
        return jax.debug.callback(self.host_logger.log_sequence, metrics, start_step, interval, ordered=True)

    def print_summary(
            self,
            step: int,
            metrics: Dict[str, Any],
            signature: str = "Train",
            headline_params: Optional[Dict[str, Any]] = None,
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        fmt_string, static_params = self.host_logger.get_summary_template(
            metric_keys=list(metrics.keys()),
            signature=signature,
            group_configs=group_configs
        )

        print_kwargs = dict(metrics)
        print_kwargs["step"] = step
        print_kwargs.update(static_params)
        if headline_params:
            print_kwargs.update(headline_params)

        jax.debug.print(fmt_string, **print_kwargs)

    def flush(self):
        self.host_logger.flush()

    def close(self):
        self.host_logger.close()


class LoggerVmapMixIn:
    """A mixin with custom `vmap` rules that intercept batched data."""

    @jax.custom_batching.custom_vmap
    def v_log_dict(self, metrics, step):
        self._log_dict(metrics, step)
        return ()

    @v_log_dict.def_vmap
    def v_log_dict_batch(axis_size, in_batched, self, metrics, step):
        self._log_dict(metrics, step)
        return (), ()

    @jax.custom_batching.custom_vmap
    def v_log_sequence(self, metrics, start_step, interval):
        self._log_sequence(metrics, start_step, interval)
        return ()

    @v_log_sequence.def_vmap
    def v_log_sequence_batch(axis_size, in_batched, self, metrics, start_step, interval):
        self._log_sequence(metrics, start_step, interval)
        return (), ()


class JaxLogger(LoggerJaxConverter, LoggerVmapMixIn):
    """A JAX-safe user interface."""

    @classmethod
    def create(cls, log_dir: Optional[str] = None, backend: str = "tensorboard"):
        host_logger = HostLogger(log_dir=log_dir, backend=backend)
        return cls(host_logger=host_logger)

    def log_scalar(self, key: str, value: Any, step: int) -> None:
        self.v_log_dict(self, {key: value}, step)

    def log_dict(self, metrics: Dict[str, Any], step: int, prefix: Optional[str] = None) -> None:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        self.v_log_dict(self, metrics, step)

    def log_sequence(self, metrics: Dict[str, Sequence[Any]], start_step: int, interval: int, prefix: Optional[str] = None) -> None:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        metrics = {k: jnp.asarray(v) for k, v in metrics.items()}
        self.v_log_sequence(self, metrics, start_step, interval)

    def __enter__(self) -> "JaxLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
