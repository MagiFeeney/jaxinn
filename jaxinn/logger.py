import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional, Sequence, Literal, Callable

from rich import print as rprint
import equinox as eqx


class FormatParams(eqx.Module):
    signature: str = eqx.field(static=True, default="Train")
    headline_params: Optional[Dict] = eqx.field(static=True, default=None)
    group_configs: Optional[Dict] = eqx.field(static=True, default=None)


default_group_configs = {
    "eval": {
        "headline": {
            "format": "━━━ Evaluation ({n} episodes) ━━━",
            "params": {"n": "unknown"} # Default fallback placeholder
        }
    },
    "ac": {
        "headline": {
            "format": "━━━ Actor and Critic Loss ━━━"
        }
    },
    "model": {
        "headline": {
            "format": "━━━ Model Loss ━━━"
        }
    },
    "aux": {
        "headline": {
            "format": "━━━ Auxiliary ━━━"
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
            aggregate_keywords: tuple = ("eval", "test"),
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

    def _compute_bounds(self, x: np.ndarray) -> tuple[float, float, float]:
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

    def _register_layout(self, k: str):
        """Update layout if a new key is first time seen."""
        if k not in self.known_keys:
            self.known_keys.add(k)

            self.layout["Aggregated Metrics (Shaded)"][k] = [
                "Margin",
                [f"shaded/{k}/mean", f"shaded/{k}/lower", f"shaded/{k}/upper"]
            ]
            self.writer.add_custom_scalars(self.layout)

    def log_dict(self, metrics: Dict[str, Any], step: Any) -> None:
        if self.writer is None:
            return

        s_arr = np.asarray(step).ravel()
        if s_arr.size > 1 and not np.all(s_arr == s_arr[0]):
            raise ValueError(f"log: 'step' must be identical across seeds, got {s_arr}.")
        s = int(s_arr[0])

        for k, v in metrics.items():
            val = np.asarray(v)

            if val.ndim > 1:
                raise ValueError(f"log_dict: '{k}' has shape {val.shape}. Expected scalar or 1D.")

            if val.ndim == 0:
                self.writer.add_scalar(k, float(val), s)
            else:
                seed_val = {f"seed_{'_'.join(map(str, idx))}": float(x) for idx, x in np.ndenumerate(val)}
                self.writer.add_scalars(k, seed_val, s)

                should_aggregate = (
                    val.size > 1 and
                    any(keyword in k.lower() for keyword in self.aggregate_keywords)
                )

                if should_aggregate:
                    self._register_layout(k)

                    mean, lower, upper = self._compute_bounds(val)

                    self.writer.add_scalar(f"shaded/{k}/mean", float(mean), s)
                    self.writer.add_scalar(f"shaded/{k}/lower", float(lower), s)
                    self.writer.add_scalar(f"shaded/{k}/upper", float(upper), s)

    def log_sequence(self, metrics: Dict[str, Any], start_step: Any, interval: Any) -> None:
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
            metrics: Dict[str, Any],
            step: int,
            signature: str = "Train",
            headline_params: Optional[Dict[str, Any]] = None,
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        metric_shapes = {}
        print_kwargs = {}

        for k, v in metrics.items():
            shape = getattr(v, "shape", ())
            metric_shapes[k] = shape

            if v.ndim == 2:
                for i, row in enumerate(v):
                    print_kwargs[f"{k}_seed_{i}"] = np.array2string(
                        row, precision=4, separator=' ', suppress_small=True
                    )
            elif v.ndim == 1:
                print_kwargs[f"{k}_mean"] = float(v.mean())
                print_kwargs[f"{k}_std"] = float(v.std())
                print_kwargs[f"{k}_raw"] = np.array2string(
                    v, precision=4, separator=', ', suppress_small=True
                )
            else:
                print_kwargs[k] = float(v) if hasattr(v, "item") else v

        fmt_string, static_params = self._get_summary_template(
            metric_shapes=metric_shapes,
            signature=signature,
            group_configs=group_configs
        )

        print_kwargs["step"] = int(np.asarray(step).ravel()[0])
        print_kwargs.update(static_params)

        if headline_params:
            print_kwargs.update(headline_params)

        rprint(fmt_string.format(**print_kwargs))

    @classmethod
    def _get_summary_template(
            cls,
            metric_shapes: Dict[str, Tuple[int, ...]],
            signature: str = "Train",
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
            group_order: Sequence[str] = ("model", "ac", "auxiliary", "evaluation")
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Pure Python formatter. Generates the template string and static parameters.
        Runs ONCE at trace time for JAX, or at runtime for Eager mode.
        """
        grouped_metrics = {}
        static_params = {}

        # Grouping
        for k, shape in metric_shapes.items():
            parts = k.split("/", 1)
            group, name = parts if len(parts) == 2 else ("General", k)
            if group not in grouped_metrics:
                grouped_metrics[group] = []
            grouped_metrics[group].append((name, k, shape))

        lines = [f"\n[bold black on magenta]  Step {{step}}: {signature}  [/bold black on magenta]"]
        merged_group_configs = cls._merge_configs(default_group_configs, group_configs)

        sorted_group_metrics = sorted(
            grouped_metrics.items(),
            key=lambda x: group_order.index(x[0].lower()) if x[0].lower() in group_order else float('inf')
        )

        # Formatting
        for group, items in sorted_group_metrics:
            fallback_headline = f"━━━ {group.capitalize()} ━━━"
            headline, print_kwargs_update = cls._get_headline(merged_group_configs, group, fallback_headline)
            static_params.update(print_kwargs_update)
            lines.append(f"\n[bold yellow]{headline}[/bold yellow]")

            for name, k, shape in items:
                if len(shape) == 2:
                    num_seeds = shape[0]
                    pad_len = len(str(num_seeds - 1))
                    lines.append(f"    [green]{name}:[/green]")
                    for i in range(num_seeds):
                        lines.append(f"        [cyan]Seed {i:>{pad_len}}:[/cyan] [bold white]{{{k}_seed_{i}}}[/bold white]")

                elif len(shape) == 1:
                    lines.append(
                        f"    [green]{name}:[/green] "
                        f"[bold red]{{{k}_mean:.4f}} ± {{{k}_std:.4f}}[/bold red]  "
                        f"[dim white](Raw: {{{k}_raw}})[/dim white]"
                    )
                else:
                    lines.append(f"    [green]{name}:[/green] [bold white]{{{k}}}[/bold white]")

        fmt_string = "\n".join(lines) + "\n"
        return fmt_string, static_params

    @staticmethod
    def _get_headline(configs: Dict[str, Any], current_group: str, fallback: str) -> Tuple[str, Dict[str, Any]]:
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
    def _merge_configs(base: Dict[str, Any], custom: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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

    axis_name: str = eqx.field(static=True)
    host_logger: HostLogger = eqx.field(static=True)

    @property
    def _in_pmap(self) -> bool:
        return bool(self.axis_name)

    def _get_device_idx(self) -> Any:
        if not self._in_pmap:
            return 0
        return jax.lax.axis_index(self.axis_name)

    def _gather_all(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not self._in_pmap:
            return metrics
        gathered = jax.lax.all_gather(metrics, axis_name=self.axis_name)
        return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), gathered)

    def _dispatch(
        self,
        host_fn: Callable,
        metrics: Dict[str, Any],
        *args: Any,
    ) -> None:
        """Gather metrics from devices and invoke `host_fn` on device 0 only."""
        gathered = self._gather_all(metrics)
        is_primary = self._get_device_idx() == 0
        jax.lax.cond(
            is_primary,
            lambda: jax.debug.callback(host_fn, gathered, *args, ordered=False),
            lambda: None,
        )

    def _log_dict(self, metrics: Dict[str, Any], step: Any) -> None:
        self._dispatch(self.host_logger.log_dict, metrics, step)

    def _log_sequence(self, metrics: Dict[str, Any], start_step: Any, interval: Any) -> None:
        self._dispatch(self.host_logger.log_sequence, metrics, start_step, interval)

    def _print_summary(
            self,
            metrics: Dict[str, Any],
            step: int,
            format_params: FormatParams,
    ) -> None:
        signature = format_params.signature
        headline_params = format_params.headline_params
        group_configs = format_params.group_configs
        self._dispatch(self.host_logger.print_summary, metrics, step, signature, headline_params, group_configs)

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
        s = step[0] if in_batched[2] else step
        self._log_dict(metrics, s)
        return (), ()

    @jax.custom_batching.custom_vmap
    def v_log_sequence(self, metrics, start_step, interval):
        self._log_sequence(metrics, start_step, interval)
        return ()

    @v_log_sequence.def_vmap
    def v_log_sequence_batch(axis_size, in_batched, self, metrics, start_step, interval):
        st = start_step[0] if in_batched[2] else start_step
        inv = interval[0] if in_batched[3] else interval
        self._log_sequence(metrics, st, inv)
        return (), ()

    @jax.custom_batching.custom_vmap
    def v_print_summary(self, metrics, step, format_params: FormatParams):
        self._print_summary(metrics, step, format_params)
        return ()

    @v_print_summary.def_vmap
    def v_print_summary_batch(axis_size, in_batched, self, metrics, step, format_params: FormatParams):
        s = step[0] if in_batched[2] else step
        self._print_summary(metrics, s, format_params)
        return (), ()


class JaxLogger(LoggerJaxConverter, LoggerVmapMixIn):
    """A JAX-safe user interface."""

    @classmethod
    def create(
            cls,
            log_dir: Optional[str] = None,
            backend: str = "tensorboard",
            aggregate_keywords: tuple = ("eval", "test"),
            shaded_method: Literal["std", "se", "ci", "iqr"] = "std",
            *,
            axis_name: str = "",
    ):
        host_logger = HostLogger(
            log_dir=log_dir,
            backend=backend,
            aggregate_keywords=aggregate_keywords,
            shaded_method=shaded_method
        ) # For LoggerJaxConverter
        return cls(axis_name=axis_name, host_logger=host_logger)

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

    def print_summary(
            self,
            metrics: Dict[str, Any],
            step: int,
            signature: str = "Train",
            headline_params: Optional[Dict[str, Any]] = None,
            group_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        # Close over the non-JAX arguments
        format_params = FormatParams(
            signature=signature,
            headline_params=headline_params,
            group_configs=group_configs
        )
        self.v_print_summary(self, metrics, step, format_params)

    def __enter__(self) -> "JaxLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
