from dataclasses import dataclass, fields, asdict, is_dataclass, MISSING
from typing import Any
from types import SimpleNamespace
from functools import cache


class ConfigNamespace(SimpleNamespace):
    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def __contains__(self, key):
        return key in vars(self)

    def __iter__(self):
        return iter(vars(self))

    def __getitem__(self, key):
        return vars(self)[key]

    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.items()]
        return f"Namespace({', '.join(items)})"


@dataclass
class Resolvable:
    """Depth-first recursive resolution."""

    def resolve(self, ctx: dict) -> "Resolvable":
        """Check the child nodes."""
        for f in fields(self):
            child = getattr(self, f.name)
            if isinstance(child, Resolvable):
                child.resolve(ctx)
        self._resolve(ctx)
        return self

    def _resolve(self, ctx: dict) -> None:
        """Resolve current level."""
        pass


@dataclass
class Base:
    """
    Nested configuration management.

    If there is sub-config, return the arguments at the current level excluding those from sub-configs.
    """
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)     # Preserve hierarchy

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not is_dataclass(v)} # Sub-node is considered as next phase

    def update(self, updates: dict[str, Any] = None, **kwargs) -> None:
        """Update or add config attributes. Allows adding new attributes."""
        if updates is None:
            updates = kwargs
        else:
            updates.update(kwargs)

        for key, value in updates.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if is_dataclass(attr) and isinstance(value, dict):
                    attr.update(value)
                else:
                    setattr(self, key, value)
            else:
                # Add new attribute that doesn't exist
                setattr(self, key, value)


def _sync_statics(root_node: any) -> None:
    """Two-pass structural sweep to sync all StaticShared dataclasses."""

    @cache
    def get_static_fields(cls: type, marker: type) -> frozenset:
        """Recursively finds fields from classes that directly inherit the marker."""

        if marker in cls.__bases__:
            return frozenset(f.name for f in fields(cls))

        static_fields = set()
        for base in cls.__bases__:
            if is_dataclass(base):
                static_fields.update(get_static_fields(base, marker))

        return frozenset(static_fields)

    updates = {}

    def gather(obj):
        if isinstance(obj, StaticShared):
            valid_fields = get_static_fields(obj.__class__, StaticShared)
            for f in fields(obj):
                if f.name not in valid_fields:
                    continue
                val = getattr(obj, f.name)
                default = f.default_factory() if f.default_factory is not MISSING else f.default
                # If there is a difference, cache the change
                if val != default:
                    updates[f.name] = val

        # Traverse child nodes
        if is_dataclass(obj):
            for f in fields(obj):
                gather(getattr(obj, f.name))

    gather(root_node)

    # Sync all StaticShared nodes
    def apply(obj):
        if isinstance(obj, StaticShared):
            valid_fields = get_static_fields(obj.__class__, StaticShared)
            for k, v in updates.items():
                if k in valid_fields and hasattr(obj, k):
                    setattr(obj, k, v)

        if is_dataclass(obj):
            for f in fields(obj):
                apply(getattr(obj, f.name))

    apply(root_node)


class StaticShared:
    """Marker: Fields in subclasses will sync globally across all instances."""
    pass
