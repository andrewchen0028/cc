"""Data types and utilities."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class ID(Generic[T]):
    """Phantom-typed identifier that prevents cross-type comparison at runtime.

    ``ID`` uses ``__class_getitem__`` to dynamically generate and cache a distinct
    subclass for each type parameter (e.g. ``ID[User]`` and ``ID[Order]`` are
    different classes). This makes ``type(ID[User](1)) is not type(ID[Order](1))``,
    which ``__eq__`` exploits to raise ``TypeError`` on cross-type comparisons
    rather than silently returning ``False``.

    ``Generic[T]`` is included solely for static analysis: mypy/pyright treat
    ``ID[User]`` and ``ID[Order]`` as distinct types, but ``Generic``'s
    ``__class_getitem__`` is fully shadowed at runtime by the custom override.
    Runtime generic introspection (``typing.get_origin``, ``typing.get_args``)
    is therefore non-functional.

    Attributes:
        value: The underlying identifier value (int or str).

    Example:
        ```python
        class User: ...
        class Order: ...

        uid = ID[User](42)
        oid = ID[Order](42)

        uid == ID[User](42)   # True
        uid == ID[User](99)   # False
        uid == oid             # TypeError: cannot compare ID[User] with ID[Order]

        # Safe to use in sets and dicts — hash is keyed on (type, value):
        {uid, oid}             # two distinct elements despite equal .value
        ```
    """

    _registry: dict[str, type[ID]] = {}

    def __class_getitem__(cls, key: type) -> type[ID]:
        name = key.__name__
        if name not in cls._registry:
            cls._registry[name] = type(f"ID[{name}]", (cls,), {"__slots__": ()})
        return cls._registry[name]

    __slots__ = ("value",)

    def __init__(self, value: int | str) -> None:
        self.value = value

    def _check_type(self, other: object) -> None:
        if isinstance(other, ID) and type(self) is not type(other):
            raise TypeError(
                f"cannot compare {type(self).__name__} with {type(other).__name__}"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ID):
            return NotImplemented
        self._check_type(other)
        return self.value == other.value

    def __hash__(self) -> int:
        return hash((type(self), self.value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"
