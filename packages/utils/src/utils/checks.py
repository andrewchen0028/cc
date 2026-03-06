import ibis
from ibis import Table
from ibis.expr.schema import IntoSchema


def check_schema(table: Table, schema: IntoSchema, *, strict: bool = True) -> Table:
    expected = ibis.schema(schema)
    actual = table.schema()

    errors = []
    for col, dtype in expected.items():
        if col not in actual:
            errors.append(f"  missing column: {col!r}")
        elif actual[col] != dtype:
            errors.append(f"  column {col!r}: expected {dtype}, got {actual[col]}")

    if strict and (extra := set(actual) - set(expected)):
        errors.append(f"  unexpected columns: {sorted(extra)}")

    if errors:
        raise TypeError(
            "\n" + "\n".join(errors) + f"\nexpected: {expected}\nactual:   {actual}"
        )

    return table
