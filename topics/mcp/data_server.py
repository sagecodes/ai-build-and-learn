"""
Specialized MCP server for data analysis.
Demonstrates stateful tools, filtering, aggregation, and chart generation.
"""

from fastmcp import FastMCP
from typing import Optional
import json
import csv
import io
import base64

mcp = FastMCP("Data Analysis Server")

# In-memory dataset store
datasets: dict[str, list[dict]] = {}

# --- Sample data for demo ---

SAMPLE_DATA = [
    {"city": "Seattle", "state": "WA", "population": 749256, "avg_temp_f": 52, "tech_companies": 412},
    {"city": "San Francisco", "state": "CA", "population": 873965, "avg_temp_f": 60, "tech_companies": 1253},
    {"city": "Austin", "state": "TX", "population": 979882, "avg_temp_f": 68, "tech_companies": 387},
    {"city": "New York", "state": "NY", "population": 8336817, "avg_temp_f": 55, "tech_companies": 2108},
    {"city": "Denver", "state": "CO", "population": 715522, "avg_temp_f": 50, "tech_companies": 298},
    {"city": "Portland", "state": "OR", "population": 652503, "avg_temp_f": 53, "tech_companies": 215},
    {"city": "Miami", "state": "FL", "population": 442241, "avg_temp_f": 77, "tech_companies": 189},
    {"city": "Chicago", "state": "IL", "population": 2693976, "avg_temp_f": 50, "tech_companies": 891},
    {"city": "Boston", "state": "MA", "population": 675647, "avg_temp_f": 51, "tech_companies": 623},
    {"city": "Los Angeles", "state": "CA", "population": 3898747, "avg_temp_f": 66, "tech_companies": 1847},
]


@mcp.tool
def load_sample_data(name: str = "cities") -> str:
    """Load the built-in sample dataset into memory.

    Args:
        name: Name to store the dataset under (default: "cities").
    """
    datasets[name] = [row.copy() for row in SAMPLE_DATA]
    return f"Loaded {len(datasets[name])} rows into '{name}'. Columns: {list(datasets[name][0].keys())}"


@mcp.tool
def load_csv(name: str, csv_text: str) -> str:
    """Load a CSV string into memory as a named dataset.

    Args:
        name: Name to store the dataset under.
        csv_text: Raw CSV content as a string.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = []
    for row in reader:
        # Try to convert numeric values
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = int(v)
            except ValueError:
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
        rows.append(parsed)
    datasets[name] = rows
    return f"Loaded {len(rows)} rows into '{name}'. Columns: {list(rows[0].keys()) if rows else []}"


@mcp.tool
def list_datasets() -> dict[str, int]:
    """List all loaded datasets and their row counts."""
    return {name: len(rows) for name, rows in datasets.items()}


@mcp.tool
def describe(name: str) -> dict:
    """Get summary statistics for numeric columns in a dataset.

    Args:
        name: Name of the dataset to describe.
    """
    if name not in datasets:
        return {"error": f"Dataset '{name}' not found. Load it first."}

    rows = datasets[name]
    if not rows:
        return {"error": "Dataset is empty."}

    stats = {}
    for key in rows[0]:
        values = [r[key] for r in rows if isinstance(r[key], (int, float))]
        if values:
            stats[key] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": round(sum(values) / len(values), 2),
                "sum": round(sum(values), 2),
            }
    return stats


@mcp.tool
def filter_rows(
    name: str,
    column: str,
    operator: str,
    value: str,
    save_as: Optional[str] = None,
) -> str:
    """Filter rows in a dataset by a condition.

    Args:
        name: Dataset to filter.
        column: Column name to filter on.
        operator: One of "==", "!=", ">", "<", ">=", "<=", "contains".
        value: Value to compare against (will be auto-typed).
        save_as: Optional name to save the filtered result as a new dataset.
    """
    if name not in datasets:
        return f"Dataset '{name}' not found."

    rows = datasets[name]

    # Auto-type the value
    try:
        typed_value = int(value)
    except ValueError:
        try:
            typed_value = float(value)
        except ValueError:
            typed_value = value

    ops = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "contains": lambda a, b: str(b).lower() in str(a).lower(),
    }

    if operator not in ops:
        return f"Unknown operator '{operator}'. Use one of: {list(ops.keys())}"

    filtered = [r for r in rows if column in r and ops[operator](r[column], typed_value)]

    result_name = save_as or name
    datasets[result_name] = filtered
    return f"Filtered to {len(filtered)} rows (saved as '{result_name}'). Preview:\n{json.dumps(filtered[:5], indent=2)}"


@mcp.tool
def aggregate(name: str, column: str, operation: str) -> dict:
    """Aggregate a numeric column.

    Args:
        name: Dataset name.
        column: Numeric column to aggregate.
        operation: One of "sum", "avg", "min", "max", "count".
    """
    if name not in datasets:
        return {"error": f"Dataset '{name}' not found."}

    values = [r[column] for r in datasets[name] if isinstance(r.get(column), (int, float))]

    if not values:
        return {"error": f"No numeric values in column '{column}'."}

    result = {
        "sum": sum(values),
        "avg": round(sum(values) / len(values), 2),
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }

    if operation in result:
        return {"column": column, "operation": operation, "result": result[operation]}
    return {"error": f"Unknown operation '{operation}'. Use: sum, avg, min, max, count"}


@mcp.tool
def top_n(name: str, column: str, n: int = 5, ascending: bool = False) -> list[dict]:
    """Get the top N rows sorted by a column.

    Args:
        name: Dataset name.
        column: Column to sort by.
        n: Number of rows to return (default: 5).
        ascending: Sort ascending instead of descending (default: false).
    """
    if name not in datasets:
        return [{"error": f"Dataset '{name}' not found."}]

    rows = datasets[name]
    sorted_rows = sorted(rows, key=lambda r: r.get(column, 0), reverse=not ascending)
    return sorted_rows[:n]


@mcp.tool
def create_chart(
    name: str,
    x_column: str,
    y_column: str,
    chart_type: str = "bar",
    title: Optional[str] = None,
) -> str:
    """Create a chart from a dataset and return it as a base64-encoded PNG.

    Args:
        name: Dataset name.
        x_column: Column for the x-axis.
        y_column: Column for the y-axis.
        chart_type: Type of chart - "bar", "line", "scatter", or "pie" (default: "bar").
        title: Optional chart title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if name not in datasets:
        return f"Dataset '{name}' not found."

    rows = datasets[name]
    x_values = [r[x_column] for r in rows]
    y_values = [r[y_column] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    if chart_type == "bar":
        ax.bar(x_values, y_values)
        ax.set_xticklabels(x_values, rotation=45, ha="right")
    elif chart_type == "line":
        ax.plot(x_values, y_values, marker="o")
        ax.set_xticklabels(x_values, rotation=45, ha="right")
    elif chart_type == "scatter":
        ax.scatter(x_values, y_values)
    elif chart_type == "pie":
        ax.pie(y_values, labels=x_values, autopct="%1.1f%%")
    else:
        return f"Unknown chart type '{chart_type}'. Use: bar, line, scatter, pie"

    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(title or f"{y_column} by {x_column}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


if __name__ == "__main__":
    mcp.run(transport="sse", host="localhost", port=8001)
