import logging

from odmantic import ObjectId

from simstack.core.context import context
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)

logger = logging.getLogger("GaussianOptArtifacts")

OPT_CHART_STEPS = 20


def task_parent_id(kwargs: dict):
    task_id = kwargs.get("task_id") if kwargs else None
    if task_id is None and kwargs:
        node_runner = kwargs.get("node_runner")
        task_id = getattr(node_runner, "task_id", None)
    if task_id is None:
        return None
    try:
        return ObjectId(str(task_id))
    except Exception:
        return None


def _get_db():
    try:
        return context.db
    except RuntimeError:
        return None


def opt_line_chart(data, y_key, title, y_label, parent_id, existing=None):
    series = AGLineSeriesConfig(
        type="line",
        xKey="step",
        yKey=y_key,
        title=y_label,
        data=data,
        marker={"enabled": False},
    )
    axes = [
        AGChartAxisConfig(type="number", position="bottom", title="Optimization step"),
        AGChartAxisConfig(type="number", position="left", title=y_label),
    ]
    if existing is not None:
        existing.data = data
        existing.title = AGChartTitleConfig(text=title)
        existing.series = [series]
        existing.axes = axes
        existing.parent_id = parent_id
        return existing
    return ChartArtifactModel(
        parent_id=parent_id,
        data=data,
        title=AGChartTitleConfig(text=title),
        series=[series],
        axes=axes,
    )


async def persist_opt_charts(energy_data, grad_data, kwargs, existing=(None, None)):
    node_runner = None if not kwargs else kwargs.get("node_runner")
    parent_id = task_parent_id(kwargs)
    if parent_id is None:
        if node_runner is not None:
            node_runner.warning("Skipping optimization charts: missing task_id")
        return existing
    db = _get_db()
    if db is None:
        return existing
    energy_chart = opt_line_chart(
        list(energy_data)[-OPT_CHART_STEPS:],
        "energy",
        "Gaussian optimization energy",
        "Energy (Ha)",
        parent_id,
        existing[0],
    )
    grad_chart = opt_line_chart(
        list(grad_data)[-OPT_CHART_STEPS:],
        "grad_norm",
        "Gaussian optimization gradient norm",
        "|g| (Ha/Bohr)",
        parent_id,
        existing[1],
    )
    try:
        await db.save(energy_chart)
        await db.save(grad_chart)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Failed to store optimization charts: {exc}")
        else:
            logger.warning("Failed to store optimization charts: %s", exc)
        return existing
    if node_runner is not None and energy_data:
        node_runner.info(
            f"Saved optimization charts at step {energy_data[-1]['step']} "
            f"(task_id={parent_id})"
        )
    return energy_chart, grad_chart
