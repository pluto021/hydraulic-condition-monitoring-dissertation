from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ZIP_CANDIDATES = [
    ROOT / "condition+monitoring+of+hydraulic+systems.zip",
    ROOT / "ref" / "condition+monitoring+of+hydraulic+systems.zip",
    ROOT / "参考文献" / "condition+monitoring+of+hydraulic+systems.zip",
]
ZIP_PATH = next((path for path in ZIP_CANDIDATES if path.exists()), ZIP_CANDIDATES[0])
OUTPUT_DIR = ROOT / "outputs" / "hydraulic_data_processing"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = OUTPUT_DIR / "processed"

PROFILE_COLUMNS = [
    "cooler_condition",
    "valve_condition",
    "pump_leakage",
    "accumulator_pressure",
    "stable_flag",
]

SENSOR_META = [
    ("PS1", "Pressure", "bar", 100, 6000, "physical"),
    ("PS2", "Pressure", "bar", 100, 6000, "physical"),
    ("PS3", "Pressure", "bar", 100, 6000, "physical"),
    ("PS4", "Pressure", "bar", 100, 6000, "physical"),
    ("PS5", "Pressure", "bar", 100, 6000, "physical"),
    ("PS6", "Pressure", "bar", 100, 6000, "physical"),
    ("EPS1", "Motor power", "W", 100, 6000, "physical"),
    ("FS1", "Volume flow", "l/min", 10, 600, "physical"),
    ("FS2", "Volume flow", "l/min", 10, 600, "physical"),
    ("TS1", "Temperature", "deg C", 1, 60, "physical"),
    ("TS2", "Temperature", "deg C", 1, 60, "physical"),
    ("TS3", "Temperature", "deg C", 1, 60, "physical"),
    ("TS4", "Temperature", "deg C", 1, 60, "physical"),
    ("VS1", "Vibration", "mm/s", 1, 60, "physical"),
    ("CE", "Cooling efficiency", "%", 1, 60, "virtual"),
    ("CP", "Cooling power", "kW", 1, 60, "virtual"),
    ("SE", "Efficiency factor", "%", 1, 60, "virtual"),
]

PHYSICAL_SENSORS = [row[0] for row in SENSOR_META if row[5] == "physical"]
FEATURE_NAMES = ["mean", "std", "min", "max", "range"]

SELECTED_FEATURES = {
    "valve_condition": ["FS1_range", "PS2_std", "EPS1_mean", "FS1_mean"],
    "pump_leakage": ["FS1_mean", "EPS1_mean", "PS3_mean", "PS2_std"],
    "accumulator_pressure": ["FS1_max", "PS3_max", "EPS1_mean", "PS3_range"],
    "stable_flag": ["FS1_max", "FS1_range", "FS1_mean", "PS2_std"],
}

CHART_BG = "#FFFFFF"
INK = "#1F2933"
MUTED = "#52616B"
GRID = "#D6DEE3"
ACCENT = "#2F6F73"
ACCENT_LIGHT = "#9BC7C6"


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "Calibri Bold.ttf" if bold else "calibri.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, image_font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=image_font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    image_font: ImageFont.ImageFont,
    fill: str = INK,
) -> None:
    width, height = text_size(draw, text, image_font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, fill=fill, font=image_font)


def draw_right(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    image_font: ImageFont.ImageFont,
    fill: str = MUTED,
) -> None:
    width, height = text_size(draw, text, image_font)
    draw.text((xy[0] - width, xy[1] - height / 2), text, fill=fill, font=image_font)


def nice_ticks(max_value: float, count: int = 4) -> list[float]:
    if max_value <= 0:
        return [0]
    rough = max_value / count
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude
    if residual > 5:
        step = 10 * magnitude
    elif residual > 2:
        step = 5 * magnitude
    elif residual > 1:
        step = 2 * magnitude
    else:
        step = magnitude
    top = math.ceil(max_value / step) * step
    return [i * step for i in range(int(top / step) + 1)]


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, PROCESSED_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_matrix(zp: zipfile.ZipFile, name: str) -> np.ndarray:
    with zp.open(f"{name}.txt") as handle:
        return np.loadtxt(handle)


def read_profile(zp: zipfile.ZipFile) -> pd.DataFrame:
    with zp.open("profile.txt") as handle:
        profile = pd.read_csv(handle, sep="\t", header=None, names=PROFILE_COLUMNS)
    return profile.astype(int)


def count_rows_cols(zp: zipfile.ZipFile, filename: str) -> tuple[int, int]:
    rows = 0
    cols = 0
    with zp.open(filename) as handle:
        for line in handle:
            stripped = line.decode("utf-8").strip()
            if rows == 0:
                cols = len(stripped.split("\t")) if stripped else 0
            rows += 1
    return rows, cols


def verify_dataset(zp: zipfile.ZipFile) -> pd.DataFrame:
    records = []
    profile_rows, profile_cols = count_rows_cols(zp, "profile.txt")
    records.append(
        {
            "file": "profile.txt",
            "sensor": "profile",
            "physical_quantity": "condition labels",
            "unit": "mixed",
            "sampling_rate_hz": "",
            "expected_columns": 5,
            "observed_rows": profile_rows,
            "observed_columns": profile_cols,
            "row_check": profile_rows == 2205,
            "column_check": profile_cols == 5,
            "sensor_type": "target labels",
        }
    )
    for sensor, quantity, unit, rate, expected_cols, sensor_type in SENSOR_META:
        rows, cols = count_rows_cols(zp, f"{sensor}.txt")
        records.append(
            {
                "file": f"{sensor}.txt",
                "sensor": sensor,
                "physical_quantity": quantity,
                "unit": unit,
                "sampling_rate_hz": rate,
                "expected_columns": expected_cols,
                "observed_rows": rows,
                "observed_columns": cols,
                "row_check": rows == 2205,
                "column_check": cols == expected_cols,
                "sensor_type": sensor_type,
            }
        )
    return pd.DataFrame(records)


def feature_frame_for_sensor(values: np.ndarray, sensor: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"{sensor}_mean": values.mean(axis=1),
            f"{sensor}_std": values.std(axis=1, ddof=0),
            f"{sensor}_min": values.min(axis=1),
            f"{sensor}_max": values.max(axis=1),
            f"{sensor}_range": values.max(axis=1) - values.min(axis=1),
        }
    )


def build_cycle_features(zp: zipfile.ZipFile, profile: pd.DataFrame) -> pd.DataFrame:
    features = [profile.reset_index(drop=True)]
    for sensor in PHYSICAL_SENSORS:
        values = read_matrix(zp, sensor)
        features.append(feature_frame_for_sensor(values, sensor))
    output = pd.concat(features, axis=1)
    output.insert(0, "cycle_id", np.arange(1, len(output) + 1))
    return output


def label_distribution(profile: pd.DataFrame) -> pd.DataFrame:
    records = []
    for label in PROFILE_COLUMNS:
        counts = profile[label].value_counts().sort_index()
        for state, count in counts.items():
            records.append(
                {
                    "target": label,
                    "state": int(state),
                    "count": int(count),
                    "percentage": round(100 * count / len(profile), 2),
                }
            )
    return pd.DataFrame(records)


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    x = values.to_numpy(dtype=float)
    g = groups.to_numpy()
    grand_mean = x.mean()
    ss_total = float(((x - grand_mean) ** 2).sum())
    if math.isclose(ss_total, 0.0):
        return 0.0
    ss_between = 0.0
    for group in np.unique(g):
        mask = g == group
        ss_between += mask.sum() * float((x[mask].mean() - grand_mean) ** 2)
    return ss_between / ss_total


def eta_screening(cycle_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for col in cycle_features.columns
        if any(col.endswith(f"_{feature}") for feature in FEATURE_NAMES)
    ]
    records = []
    for target in ["valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"]:
        for feature in feature_cols:
            records.append(
                {
                    "target": target,
                    "feature": feature,
                    "eta_squared": eta_squared(cycle_features[feature], cycle_features[target]),
                }
            )
    result = pd.DataFrame(records)
    result["eta_squared"] = result["eta_squared"].round(6)
    result["rank_within_target"] = result.groupby("target")["eta_squared"].rank(
        method="first", ascending=False
    ).astype(int)
    return result.sort_values(["target", "rank_within_target"])


def feature_summary(cycle_features: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    records = []
    grouped = cycle_features.groupby(target, sort=True)
    for state, group in grouped:
        for feature in features:
            records.append(
                {
                    "target": target,
                    "state": int(state),
                    "feature": feature,
                    "n": int(len(group)),
                    "mean": round(float(group[feature].mean()), 4),
                    "std": round(float(group[feature].std(ddof=1)), 4),
                    "min": round(float(group[feature].min()), 4),
                    "max": round(float(group[feature].max()), 4),
                }
            )
    return pd.DataFrame(records)


def cooccurrence_tables(profile: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pairs = [
        ("valve_condition", "pump_leakage"),
        ("valve_condition", "accumulator_pressure"),
        ("pump_leakage", "accumulator_pressure"),
        ("stable_flag", "valve_condition"),
        ("stable_flag", "pump_leakage"),
        ("stable_flag", "accumulator_pressure"),
    ]
    tables = {}
    for left, right in pairs:
        tables[f"{left}_x_{right}"] = pd.crosstab(profile[left], profile[right])
    return tables


def save_label_distribution_plot(distribution: pd.DataFrame) -> None:
    width, height = 1800, 1500
    image = Image.new("RGB", (width, height), CHART_BG)
    draw = ImageDraw.Draw(image)
    title_font = font(36, bold=True)
    panel_title_font = font(24, bold=True)
    label_font = font(18)
    small_font = font(15)

    draw_centered(
        draw,
        (width / 2, 48),
        "UCI Hydraulic Dataset Target Label Distributions",
        title_font,
    )

    panel_w, panel_h = 800, 400
    x_starts = [90, 910]
    y_starts = [110, 560, 1010]
    targets = PROFILE_COLUMNS
    for idx, target in enumerate(targets):
        px = x_starts[idx % 2]
        py = y_starts[idx // 2]
        sub = distribution[distribution["target"] == target].copy()
        states = [str(value) for value in sub["state"].tolist()]
        counts = sub["count"].astype(float).tolist()
        ticks = nice_ticks(max(counts), count=4)
        y_max = max(ticks) if ticks else max(counts)

        plot_left = px + 85
        plot_top = py + 60
        plot_right = px + panel_w - 30
        plot_bottom = py + panel_h - 70
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        draw_centered(draw, (px + panel_w / 2, py + 26), target.replace("_", " ").title(), panel_title_font)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)

        for tick in ticks:
            y = plot_bottom - (tick / y_max) * plot_h if y_max else plot_bottom
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw_right(draw, (plot_left - 10, y), str(int(tick)), small_font)

        slot = plot_w / max(len(states), 1)
        bar_w = min(95, slot * 0.58)
        for i, (state, count) in enumerate(zip(states, counts)):
            cx = plot_left + slot * (i + 0.5)
            bar_h = (count / y_max) * plot_h if y_max else 0
            draw.rectangle(
                (cx - bar_w / 2, plot_bottom - bar_h, cx + bar_w / 2, plot_bottom),
                fill=ACCENT,
            )
            draw_centered(draw, (cx, plot_bottom + 24), state, label_font, fill=MUTED)
            draw_centered(draw, (cx, plot_bottom - bar_h - 18), str(int(count)), small_font, fill=INK)

        draw.text((plot_left, plot_top - 26), "Cycle count", fill=MUTED, font=small_font)
        draw_centered(draw, (px + panel_w / 2, py + panel_h - 22), "State", label_font, fill=MUTED)

    note = "Counts are experimental benchmark cycle labels, not naturally occurring aircraft failure counts."
    draw_centered(draw, (width / 2, height - 38), note, label_font, fill=MUTED)
    image.save(FIGURE_DIR / "figure_3_label_distribution_bar_chart.png")


def save_boxplot(cycle_features: pd.DataFrame, target: str, features: list[str]) -> None:
    width, height = 1800, 1250
    image = Image.new("RGB", (width, height), CHART_BG)
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    panel_title_font = font(23, bold=True)
    label_font = font(17)
    small_font = font(14)

    target_label = target.replace("_", " ").title()
    draw_centered(draw, (width / 2, 45), f"Selected Features by {target_label}", title_font)

    panel_w, panel_h = 800, 500
    x_starts = [90, 910]
    y_starts = [95, 620]
    states = sorted(cycle_features[target].unique())

    for idx, feature in enumerate(features):
        px = x_starts[idx % 2]
        py = y_starts[idx // 2]
        grouped = [
            cycle_features.loc[cycle_features[target] == state, feature].to_numpy(dtype=float)
            for state in states
        ]
        stats = []
        for values in grouped:
            stats.append(
                {
                    "p05": float(np.percentile(values, 5)),
                    "q1": float(np.percentile(values, 25)),
                    "median": float(np.percentile(values, 50)),
                    "q3": float(np.percentile(values, 75)),
                    "p95": float(np.percentile(values, 95)),
                }
            )
        y_min = min(item["p05"] for item in stats)
        y_max = max(item["p95"] for item in stats)
        padding = (y_max - y_min) * 0.12 if y_max > y_min else 1
        y_min -= padding
        y_max += padding

        plot_left = px + 95
        plot_top = py + 65
        plot_right = px + panel_w - 35
        plot_bottom = py + panel_h - 75
        plot_w = plot_right - plot_left
        plot_h = plot_bottom - plot_top

        def y_pos(value: float) -> float:
            return plot_bottom - ((value - y_min) / (y_max - y_min)) * plot_h

        draw_centered(draw, (px + panel_w / 2, py + 28), feature, panel_title_font)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)

        for frac in [0, 0.25, 0.5, 0.75, 1]:
            value = y_min + frac * (y_max - y_min)
            y = y_pos(value)
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw_right(draw, (plot_left - 10, y), f"{value:.2f}", small_font)

        slot = plot_w / max(len(states), 1)
        box_w = min(85, slot * 0.48)
        for i, (state, stat) in enumerate(zip(states, stats)):
            cx = plot_left + slot * (i + 0.5)
            y_p05 = y_pos(stat["p05"])
            y_q1 = y_pos(stat["q1"])
            y_med = y_pos(stat["median"])
            y_q3 = y_pos(stat["q3"])
            y_p95 = y_pos(stat["p95"])
            draw.line((cx, y_p95, cx, y_p05), fill=INK, width=2)
            draw.line((cx - box_w / 3, y_p95, cx + box_w / 3, y_p95), fill=INK, width=2)
            draw.line((cx - box_w / 3, y_p05, cx + box_w / 3, y_p05), fill=INK, width=2)
            draw.rectangle(
                (cx - box_w / 2, y_q3, cx + box_w / 2, y_q1),
                fill=ACCENT_LIGHT,
                outline=ACCENT,
                width=2,
            )
            draw.line((cx - box_w / 2, y_med, cx + box_w / 2, y_med), fill=INK, width=3)
            draw_centered(draw, (cx, plot_bottom + 24), str(state), label_font, fill=MUTED)

        draw.text((plot_left, plot_top - 26), "Feature value", fill=MUTED, font=small_font)
        draw_centered(
            draw,
            (px + panel_w / 2, py + panel_h - 24),
            target.replace("_", " "),
            label_font,
            fill=MUTED,
        )

    note = "Boxes show median and interquartile range; whiskers show 5th to 95th percentiles."
    draw_centered(draw, (width / 2, height - 35), note, label_font, fill=MUTED)
    image.save(FIGURE_DIR / f"figure_4_boxplots_{target}.png")


def save_markdown_report(
    verification: pd.DataFrame,
    distribution: pd.DataFrame,
    eta: pd.DataFrame,
) -> None:
    def as_markdown_table(frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        rows = [[str(value) for value in row] for row in frame.to_numpy()]
        widths = [
            max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
            for col_idx in range(len(headers))
        ]
        header_line = "| " + " | ".join(
            headers[col_idx].ljust(widths[col_idx]) for col_idx in range(len(headers))
        ) + " |"
        rule_line = "| " + " | ".join("-" * widths[col_idx] for col_idx in range(len(headers))) + " |"
        body_lines = [
            "| " + " | ".join(row[col_idx].ljust(widths[col_idx]) for col_idx in range(len(headers))) + " |"
            for row in rows
        ]
        return "\n".join([header_line, rule_line, *body_lines])

    top_eta = (
        eta[eta["rank_within_target"] <= 5]
        .sort_values(["target", "rank_within_target"])
        .loc[:, ["target", "rank_within_target", "feature", "eta_squared"]]
    )
    lines = [
        "# Hydraulic Dataset Processing Summary",
        "",
        "This file records the reproducible outputs generated from the uploaded UCI hydraulic condition-monitoring dataset.",
        "",
        "## Dataset Verification",
        "",
        f"- Files checked: {len(verification)}",
        f"- All row checks passed: {bool(verification['row_check'].all())}",
        f"- All column checks passed: {bool(verification['column_check'].all())}",
        "",
        "## Label Distribution",
        "",
        as_markdown_table(distribution),
        "",
        "## Top Exploratory Eta-Squared Features",
        "",
        as_markdown_table(top_eta),
        "",
        "## Dissertation Use",
        "",
        "Use these outputs as benchmark-level hydraulic evidence. Do not interpret them as aircraft-certified maintenance thresholds.",
    ]
    (OUTPUT_DIR / "processing_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset zip not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as zp:
        verification = verify_dataset(zp)
        profile = read_profile(zp)
        cycle_features = build_cycle_features(zp, profile)

    sensor_structure = pd.DataFrame(
        [
            {
                "sensor": sensor,
                "physical_quantity": quantity,
                "unit": unit,
                "sampling_rate_hz": rate,
                "points_per_60s_cycle": expected_cols,
                "sensor_type": sensor_type,
            }
            for sensor, quantity, unit, rate, expected_cols, sensor_type in SENSOR_META
        ]
    )
    distribution = label_distribution(profile)
    eta = eta_screening(cycle_features)

    verification.to_csv(TABLE_DIR / "table_dataset_verification.csv", index=False)
    sensor_structure.to_csv(TABLE_DIR / "table_sensor_structure.csv", index=False)
    distribution.to_csv(TABLE_DIR / "table_target_label_distributions.csv", index=False)
    eta.to_csv(TABLE_DIR / "table_eta_squared_screening.csv", index=False)
    eta[eta["rank_within_target"] <= 10].to_csv(
        TABLE_DIR / "table_eta_squared_top10_by_target.csv", index=False
    )

    for target, features in SELECTED_FEATURES.items():
        summary = feature_summary(cycle_features, target, features)
        summary.to_csv(TABLE_DIR / f"table_feature_summary_by_{target}.csv", index=False)
        save_boxplot(cycle_features, target, features)

    for name, table in cooccurrence_tables(profile).items():
        table.to_csv(TABLE_DIR / f"table_cooccurrence_{name}.csv")

    cycle_features.to_csv(PROCESSED_DIR / "cycle_level_features_physical_sensors.csv", index=False)
    profile.to_csv(PROCESSED_DIR / "profile_renamed_labels.csv", index=False)

    save_label_distribution_plot(distribution)
    save_markdown_report(verification, distribution, eta)

    manifest = {
        "source_zip": str(ZIP_PATH),
        "output_dir": str(OUTPUT_DIR),
        "cycle_count": int(len(profile)),
        "physical_sensor_feature_count": int(cycle_features.shape[1] - 1 - len(PROFILE_COLUMNS)),
        "tables": sorted(path.name for path in TABLE_DIR.glob("*.csv")),
        "figures": sorted(path.name for path in FIGURE_DIR.glob("*.png")),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
