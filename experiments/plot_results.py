"""Render the saved mirror-circuit summary as a dependency-free SVG line chart."""

import csv
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"
NOISES = ("ideal", "depolarizing", "amplitude_damping", "readout_flip", "fake_lagos", "fake_nairobi")
LABELS = {
    "ideal": "Ideal",
    "depolarizing": "Depolarizing",
    "amplitude_damping": "Amplitude damping",
    "readout_flip": "Readout flip",
    "fake_lagos": "Fake Lagos",
    "fake_nairobi": "Fake Nairobi",
}


def main():
    with (RESULTS / "mirror_summary.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    values = {(int(row["qubits"]), int(row["layers"]), row["noise"]): float(row["p_all_zero"])
              for row in rows}
    assert len(values) == len(rows) and values

    colors = {
        "ideal": "#dce6f1",
        "depolarizing": "#4ea5d9",
        "amplitude_damping": "#f4a261",
        "readout_flip": "#8ecf72",
        "fake_lagos": "#e76f8a",
        "fake_nairobi": "#b69cff",
    }
    width, height = 1386, 746
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 1040 560">',
             '<rect width="1040" height="560" fill="#101827"/>',
             '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#e5edf7}.muted{fill:#9aabc0;font-size:12px}.title{font-size:21px;font-weight:700}.panel{font-size:15px;font-weight:650}.legend{font-size:12px}</style>',
             '<text x="34" y="38" class="title">All-zero outcomes</text>']
    qubits = (2, 4, 6, 8)
    chart_y, chart_h, chart_w, panel_gap = 96, 300, 270, 55
    for panel, layers in enumerate((2, 4, 8)):
        px = 58 + panel * (chart_w + panel_gap)
        parts.append(f'<text x="{px + chart_w / 2}" y="76" text-anchor="middle" class="panel">{layers} layers</text>')
        left, top = px + 30, chart_y
        plot_w, plot_h = chart_w - 42, chart_h
        for percent in (0, 25, 50, 75, 100):
            y = top + plot_h * (1 - percent / 100)
            parts.append(f'<line x1="{left}" y1="{y}" x2="{left + plot_w}" y2="{y}" stroke="#344257" stroke-width="1"/>')
            if panel == 0:
                parts.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="muted">{percent}%</text>')
        for index, count in enumerate(qubits):
            x = left + plot_w * index / (len(qubits) - 1)
            parts.append(f'<text x="{x}" y="{top + plot_h + 22}" text-anchor="middle" class="muted">{count}</text>')
        for noise in NOISES:
            points = []
            for index, count in enumerate(qubits):
                value = values.get((count, layers, noise))
                if value is not None:
                    x = left + plot_w * index / (len(qubits) - 1)
                    y = top + plot_h * (1 - value)
                    points.append((x, y))
            if points:
                coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[noise]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
                for x, y in points:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[noise]}"/>')
    for index, noise in enumerate(NOISES):
        x = 100 + (index % 3) * 280
        y = 465 + (index // 3) * 34
        parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 24}" y2="{y - 4}" stroke="{colors[noise]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 32}" y="{y}" class="legend">{LABELS[noise]}</text>')
    parts.extend(['<text x="520" y="424" text-anchor="middle" class="muted">Qubits</text>', '</svg>'])
    (RESULTS / "mirror_lines.svg").write_text("\n".join(parts) + "\n")
    print(f"Saved {RESULTS / 'mirror_lines.svg'}")


if __name__ == "__main__":
    main()
