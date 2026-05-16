from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierPoint:
    label: str
    pass_rate: float
    cost_per_success: float
    p50_latency_s: float


# Hand-written SVG instead of a plotting library: two small panels don't justify a dependency, and
# the colors follow the viewer's light or dark theme through a media query inside the file.
STYLE = """
  .surface { fill: #fcfcfb; } .grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; }
  .dot { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }
  .label { fill: #52514e; } .muted { fill: #898781; } .title { fill: #0b0b0b; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; } .grid { stroke: #2c2c2a; } .axis { stroke: #383835; }
    .dot { fill: #3987e5; stroke: #1a1a19; } .label { fill: #c3c2b7; } .title { fill: #f0efec; }
  }
  text { font-family: system-ui, -apple-system, sans-serif; font-size: 12px; }
"""

WIDTH, HEIGHT = 760, 330
PANEL_W, PANEL_H = 300, 220
TOP, LEFT, GAP = 50, 60, 90
Y_MIN, Y_MAX = 0.2, 0.7


def _panel(
    points: list[tuple[str, float, float]], x0: float, x_max: float, x_title: str, fmt: str
) -> list[str]:
    def sx(v: float) -> float:
        return x0 + v / x_max * PANEL_W

    def sy(v: float) -> float:
        return TOP + (Y_MAX - v) / (Y_MAX - Y_MIN) * PANEL_H

    out = []
    for i in range(6):
        y = Y_MIN + i * 0.1
        out.append(
            f'<line class="grid" x1="{x0}" x2="{x0 + PANEL_W}" y1="{sy(y):.1f}" y2="{sy(y):.1f}"/>'
        )
        out.append(
            f'<text class="muted" x="{x0 - 8}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.1f}</text>'
        )
    for i in range(5):
        x = x_max * i / 4
        out.append(
            f'<text class="muted" x="{sx(x):.1f}" y="{TOP + PANEL_H + 18}" text-anchor="middle">'
            f"{fmt.format(x)}</text>"
        )
    bottom = TOP + PANEL_H
    out.append(f'<line class="axis" x1="{x0}" x2="{x0 + PANEL_W}" y1="{bottom}" y2="{bottom}"/>')
    out.append(
        f'<text class="muted" x="{x0 + PANEL_W / 2}" y="{bottom + 40}" text-anchor="middle">'
        f"{x_title}</text>"
    )
    for label, x, y in points:
        # Labels go right of the dot, or left of it near the panel's right edge.
        right = sx(x) < x0 + PANEL_W * 0.7
        dx, anchor = (9, "start") if right else (-9, "end")
        out.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5"/>')
        out.append(
            f'<text class="label" x="{sx(x) + dx:.1f}" y="{sy(y) + 4:.1f}" text-anchor="{anchor}">'
            f"{label}</text>"
        )
    return out


def frontier_svg(points: list[FrontierPoint]) -> str:
    """Pass rate against cost per successful task and against p50 latency, side by side: two
    panels with one y axis each, never one chart with two x scales."""
    max_cost = max(p.cost_per_success for p in points)
    max_latency = max(p.p50_latency_s for p in points)
    cost_max = 0.1 * (int(max_cost / 0.1) + 1)
    latency_max = 20.0 * (int(max_latency / 20) + 1)
    title = "Pass rate vs cost and latency, 40 core scenarios"
    body = [
        f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="title" x="20" y="24" font-weight="600">{title}</text>',
        f'<text class="muted" x="{LEFT}" y="{TOP - 12}">pass rate</text>',
        *_panel(
            [(p.label, p.cost_per_success, p.pass_rate) for p in points],
            LEFT,
            cost_max,
            "$ per successful task",
            "${:.2f}",
        ),
        *_panel(
            [(p.label, p.p50_latency_s, p.pass_rate) for p in points],
            LEFT + PANEL_W + GAP,
            latency_max,
            "p50 scenario latency (s)",
            "{:.0f}",
        ),
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}"'
        f' width="{WIDTH}" height="{HEIGHT}" role="img"'
        ' aria-label="Pass rate against cost per successful task and p50 latency per config">\n'
        f"<style>{STYLE}</style>\n" + "\n".join(body) + "\n</svg>\n"
    )
