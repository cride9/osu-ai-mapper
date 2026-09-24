from __future__ import annotations

import numpy as np


def bezier(points, count=80):
    points = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, count)[:, None]
    current = np.broadcast_to(points, (count, len(points), 2)).copy()
    for size in range(len(points) - 1, 0, -1):
        current = current[:, :size] * (1 - t[:, :, None]) + current[:, 1:size + 1] * t[:, :, None]
    return current[:, 0]


def slider_path(obj):
    points = np.asarray([(obj.x, obj.y)] + obj.points, float)
    if obj.curve == "L":
        path = points
    elif obj.curve == "P" and len(points) == 3:
        a, b, c = points
        matrix = 2 * np.array([b - a, c - a])
        if abs(np.linalg.det(matrix)) < 1e-6:
            path = points
        else:
            center = np.linalg.solve(matrix, np.array([b @ b - a @ a, c @ c - a @ a]))
            angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
            direction = 1 if ((angles[1] - angles[0]) % (2 * np.pi)) < ((angles[2] - angles[0]) % (2 * np.pi)) else -1
            delta = (angles[2] - angles[0]) % (2 * np.pi) if direction > 0 else -((angles[0] - angles[2]) % (2 * np.pi))
            angle = np.linspace(angles[0], angles[0] + delta, 100)
            path = center + np.linalg.norm(a - center) * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    elif obj.curve == "C":
        padded = np.vstack([points[0], points, points[-1]])
        sections = []
        for i in range(len(points) - 1):
            p0, p1, p2, p3 = padded[i:i + 4]
            t = np.linspace(0, 1, 32)[:, None]
            sections.append(0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t*t + (-p0+3*p1-3*p2+p3)*t*t*t))
        path = np.vstack(sections)
    else:
        sections, start = [], 0
        for i in range(1, len(points)):
            if np.array_equal(points[i], points[i - 1]):
                if i > start:
                    sections.append(bezier(points[start:i]))
                start = i
        if start < len(points):
            sections.append(bezier(points[start:]))
        path = np.vstack(sections)
    if len(path) < 2:
        return path
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    if cumulative[-1] > obj.length:
        index = min(len(path) - 1, int(np.searchsorted(cumulative, obj.length)))
        fraction = (obj.length - cumulative[index - 1]) / max(lengths[index - 1], 1e-6)
        path = np.vstack([path[:index], path[index - 1] + fraction * (path[index] - path[index - 1])])
    elif cumulative[-1] < obj.length and lengths[-1] > 0:
        path = np.vstack([path, path[-1] + (obj.length - cumulative[-1]) * (path[-1] - path[-2]) / lengths[-1]])
    return path


def plot_map(bm, at_seconds=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if at_seconds is None:
        at_seconds = bm.objects[0].time / 1000 if bm.objects else 0
    start = at_seconds * 1000
    selected = [o for o in bm.objects if start <= o.time < start + 4000][:48]
    fig, (ax, timeline) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.set(xlim=(0, 512), ylim=(384, 0), aspect="equal", title="4-second pattern preview")
    ax.set_facecolor("#131b2b")
    if selected:
        ax.plot([o.x for o in selected], [o.y for o in selected], color="#64748b", alpha=0.4)
    for i, obj in enumerate(selected):
        color = plt.cm.viridis(i / max(1, len(selected)))
        if obj.kind == "slider":
            curve = slider_path(obj)
            ax.plot(curve[:, 0], curve[:, 1], color=color, linewidth=6, alpha=0.65)
        ax.scatter(obj.x, obj.y, s=140, c=[color], edgecolors="white")
        ax.text(obj.x, obj.y, str(i + 1), fontsize=7, ha="center", va="center", color="white")
    times = [o.time / 1000 for o in bm.objects]
    timeline.hist(times, bins=max(1, min(120, int(max(times, default=10) / 2))), color="#7289da")
    timeline.axvspan(at_seconds, at_seconds + 4, color="#fb923c", alpha=0.4)
    timeline.set(title="Object density through the song", xlabel="Seconds", ylabel="Objects")
    fig.tight_layout()
    plt.close(fig)
    return fig
