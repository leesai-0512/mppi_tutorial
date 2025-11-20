# utils/quadrotor3d_renderer.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def render_quad3d_gif(
    ts, xs, Xss=None,
    save_path="outputs/quad3d.gif",
    world=None,          # (xmin, xmax, ymin, ymax, zmin, zmax)
    goal=None,           # (gx, gy, gz)
    trail_stride=2,      # 메인 궤적 샘플링 stride
):
    """
    ts   : (T,)      시간 [s]
    xs   : (T,3)     실행 궤적: [px, py, pz]
    Xss  : (T, B, H, 3) 또는 (T, B, H+1, 3), optional
           각 실행 스텝에서의 샘플 롤아웃 배치 (계획 horizon).
    """
    ts = np.asarray(ts)
    xs = np.asarray(xs)
    T = len(ts)
    assert xs.shape == (T, 3), "xs must be (T,3) = [px,py,pz]"

    # ---- 월드 범위 결정 ----
    if world is None:
        pad = 0.5
        xmin = float(xs[:, 0].min())
        xmax = float(xs[:, 0].max())
        ymin = float(xs[:, 1].min())
        ymax = float(xs[:, 1].max())
        zmin = float(xs[:, 2].min())
        zmax = float(xs[:, 2].max())
        if Xss is not None:
            Xss = np.asarray(Xss)
            assert Xss.shape[0] == T, "Xss first dim must match len(ts)"
            xmin = min(xmin, float(Xss[..., 0].min()))
            xmax = max(xmax, float(Xss[..., 0].max()))
            ymin = min(ymin, float(Xss[..., 1].min()))
            ymax = max(ymax, float(Xss[..., 1].max()))
            zmin = min(zmin, float(Xss[..., 2].min()))
            zmax = max(zmax, float(Xss[..., 2].max()))
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
        zmin, zmax = zmin - pad, zmax + pad
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = world

    # ---- Figure / Axes ----
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Quadrotor 3D trajectory")

    # 목표점 표시
    if goal is not None:
        gx, gy, gz = goal
        ax.scatter([gx], [gy], [gz], c='r', marker='x', s=60, label='goal')

    # 메인 궤적 라인 + 현재 위치 점
    trail_line, = ax.plot([], [], [], '-', lw=2, alpha=0.9, label='trajectory')
    point, = ax.plot([], [], [], 'bo', ms=6)

    # 샘플 롤아웃 라인 컬렉션
    lc = None
    if Xss is not None:
        # ★ 여기서 완전 빈 상태로 만들면 add_collection3d에서 에러 나므로,
        #   더미 segment 하나 넣어둔 뒤 init에서 비워준다.
        dummy = np.zeros((1, 2, 3), dtype=float)   # 하나의 3D 선분: [[p0],[p1]]
        lc = Line3DCollection(dummy, linewidths=0.6, colors='gray', alpha=0.18)
        ax.add_collection3d(lc)

    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc='lower right')

    # ---- segments 헬퍼 ----
    def _segments_from_batch(batch_xyz):
        """
        batch_xyz: (B,H,3) or (B,H+1,3)
        → Line3DCollection용 segments: (Nseg, 2, 3)
        """
        segs = []
        for P in batch_xyz:
            if P.shape[0] >= 2:
                segs.append(np.stack([P[:-1], P[1:]], axis=1))  # (H-1, 2, 3)
        if segs:
            return np.concatenate(segs, axis=0)
        return np.empty((0, 2, 3), dtype=float)

    # ---- init / animate ----
    def init():
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        if lc is not None:
            lc.set_segments([])   # 더미는 날려버리기
        time_text.set_text("")
        if lc is None:
            return trail_line, point, time_text
        else:
            return trail_line, point, time_text, lc

    def animate(i):
        # 메인 궤적
        trail_line.set_data(xs[:i+1:trail_stride, 0], xs[:i+1:trail_stride, 1])
        trail_line.set_3d_properties(xs[:i+1:trail_stride, 2])

        # 현재 위치
        px, py, pz = xs[i]
        point.set_data([px], [py])
        point.set_3d_properties([pz])

        # 샘플 롤아웃 (이 프레임의 배치 전체)
        if lc is not None:
            batch_xyz = Xss[i]  # (B, H or H+1, 3)
            segs = _segments_from_batch(batch_xyz)
            lc.set_segments(segs)

        time_text.set_text(f"t = {ts[i]:.2f}s")
        if lc is None:
            return trail_line, point, time_text
        else:
            return trail_line, point, time_text, lc

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=T, interval=1000*(ts[1]-ts[0]), blit=True
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=int(1.0/(ts[1]-ts[0])))
    plt.close(fig)
    return save_path
