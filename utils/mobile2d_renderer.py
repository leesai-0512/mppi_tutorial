# examples/renderer_mobile2d.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from pathlib import Path


def render_mobile2d_gif(
    ts, xs, Xss=None,
    save_path="outputs/mobile2d.gif",
    body_radius=0.15,      # 원형 로봇 반경
    world=None,            # (xmin, xmax, ymin, ymax) 없으면 궤적으로 자동 결정
    goal=None,             # (gx, gy) 선택
    trail_stride=2,        # 궤적 라인 샘플링 간격(메인 궤적)
    show_heading=True,     # 진행방향(heading) 짧은 선 표시
    heading_scale=1.0,     # heading 선 길이 배율 (radius * scale)
):
    """
    ts   : (T,)   시간 [s]
    xs   : (T,3)  실행 궤적 상태: [x, y, theta]
    Xss  : (T, B, H, 2) 또는 (T, B, H+1, 2)
           각 실행 스텝 t에서의 샘플 롤아웃 배치(계획 시간축). (옵션)
    """
    T = len(ts)
    assert xs.shape == (T, 3), "xs should be (T,3)=[x,y,theta]"

    # 월드 범위 자동 결정 (샘플이 있으면 포함)
    if world is None:
        pad = max(0.8, body_radius * 4.0)
        xmin = float(xs[:,0].min())
        xmax = float(xs[:,0].max())
        ymin = float(xs[:,1].min())
        ymax = float(xs[:,1].max())
        if Xss is not None:
            Xss = np.asarray(Xss)
            assert Xss.shape[0] == T, "Xss first dim must match len(ts)"
            xmin = min(xmin, float(Xss[...,0].min()))
            xmax = max(xmax, float(Xss[...,0].max()))
            ymin = min(ymin, float(Xss[...,1].min()))
            ymax = max(ymax, float(Xss[...,1].max()))
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
    else:
        xmin, xmax, ymin, ymax = world

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, ls='--', alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Mobile2D (unicycle) trajectory")

    # 목표점
    if goal is not None:
        gx, gy = goal
        ax.plot([gx], [gy], 'rx', ms=10, label='goal')

    # 메인 궤적(실행)
    trail_line, = ax.plot([], [], '-', lw=2, alpha=0.8, label='trajectory')

    # 로봇 바디(원)
    cx0, cy0, th0 = xs[0]
    body = Circle((cx0, cy0), radius=body_radius, fill=False, lw=2)
    ax.add_patch(body)

    # heading 표시(원 중심에서 짧은 선)
    if show_heading:
        hd_x = [cx0, cx0 + body_radius * heading_scale * np.cos(th0)]
        hd_y = [cy0, cy0 + body_radius * heading_scale * np.sin(th0)]
        heading_line, = ax.plot(hd_x, hd_y, lw=2)
    else:
        heading_line = None

    # 샘플 롤아웃 라인 컬렉션 (프레임마다 교체)
    lc = None
    if Xss is not None:
        lc = LineCollection([], linewidths=0.8, colors='gray', alpha=0.18)
        ax.add_collection(lc)

    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes)
    ax.legend(loc='lower right')

    def init():
        trail_line.set_data([], [])
        body.center = (xs[0,0], xs[0,1])
        if heading_line is not None:
            x, y, th = xs[0]
            hd_x = [x, x + body_radius * heading_scale * np.cos(th)]
            hd_y = [y, y + body_radius * heading_scale * np.sin(th)]
            heading_line.set_data(hd_x, hd_y)
        if lc is not None:
            lc.set_segments([])
        time_text.set_text("")
        return (trail_line, body, time_text) if heading_line is None and lc is None else \
               (trail_line, body, heading_line, time_text) if lc is None else \
               (trail_line, body, time_text, lc) if heading_line is None else \
               (trail_line, body, heading_line, time_text, lc)

    def _segments_from_batch(batch_xy):
        """
        batch_xy: (B, H, 2) 또는 (B, H+1, 2)
        → LineCollection용 segments: (n_segments, 2, 2)
        """
        segs = []
        for P in batch_xy:
            # H 또는 H+1 길이 모두 지원
            if P.shape[0] >= 2:
                segs.append(np.stack([P[:-1], P[1:]], axis=1))  # (H-1, 2, 2)
        if segs:
            return np.concatenate(segs, axis=0)
        return np.empty((0, 2, 2), dtype=float)

    def animate(i):
        # 메인 궤적 업데이트
        trail_line.set_data(xs[:i+1:trail_stride, 0], xs[:i+1:trail_stride, 1])

        # 바디 위치
        x, y, th = xs[i]
        body.center = (x, y)

        # heading 업데이트
        if heading_line is not None:
            hd_x = [x, x + body_radius * heading_scale * np.cos(th)]
            hd_y = [y, y + body_radius * heading_scale * np.sin(th)]
            heading_line.set_data(hd_x, hd_y)

        # 샘플 롤아웃(이 프레임에서의 배치 전체)
        if lc is not None:
            batch_xy = Xss[i]  # (B, H or H+1, 2)
            segs = _segments_from_batch(batch_xy)
            lc.set_segments(segs)

        time_text.set_text(f"t = {ts[i]:.2f}s")
        return (trail_line, body, time_text) if heading_line is None and lc is None else \
               (trail_line, body, heading_line, time_text) if lc is None else \
               (trail_line, body, time_text, lc) if heading_line is None else \
               (trail_line, body, heading_line, time_text, lc)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=T, interval=1000*(ts[1]-ts[0]), blit=True
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=int(1.0/(ts[1]-ts[0])))
    plt.close(fig)
    return save_path