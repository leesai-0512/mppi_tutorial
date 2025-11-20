# examples/renderer_cartpole.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path

def render_cartpole_gif(ts, xs, pole_half_length=0.5, save_path="outputs/cartpole.gif"):
    """
    ts: (T,) 시간 벡터 [sec]
    xs: (T,4) 상태 시퀀스 = [x, x_dot, theta, theta_dot]
    pole_half_length: dynamics에서 쓰는 half-length (기본 0.5)
    save_path: 저장 경로 (gif)
    """
    cart_width, cart_height = 0.4, 0.2
    pole_len = 2.0 * pole_half_length
    track_half = 2.4  # 트랙 표시 범위

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-track_half - 0.5, track_half + 0.5)
    ax.set_ylim(-0.6, 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_yticks([])
    ax.plot([-track_half - 1, track_half + 1], [0, 0], lw=1.0, color="k")

    # cart (rectangle) + pole (line)
    cart = plt.Rectangle((-cart_width/2, 0.0), cart_width, cart_height, fill=False, lw=1.5)
    ax.add_patch(cart)
    pole_line, = ax.plot([], [], lw=2.0)
    time_text = ax.text(0.02, 0.9, "", transform=ax.transAxes)

    def init():
        cart.set_xy((-cart_width/2, 0.0))
        pole_line.set_data([], [])
        time_text.set_text("")
        return cart, pole_line, time_text

    def animate(i):
        x_pos  = xs[i, 0]
        theta  = xs[i, 2]

        # cart 위치 업데이트
        cart.set_xy((x_pos - cart_width/2, 0.0))

        # pole 끝점 계산 (base = cart 상단 중앙)
        base_x, base_y = x_pos, cart_height
        tip_x = base_x + pole_len * np.sin(theta)
        tip_y = base_y + pole_len * np.cos(theta)
        pole_line.set_data([base_x, tip_x], [base_y, tip_y])

        time_text.set_text(f"t = {ts[i]:.2f}s")
        return cart, pole_line, time_text

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(ts), interval=1000*(ts[1]-ts[0]), blit=True
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer="pillow", fps=int(1.0/(ts[1]-ts[0])))
    plt.close(fig)
    return save_path
