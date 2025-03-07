from collections import deque
from collections.abc import Iterable
from typing import Callable, Any

import matplotlib.pyplot as plt
from tqdm import tqdm


class GUI:
    def __init__(self, horizon: int, title: str):
        self.N = horizon
        self.N_max = horizon - 1

        # --- Figure initialisation ---
        self.figure = plt.figure(title)
        self.subplots, self.lines = dict(), dict()

    def add_subplot(self, row, col, idx):
        self.subplots[idx] = self.figure.add_subplot(row, col, idx)
        ax = self.subplots[idx]

        ax.set_navigate(False)
        ax.set_autoscalex_on(False)
        ax.set_autoscaley_on(False)
        ax.xaxis.set_animated(True)

        return ax

    def add_line(self, idx: int, name: str):
        ln, = self.subplots[idx].plot([], animated=False)
        self.lines[name] = (ln, idx, deque(maxlen=self.N), deque(maxlen=self.N))
        return ln

    def config(self):
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)

    def add_data(self, name: str, x: int, y: float):
        _, _, qx, qy = self.lines[name]
        self.N_max = max(self.N_max, x)
        if len(qx) >= self.N:
            qx.popleft()
            qy.popleft()
        qx.append(x)
        qy.append(y)

    def update(self, custom_updates=None):
        if plt.fignum_exists(self.figure.number):
            self.figure.canvas.restore_region(self.bg)

            for ax in self.subplots.values():
                ax.set_xlim(self.N_max - (self.N - 1), self.N_max + 1)
                self.figure.draw_artist(ax.xaxis)

            for ln, idx, qx, qy in self.lines.values():
                ln.set_xdata(qx)
                ln.set_ydata(qy)
                self.subplots[idx].draw_artist(ln)

            if custom_updates:
                custom_updates()

            self.figure.canvas.blit(self.figure.bbox)
            self.figure.canvas.flush_events()


class ProgressBar(tqdm):
    def __init__(self, iterable: Iterable, epoch: int, postfix: Callable, **kwargs):
        bar_fmt = "{l_bar}{bar:20}|{n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}] -- {postfix}"
        tqdm.__init__(self, iterable, desc=f"epoch {epoch}", unit="it", bar_format=bar_fmt, **kwargs)
        self.get_postfix_str = postfix

    def update(self, n: float | None = ...) -> bool | None:
        self.set_postfix_str(self.get_postfix_str())
        return tqdm.update(self, n)


class Log:
    def __init__(self, fmt: str, unit: str = None):
        self.fmt = fmt
        self.unit = unit
        self.data = []

    @property
    def value(self) -> float | int | None:
        return self.data[-1] if len(self.data) else None

    def add(self, value):
        self.data.append(value)

    def finish(self):
        return np.array(self.data)

    @staticmethod
    def string(**kwargs: "Log"):
        out = ""
        for name, data in kwargs.items():
            if data.value is not None:
                out += f"{name}: {data.value: {data.fmt}}"
                if data.unit is not None:
                    out += data.unit
                out += ", "
        return out[:-2]


if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    import time

    # GUI & Visualization
    visualizer = GUI(horizon=100, title="Title")
    ax1 = visualizer.add_subplot(1, 1, 1)
    ax1.set_title("hello")

    ax1.set_ylim(-1, 1)

    ln1 = visualizer.add_line(1, "line1")

    visualizer.config()

    # --- Iterate sample over the data in a live fashion ---
    for iter in tqdm(range(1000)):
        visualizer.add_data("line1", iter, np.sin(0.2*iter))

        # time.sleep(0.001)

        visualizer.update()
