import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, zoom
from matplotlib.image import BboxImage, AxesImage
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Slider, Button
import math

GRASS = np.tile(plt.imread('img/grass.png'), (3, 3, 1))

CAR = {
    color: zoom(plt.imread('img/car-{}.png'.format(color)), [0.3, 0.3, 1.])
    for color in ['gray', 'orange', 'purple', 'red', 'white', 'yellow']
}
CAR_HUMAN = CAR['red']
CAR_ROBOT = CAR['yellow']
CAR_SCALE = 0.15/max(CAR.values()[0].shape[:2])

LANE_SCALE = 10.
LANE_COLOR = (0.4, 0.4, 0.4) # 'gray'
LANE_BCOLOR = 'white'

STEPS = 100


def set_image(obj, data, scale=CAR_SCALE, x=[0., 0., 0., 0.]):
    ox = x[0]
    oy = x[1]
    angle = x[2]
    img = rotate(data, np.rad2deg(angle))
    h, w = img.shape[0], img.shape[1]
    obj.set_data(img)
    obj.set_extent([ox-scale*w*0.5, ox+scale*w*0.5, oy-scale*h*0.5, oy+scale*h*0.5])

class Scene(object):
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        set_image(self.human, CAR_HUMAN, x=self.snapshot.human.ix(self.t))
        xs = self.snapshot.human.ix(self.ts[self.ts<=self.t])
        self.h_past.set_data(xs[:, 0], xs[:, 1])
        xs = self.snapshot.human.ix(self.ts[self.ts>=self.t])
        self.h_future.set_data(xs[:, 0], xs[:, 1])
        for robot, traj in zip(self.robots, self.snapshot.robots):
            set_image(robot, CAR_ROBOT, x=traj.ix(self.t))
    def __init__(self, ax, snapshot):
        self.snapshot = snapshot
        self.ax = ax
        self.ax.set_aspect('equal', 'box-forced')

        self.grass = BboxImage(ax.bbox, interpolation='bicubic', zorder=-1000)
        self.ax.add_artist(self.grass)
        self.grass.set_data(GRASS)

        for lane in self.snapshot.lanes:
            path = Path([
                lane.p-LANE_SCALE*lane.m-lane.n*lane.w*0.5,
                lane.p-LANE_SCALE*lane.m+lane.n*lane.w*0.5,
                lane.q+LANE_SCALE*lane.m+lane.n*lane.w*0.5,
                lane.q+LANE_SCALE*lane.m-lane.n*lane.w*0.5,
                lane.p-LANE_SCALE*lane.m-lane.n*lane.w*0.5
            ], [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY
            ])
            ax.add_artist(PathPatch(path, facecolor=LANE_COLOR, lw=0.5, edgecolor=LANE_BCOLOR, zorder=-100))

        self.robots = [AxesImage(self.ax, interpolation='bicubic', zorder=0) for robot in self.snapshot.robots]
        for robot in self.robots:
            self.ax.add_artist(robot)

        self.human = AxesImage(self.ax, interpolation='bicubic', zorder=100)
        self.ax.add_artist(self.human)
        self.ts = np.linspace(0., self.snapshot.human.T, STEPS)
        xs = self.snapshot.human.ix(self.ts)
        self.h_future = ax.plot(xs[:, 0], xs[:, 1], zorder=50, linestyle='--', color='white', linewidth=1.)[0]
        self.h_past = ax.plot(xs[:, 0], xs[:, 1], zorder=40, linestyle='-', color='blue', linewidth=2.)[0]

        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.2, 0.8)


class Visualizer(object):
    def __init__(self, snapshot, save_on_close=None):
        self.snapshot = snapshot
        self.save_on_close = save_on_close
        self.T = self.snapshot.human.values()[0].T
        self.choice = None
    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value
        for scene in self.scenes:
            scene.t = self.t
    def select(self, key):
        self.choice = key
        plt.close(self.fig)
    def close(self, event):
        if self.save_on_close is not None:
            self.fig.savefig(self.save_on_close)
    def run(self):
        self.fig, self.ax = plt.subplots(1, len(self.snapshot.keys()), sharex=True, sharey=True, figsize=(13, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('close_event', self.close)
        self.scenes = [Scene(ax, self.snapshot.view(key)) for ax, key in zip(self.ax, self.snapshot.keys())]

        self.fig.subplots_adjust(bottom=0.15, top=0.85)

        box = self.fig.add_axes([0.15, 0.05, 0.7, 0.05])
        self.slider = Slider(box, 'Time', 0., self.T, valinit=0.)
        self.t = 0.
        def update_t(t):
            self.t = t
        self.slider.on_changed(update_t)

        def click(key):
            def f(event):
                self.select(key)
            return f
        self.buttons = []
        for ax, key in zip(self.ax, self.snapshot.keys()):
            box = ax.figbox
            box = self.fig.add_axes([box.x0, box.y1+0.05, box.width, 0.05])
            self.buttons.append(Button(box, 'Prefer {}'.format(key)))
            self.buttons[-1].on_clicked(click(key))

        plt.show()
    def key_press(self, event):
        if event.key=='escape':
            plt.close(self.fig)
        elif event.key=='r':
            self.slider.set_val(0.)
        elif event.key=='up':
            self.slider.set_val(min(max(self.t+0.2, 0), self.T))
        elif event.key=='down':
            self.slider.set_val(min(max(self.t-0.2, 0), self.T))
        elif event.key.lower() in [s.lower() for s in self.snapshot.keys()]:
            for key in self.snapshot.keys():
                if event.key.lower()==key.lower():
                    self.select(key)

def select(snapshot, save_on_close=None):
    vis = Visualizer(snapshot, save_on_close)
    vis.run()
    return vis.choice

def test_visualizer():
    from world import world
    for v in ['A', 'B']:
        traj = world.human[v]
        traj.x[0].set_value([0., 0., math.pi/2., 0.5])
        for u in traj.u:
            u.set_value([1 if v=='A' else -1, 1])
    world.robots[0].x[0].set_value([-0.13, 0., math.pi/2., 0.5])
    print select(world.dump())

if __name__=='__main__':
    from world import world
    for v in ['A', 'B']:
        traj = world.human[v]
        traj.x[0].set_value([0., 0., math.pi/2., 0.5])
        for u in traj.u:
            u.set_value([1 if v=='A' else -1, 1])
    world.robots[0].x[0].set_value([-0.13, 0., math.pi/2., 0.5])
    test_visualizer()
