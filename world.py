import numpy as np
from utils import vector, matrix, Maximizer
from dynamics import CarDynamics
from scipy.interpolate import interp1d
import pickle
import math
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
import itertools

def dumpable(obj):
    if isinstance(obj, Dumpable):
        return True
    if isinstance(obj, list) or isinstance(obj, tuple):
        return all([dumpable(x) for x in obj])
    if isinstance(obj, dict):
        return all([dumpable(x) for x in obj.values()])
    return False

def dump(obj):
    if isinstance(obj, list):
        return [dump(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple([dump(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: dump(v) for k, v in obj.iteritems()}
    elif isinstance(obj, Dumpable):
        return obj.dump()
    else:
        ret = Snapshot()
        for k, v in vars(obj).iteritems():
            if dumpable(v):
                setattr(ret, k, dump(v))
        return ret

def load(obj, snapshot):
    if isinstance(snapshot, list) or isinstance(snapshot, tuple):
        if len(obj)!=len(snapshot):
            raise Exception('Length mistmatch.')
        for x, y in zip(obj, snapshot):
            load(x, y)
    elif isinstance(snapshot, dict):
        for k, v in snapshot.iteritems():
            load(obj[k], v)
    elif isinstance(obj, Dumpable):
        obj.load(snapshot)
    else:
        for k, v in vars(snapshot).iteritems():
            if not hasattr(obj, k):
                continue
            load(getattr(obj, k), v)

class Snapshot(object):
    @property
    def answer(self):
        if not hasattr(self, '_answer'):
            self.answer = None
        return self._answer
    @answer.setter
    def answer(self, value):
        self._answer = value
    @property
    def user(self):
        if not hasattr(self, '_user'):
            self.user = None
        return self._user
    @user.setter
    def user(self, value):
        self._user = value
    def view(self, key):
        ret = Snapshot()
        for k, v in vars(self).iteritems():
            setattr(ret, k, v[key] if isinstance(v, dict) else v)
        return ret
    def keys(self):
        ret = set()
        for k, v in vars(self).iteritems():
            if isinstance(v, dict):
                ret = ret|set(v.keys())
        return list(sorted(ret))
    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self, f)
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Dumpable(object): pass

class Lane(Dumpable):
    def __init__(self, p, q, w):
        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w
        self.m = (self.q-self.p)/np.linalg.norm(self.q-self.p)
        self.n = np.asarray([-self.m[1], self.m[0]])
    def gaussian(self, x, width=0.5):
        d = (x[0]-self.p[0])*self.n[0]+(x[1]-self.p[1])*self.n[1]
        return tt.exp(-0.5*d**2/(width**2*self.w*self.w/4.))
    def direction(self, x):
        return tt.cos(x[2])*self.m[0]+tt.sin(x[2])*self.m[1]
    def shifted(self, m):
        return Lane(self.p+self.n*self.w*m, self.q+self.n*self.w*m, self.w)
    def dump(self):
        return self
    def load(self, snapshot):
        self.__init__(snapshot.p, self.q, self.w)

class Trajectory(Dumpable):
    def __init__(self, T, dyn, x0=None):
        self.x = [vector(4) if x0 is None else x0]
        self.u = [vector(2) for _ in range(T)]
        self.dyn = dyn
        for t in range(T):
            self.x.append(dyn(self.x[t], self.u[t]))
    def dump(self):
        us = [u.get_value() for u in self.u]
        xs = [self.x[0].get_value()]
        for t in range(len(self.u)):
            xs.append(self.dyn.compiled(xs[t], us[t]))
        return TrajectorySnapshot(xs, us)
    def load(self, snapshot):
        if len(snapshot.x)!=len(self.x) or len(snapshot.u)!=len(self.u):
            raise Exception('Trajectory length mismatch.')
        for x, u, y in zip(snapshot.x[:-1], snapshot.u, snapshot.x[1:]):
            if sum(np.abs(y-self.dyn.compiled(x, u)))>1e-5:
                raise Exception('Dynamics mistmatch.')
        self.x[0].set_value(snapshot.x[0])
        for u, v in zip(self.u, snapshot.u):
            u.set_value(v)

class TrajectorySnapshot(object):
    def __init__(self, xs, us):
        self.x = [np.asarray(x) for x in xs]
        self.u = [np.asarray(u) for u in us]
        self.ix = interp1d(np.asarray(range(self.T+1)), np.asarray(self.x), axis=0, kind='cubic')
        self.iu = interp1d(np.asarray(range(self.T)), np.asarray(self.u), axis=0, kind='cubic')
    def __getstate__(self):
        return (self.x, self.u)
    def __setstate__(self, state):
        xs, us = state
        self.__init__(xs, us)
    @property
    def T(self):
        return len(self.u)

class Feature(object):
    def __init__(self, f, name):
        self.f = f
        self.name = name
    def total(self, world, *args, **vargs):
        return sum(self.f(world.moment(t, *args, **vargs))
                for t in range(1, world.T+1))

def feature(f):
    return Feature(f, f.__name__)

@feature
def lanes(moment):
    return sum(lane.gaussian(moment.human) for lane in moment.lanes)

@feature
def fences(moment):
    return sum(fence.gaussian(moment.human) for fence in moment.fences)

@feature
def roads(moment):
    return sum(road.direction(moment.human) for road in moment.roads)

@feature
def speed(moment):
    return (moment.human[3]-1.)**2

def car_gaussian(x, y, height=.07, width=.03):
    d = (x[0]-y[0], x[1]-y[1])
    dh = tt.cos(x[2])*d[0]+tt.sin(x[2])*d[1]
    dw = -tt.sin(x[2])*d[0]+tt.cos(x[2])*d[1]
    return tt.exp(-0.5*(dh*dh/(height*height)+dw*dw/(width*width)))

@feature
def cars(moment):
    return sum(car_gaussian(robot, moment.human) for robot in moment.robots)

features = [lanes, fences, roads, cars, speed]

class Moment(object):
    def __init__(self, world, t, version):
        self.lanes = world.lanes
        self.fences = world.fences
        self.roads = world.roads
        self.robots = [robot.x[t] for robot in world.robots]
        self.human = world.human[version].x[t]

class World(object):
    def __init__(self, T, dyn, lanes=[], fences=[], roads=[], ncars=1):
        self.lanes = lanes
        self.fences = fences
        self.roads = roads
        self.robots = [Trajectory(T, dyn) for _ in range(ncars)]
        self.x0 = vector(4)
        self.human = {
            'A': Trajectory(T, dyn, self.x0),
            'B': Trajectory(T, dyn, self.x0)
        }

        self.bounds = {}
        uB = [(-2., 2.), (-1., 1.)]
        xB = [(-0.15, 0.15), (-0.1, 0.2), (math.pi*0.4, math.pi*0.6), (0., 1.)]
        for robot in self.robots:
            for u in robot.u:
                self.bounds[u] = uB
            self.bounds[robot.x[0]] = xB
        for human in self.human.values():
            for u in human.u:
                self.bounds[u] = uB
            self.bounds[human.x[0]] = xB

        self.w = vector(len(features))
        self.samples = matrix(0, len(features))
    @property
    def df(self):
        if not hasattr(self, '_df'):
            self._df = th.function([], self.fvector('A')-self.fvector('B'))
        return self._df()
    @property
    def ndf(self):
        df = self.df
        return df/max(1, np.linalg.norm(df))
    @property
    def human_optimizer(self):
        if not hasattr(self, '_human_optimizer'):
            self._human_optimizer = {
                v: Maximizer(tt.dot(self.w, self.fvector(v)), self.human[v].u)
                for v in self.human
            }
        return self._human_optimizer
    @property
    def optimizer(self):
        if not hasattr(self, '_optimizer'):
            df = self.fvector('A')-self.fvector('B')
            phi = df/(1+tn.relu(df.norm(2)-1))
            y = tt.dot(self.samples, phi)
            p = tt.sum(tt.switch(y<0, 1., 0.))
            q = tt.sum(tt.switch(y>0, 1., 0.))
            if not hasattr(self, 'avg_case'):
                obj = tt.minimum(tt.sum(1.-tt.exp(-tn.relu(y))), tt.sum(1.-tt.exp(-tn.relu(-y))))
            else:
                obj = p*tt.sum(1.-tt.exp(-tn.relu(y)))+q*tt.sum(1.-tt.exp(-tn.relu(-y)))
            variables = [self.x0]
            for robot in self.robots:
                variables += [robot.x[0]]+robot.u
            for human in self.human.values():
                variables += human.u
            self._optimizer = Maximizer(obj, variables)
        return self._optimizer
    def fvector(self, *args, **vargs):
        return tt.stacklists([f.total(self, *args, **vargs) for f in features])
    @property
    def T(self):
        return len(self.human.values()[0].u)
    def dump(self):
        return dump(self)
    def load(self, snapshot):
        load(self, snapshot)
    def moment(self, t, version):
        return Moment(self, t, version)
    def randomize(self):
        for x, B in self.bounds.iteritems():
            x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
    def gen(self, samples, dumb=False, random_init=False):
        self.samples.set_value(samples)
        for x, B in self.bounds.iteritems():
            x.set_value(np.array([np.random.uniform(a, b) for (a, b) in B]))
        if not random_init:
            for opt in self.human_optimizer.values():
                if len(samples)==0:
                    self.w.set_value(np.random.normal(size=len(features)))
                else:
                    self.w.set_value(samples[np.random.choice(len(samples)), :])
                opt.maximize(bounds=self.bounds)
        if not dumb:
            self.optimizer.maximize(bounds=self.bounds)

lane = Lane([0., -1.], [0., 1.], 0.13)
road = Lane([0., -1.], [0., 1.], 0.13*3)
env1 = {
    'lanes': [
        lane.shifted(0),
        lane.shifted(-1),
        lane.shifted(1)
    ],
    'fences': [
        lane.shifted(2),
        lane.shifted(-2)
    ],
    'roads': [
        road
    ]
}

world = World(5, CarDynamics(), **env1)
for v in ['A', 'B']:
    traj = world.human[v]
    traj.x[0].set_value([0., 0., math.pi/2., .5])
    for u in traj.u:
        u.set_value([0., 1.])
world.robots[0].x[0].set_value([-0.13, 0., math.pi/2., 0.5])

example1 = world.dump()

if __name__=='__main__':
    world.gen(np.zeros((0, len(features))))
    quit()
    from visual import select
    for opt in world.human_optimizer.values():
        world.w.set_value([np.random.normal() for _ in features])
        opt.maximize(bounds=world.bounds)
    world.optimizer.maximize(bounds=world.bounds)
    select(world.dump())
