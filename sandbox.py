from utils import scalar, vector, matrix, Maximizer, shape, randomize
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
import numpy as np
import math

T = 5
dt = 0.1
friction = 1.

def f(x, u):
    return tt.stacklists([
        x[3]*tt.cos(x[2]),
        x[3]*tt.sin(x[2]),
        x[3]*u[0],
        u[1]-x[3]*friction
    ])

def traj(x0, us):
    x = [x0]
    for t, u in enumerate(us):
        x.append(x[t]+dt*f(x[t], u))
    return x

uH1 = [vector(2) for _ in range(T)]
xH1 = traj(vector(4), uH1)

uH2 = [vector(2) for _ in range(T)]
xH2 = traj(xH1[0], uH2)

uR = [vector(2) for _ in range(T)]
xR = traj(vector(4), uR)

def fence(xH, xR):
    return sum(tt.exp(-0.5*(xH[0]-t)**2/(0.05**2)) for t in [-0.2, 0.2])

def lane(xH, xR):
    return sum(tt.exp(-0.5*(xH[0]-t)**2/(0.05**2)) for t in [-0.13, 0., 0.13])

def speed(xH, xR):
    return (xH[3]-1.)**2

def gaussian(xH, xR):
    d = (xH[0]-xR[0], xH[1]-xR[1])
    theta = xR[2]
    dh = tt.cos(theta)*d[0]+tt.sin(theta)*d[1]
    dw = -tt.sin(theta)*d[0]+tt.cos(theta)*d[1]
    return tt.exp(-0.5*(dh**2/(0.07**2)+dw**2/(0.03**2)))

features = [fence, lane, speed, gaussian]

def fvector(xH, xR):
    return tt.stacklists([
        sum(feature(xh, xr) for xh, xr in zip(xH, xR))
        for feature in features
    ])

f1 = fvector(xH1, xR)
f2 = fvector(xH2, xR)

for u in uH1+uH2+uR:
    randomize(u)

xH1[0].set_value([0., 0., math.pi/2., 0.5])
xR[0].set_value([-0.1, 0., math.pi/2., 0.3])

df = f1-f2

N = 1000
D = len(features)

#def gaussian():
#    return np.random.normal(size=D)

w = df
#w.set_value(gaussian())
v = w/w.norm(2)
X = matrix(N, D)
X.set_value(np.random.normal(size=(N, D)))

C = X.get_value()
cand = np.linalg.eigh(np.dot(C.T, C))[1][-1, :]

y = tt.dot(X, v)
o1 = tt.sum(1-tt.exp(-tn.relu(y)))
o2 = tt.sum(1-tt.exp(-tn.relu(-y)))
obj = tt.minimum(o1, o2)

f = th.function([], obj)
g = th.function([], v)
print f()
print g()

optimizer = Maximizer(obj, uH1+uH2+uR)
optimizer.maximize()
print f()
print g()
