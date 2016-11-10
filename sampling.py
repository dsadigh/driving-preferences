import pymc as mc
import numpy as np
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn
import matplotlib.pyplot as plt
from theano.ifelse import ifelse
from scipy.stats import gaussian_kde
from utils import matrix
from world import features

def plot_dist(data, labels=None):
    N = data.shape[0]
    D = data.shape[1]
    if labels is None:
        labels = range(D)
    fig, ax = plt.subplots(D, D, sharex=False, sharey=False, figsize=(16, 16))
    ax = np.flipud(ax)
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i==0:
                ax[i, j].set_xlabel(l2)
            if j==0:
                ax[i, j].set_ylabel(l1)
            ax[i, j].set_xlim(-1, 1)
            if i!=j:
                ax[i, j].set_ylim(-1, 1)
                ax[i, j].set_aspect('equal', 'box')
            ax[i, j].grid()
            if i==j:
                x = data[:, i]
                t = np.linspace(-1, 1, 100)
                y = gaussian_kde(x)(t)
                ax[i, j].plot(t, y)
            else:
                x, y = data[:, j], data[:, i]

                xy = np.vstack([x, y])
                try:
                    z = gaussian_kde(xy)(xy)
                except:
                    z = np.ones(len(x))
                idx = z.argsort()
                x, y, z = x[idx], y[idx], z[idx]

                ax[i, j].scatter(x, y, c=z, s=50, edgecolor='')
    return fig

class Sampler(object):
    def __init__(self, D):
        self.D = D
        self.Avar = matrix(0, self.D)
        x = tt.vector()
        self.f = th.function([x], -tt.sum(tn.relu(tt.dot(self.Avar, x))))
    @property
    def A(self):
        return self.Avar.get_value()
    @A.setter
    def A(self, value):
        if len(value)==0:
            self.Avar.set_value(np.zeros((0, self.D)))
        else:
            self.Avar.set_value(np.asarray(value))
    def sample(self, N, T=50, burn=1000):
        x = mc.Uniform('x', -np.ones(self.D), np.ones(self.D), value=np.zeros(self.D))
        def sphere(x):
            if (x**2).sum()>=1.:
                return -np.inf
            else:
                return self.f(x)
        p1 = mc.Potential(
            logp = sphere,
            name = 'sphere',
            parents = {'x': x},
            doc = 'Sphere potential',
            verbose = 0)
        chain = mc.MCMC([x])
        chain.use_step_method(mc.AdaptiveMetropolis, x, delay=burn)
        chain.sample(N*T+burn, thin=T, burn=burn)
        return x.trace()

sampler = Sampler(len(features))

if __name__ == '__main__':
    s = Sampler(3)
    x = np.array([-0.5, -0.2, 0.3])
    x = x/np.linalg.norm(x)
    print 'Target: {}'.format(x)
    A = []
    for i in range(100):
        v = np.random.normal(size=3)
        if np.dot(v, x)>=0:
            A.append(-v)
        else:
            A.append(v)
    A = [x if np.random.uniform()<0.7 else -x for x in A]
    s.A = A
    samples = s.sample(1000)
    samples = np.array([x/np.linalg.norm(x) for x in samples])
    plot_dist(samples)
    plt.show()
