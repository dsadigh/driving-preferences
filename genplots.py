from pylab import *
from scipy.stats import gaussian_kde
import matplotlib.font_manager as font_manager

fontpath = '/Users/anari/Downloads/transfonter/Palatino-Roman.ttf'
#fontpath = '/Users/dsadigh/Downloads/PalatinoRoman.ttf'

prop = font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = prop.get_name()

def f(name):
    s = load(name)
    d = 3
    p = [1, 0, 4, 2, 3]
    s = array([x[p]/norm(x) for x in s])
    l = ['Road Boundary', 'Staying within Lanes', 'Keeping Speed', 'Heading', 'Collision Avoidance']
    l = ['$w_{}$ for {}'.format(i+1, T) for i, T in enumerate(l)]
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].set_xlim(-1, 1)
    ax[0].set_xlabel(l[d])
    ax[0].set_ylabel('PDF')
    ax[0].set_ylim(0, 2)
    ax[0].set_aspect('equal')
    x = s[:, d]
    t = np.linspace(-1, 1, 100)
    y = gaussian_kde(x)(t)
    ax[0].plot(t, y)
    for i, u in enumerate([x for x in range(5) if x!=d]):
        ax[i+1].set_xlim(-1, 1)
        ax[i+1].set_ylim(-1, 1)
        ax[i+1].set_aspect('equal')
        x, y = s[:, u], s[:, d]
        xy = vstack([x, y])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        ax[i+1].scatter(x, y, c=z, s=50, edgecolor='')
        ax[i+1].set_xlabel(l[u])
        ax[i+1].set_ylabel(l[d])
    fig.tight_layout()
    savefig('plots/frog-{}.pdf'.format(name))

def g(name):
    s = load(name)
    d = 3
    p = [1, 0, 4, 2, 3]
    s = array([x[p]/norm(x) for x in s])
    l = ['Road Boundary', 'Staying within Lanes', 'Keeping Speed', 'Heading', 'Collision Avoidance']
    l = ['$w_{}$ for {}'.format(i+1, T) for i, T in enumerate(l)]
    figure()
    xlim(-1, 1)
    ylim(0, 3)
    hs = []
    for i in range(5):
        x = s[:, i]
        t = np.linspace(-1, 1, 100)
        y = gaussian_kde(x)(t)
        h, = plot(t, y, lw=2)
        hs.append(h)
    ylabel('PDF')
    legend(hs, l)
    tight_layout()
    savefig('plots/pdf-{}.pdf'.format(name))

def h():
    figure()
    x = np.linspace(-4, 4, 2000)
    y1 = -log(1.+exp(-x))
    y2 = minimum(0, x)
    p1, = plot(x, y1, lw=2)
    p2, = plot(x, y2, lw=2)
    ylim(-4.5, 0.1)
    legend([p1, p2], ['$\log(f^1_\\varphi(\\bf{w}))$',
                      '$\log(f^2_\\varphi(\\bf{w}))$'], loc=4)
    gca().spines['right'].set_visible(False)
    gca().spines['top'].set_visible(False)
    gca().xaxis.set_ticks_position('bottom')
    gca().yaxis.set_ticks_position('left')
    gca().set_aspect(.7)
    tight_layout()
    savefig('plots/varphi.pdf')

f('samples.npy')
f('samples0.npy')
g('samples.npy')
g('samples0.npy')
h()
