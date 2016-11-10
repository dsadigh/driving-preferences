#!/usr/bin/env python

import anydbm
anydbm._defaultmod=__import__('gdbm')
from world import world
from sampling import sampler
import numpy as np
import shelve
import sys
import getopt
from utils import vector, matrix, Maximizer
import theano as th
import theano.tensor as tt
import theano.tensor.nnet as tn

W = np.array([ 0.16396617, -0.09797175,  0.87186561, -0.43615655, -0.11460562])

if __name__=='__main__':
    optlist, args = getopt.gnu_getopt(sys.argv, 'n:s:m:p:')
    opts = dict(optlist)
    db = shelve.open(args[1] if args[1].endswith('.db') else args[1]+'.db', writeback=True)
    N = int(opts.get('-n', 200))
    S = int(opts.get('-s', 2000))
    P = float(opts.get('-p', 1.))
    method = int(opts.get('-m', 0))
    if method==4:
        world.avg_case = True
    phis = []
    db['W'] = W
    db['N'] = N
    db['S'] = S
    db['P'] = P
    db['method'] = method
    if method==2:
        f = vector(len(W))
        phi = f/tt.maximum(1., f.norm(2))
        A = matrix(0, len(W))
        y = tt.dot(A, phi)
        p = tt.sum(tt.switch(y<0, 1., 0.))
        q = tt.sum(tt.switch(y>0, 1., 0.))
        #obj = tt.minimum(tt.sum(1.-tt.exp(-tn.relu(y))), tt.sum(1.-tt.exp(-tn.relu(-y))))
        obj = p*tt.sum(1.-tt.exp(-tn.relu(y))) + q*tt.sum(1.-tt.exp(-tn.relu(-y)))
        optimizer = Maximizer(obj, [f])
    if method==5:
        cand_phis = []
        for i in range(50):
            x = np.random.normal(size=len(W))
            cand_phis.append(x/np.linalg.norm(x))
    if method==6:
        cand_phis = []
        for i in range(50):
            world.randomize()
            cand_phis.append(world.ndf)
    if method==5 or method==6:
        f = tt.vector()
        A = matrix(0, len(W))
        y = tt.dot(A, f)
        p = tt.sum(tt.switch(y<0, 1., 0.))
        q = tt.sum(tt.switch(y>0, 1., 0.))
        obj = tt.minimum(tt.sum(1.-tt.exp(-tn.relu(y))), tt.sum(1.-tt.exp(-tn.relu(-y))))
        if method==5:
            obj = p*tt.sum(1.-tt.exp(-tn.relu(y))) + q*tt.sum(1.-tt.exp(-tn.relu(-y)))
        value = th.function([f], obj)
    db.sync()
    for t in range(N):
        sampler.A = phis
        samples = sampler.sample(S, T=10)
        #db['{}:samples'.format(t)] = samples
        db['{}:prog'.format(t)] = np.average([np.dot(x, W)/np.linalg.norm(W)/np.linalg.norm(x) for x in samples if np.linalg.norm(x)>1e-10])
        if method==0 or method==4:
            world.gen(samples)
            df = world.ndf
        elif method==1:
            world.gen(samples, dumb=True)
            df = world.ndf
        elif method==2:
            A.set_value(samples)
            f.set_value(np.random.normal(size=len(W)))
            optimizer.maximize()
            df = f.get_value()
            df = df/np.linalg.norm(df)#max(1., np.linalg.norm(df))
        elif method==3:
            df = np.random.normal(size=len(W))
            df = df/np.linalg.norm(df)
        elif method==5 or method==6:
            A.set_value(samples)
            df = cand_phis[np.argmax([value(x) for x in cand_phis])]
        if np.dot(W, df)>0:
            df = -df
        else:
            df = df
        if np.random.uniform()<P:
            phis.append(df)
        else:
            phis.append(-df)
        db['{}:snapshot'.format(t)] = world.dump()
        db['{}:phi'.format(t)] = phis[-1]
        db.sync()
    db.close()
