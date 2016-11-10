#!/usr/bin/env python

from flask import Flask, render_template, jsonify, request, send_from_directory
import threading, webbrowser, os, os.path
from visual import test_visualizer, select
from world import Snapshot, world, features
from sampling import sampler, plot_dist
from Queue import Queue
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys

app = Flask(__name__)
q = Queue()

def snapshots():
    files = list(sorted([x for x in os.listdir('data') if x.endswith('.pickle')]))
    for filename in files:
        yield Snapshot.load('data/{}'.format(filename))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def get_data():
    files = list(sorted([x for x in os.listdir('data') if x.endswith('.pickle')]))
    def record(filename):
        snapshot = Snapshot.load('data/{}'.format(filename))
        ret = [
            filename[:-len('.pickle')],
            os.path.getmtime('data/{}'.format(filename)),
            snapshot.user,
            None,
            snapshot.answer
        ]
        return ret
    return jsonify({'data':[record(filename) for filename in files]})

@app.route('/snapshot/<path:path>')
def send_snapshot(path):
    return send_from_directory('snapshots', '{}.png'.format(path))

@app.route('/plot/<path:path>')
def send_plot(path):
    return send_from_directory('plots', '{}.png'.format(path))

@app.route('/do', methods=['POST'])
def do():
    if isinstance(request.json, list):
        for x in request.json:
            q.put(x)
    else:
        q.put(request.json)
    return jsonify('Done')

@app.route('/exit', methods=['POST'])
def exit():
    threading.Timer(1.5, lambda: os._exit(0)).start()
    return jsonify('Success')

def perform(vals):
    method = vals.get('method')
    ID = vals.get('id')
    def visualize(ID):
        snapshot = Snapshot.load('data/{}.pickle'.format(ID))
        ans = select(snapshot, 'snapshots/{}.png'.format(ID))
        if ans is not None:
            snapshot.answer = ans
            snapshot.save('data/{}.pickle'.format(ID))
    if method=='visualize':
        visualize(ID)
    elif method=='new':
        if not os.path.exists('samples.npy'):
            samples = np.zeros((0, len(features)))
        else:
            samples = np.load('samples.npy')
        world.gen(samples)
        snapshot = world.dump()
        name = datetime.datetime.now().isoformat()
        snapshot.save('data/{}.pickle'.format(name))
        if not vals.get('silent'):
            visualize(name)
    elif method=='sample':
        A = []
        for snapshot in snapshots():
            world.load(snapshot)
            df = world.ndf
            if snapshot.answer=='A':
                A.append(-df)
            elif snapshot.answer=='B':
                A.append(df)
        sampler.A = A
        samples = sampler.sample(5000)
        np.save('samples.npy', samples)
        fig = plot_dist(samples, labels = [f.name for f in features])
        plt.tight_layout()
        fig.savefig('plots/samples.png')
        plt.close(fig)
        fig = plot_dist(np.array([x/np.linalg.norm(x) for x in samples]), labels = [f.name for f in features])
        plt.tight_layout()
        fig.savefig('plots/nsamples.png')
        plt.close(fig)

if __name__ == '__main__':
    threading.Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
    threading.Thread(target = lambda: app.run(threaded=True, port=5000, debug=False)).start()
    while True:
        perform(q.get())
