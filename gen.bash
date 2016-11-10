#!/usr/bin/env bash

mkdir -p runs-$1

for i in {1..5}; do
  #./auto.py -m 0 -p $1 runs-$1/active_traj.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  #./auto.py -m 1 -p $1 runs-$1/random_traj.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  #./auto.py -m 2 -p $1 runs-$1/active_phi.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  #./auto.py -m 3 -p $1 runs-$1/random_phi.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  #./auto.py -m 4 -p $1 runs-$1/active_traj_alt.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  ./auto.py -m 5 -p $1 runs-$1/sampled_phi_alt.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
  ./auto.py -m 6 -p $1 runs-$1/sampled_traj_alt.`date -u +"%Y-%m-%dT%H:%M:%SZ"`.db
done
