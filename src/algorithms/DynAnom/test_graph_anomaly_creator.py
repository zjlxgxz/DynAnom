# path
import sys
import os
import functools
from os.path import join as os_join 
from os.path import dirname as up_dir
# append relative path
# current:  */git_project/kdd22-anomaly/src/algorithms/DynAnom
project_root = functools.reduce(lambda x,y: up_dir(x), range(4) ,os.path.abspath(__file__) )
sys.path.append(os_join(project_root,'src'))
sys.path.append(os_join(project_root,'src',"algorithms", "DynAnom"))
import argparse
import pprint
import scipy.stats as stats


# home-made lib
import graph_data_utils
import DynamicPPE

# computation
import numpy as np
import networkx as nx
from time import process_time as ptime
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm as tqdm
from networkx.algorithms.components import number_connected_components as num_cc
from networkx.algorithms.components import is_weakly_connected as is_wk_cc
import pickle
import numba
from scipy.sparse.linalg import norm as sp_norm

# import multiprocessing


## plot related
import matplotlib
matplotlib.rcParams['font.family'] = "serif"
import matplotlib.pyplot as plt
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', markersize=2)
DISPLAY_SNAPSHOT_INVERVAL = 2


def test_graph_stream(  path, dataset_name, timeStep, initSS , \
                        make_inject_anomaly, inject_mode, seed):
    # for output file format
    data_config_str = f'dataset_{dataset_name}-initSS_{initSS}-timeStep_{timeStep}'
    print (f'data: {data_config_str}')

    ################################################################
    ####    Graph snapshots construcution
    ################################################################
    # Some edge stream may only have 10 edges,  but results in 100 snapshots.
    # In this case, a great amount of snapshots are equal since no edge event happens between many of them.
    
    #timeStep = 60 #time interval of one snapshot, e.g.: 60 mins; or 60 sec
    n, m, timeNum, nodes, edges, snapshots, anomalous_nodes_desc = graph_data_utils.read_data(path, timeStep)
    attackLimit = 50 # threshold for attack snapshot anomaly
    Inject_param = make_inject_anomaly 
    Inject_mode = inject_mode
    injectNum = 20 # number of snapshots
    injectSize = attackLimit + 20 # 70
    numSS = timeNum//timeStep + 1
    testNum = numSS - initSS
    rs = np.random.RandomState(seed)

    print (f'#nodes:{n} , #edges:{m}, timeNum:{timeNum}, total snapshots: {numSS}, init snapshots: {initSS}, effective snapshots :{len(snapshots)}, #Anomalous-nodes: {len(anomalous_nodes_desc)}')

    # choose which snapshots should be injected with random edges as anomaly 
    # it will add new effective snapshots (step).
    if Inject_param is True:
        snapshots, injectSS = graph_data_utils.inject_snapshot(injectNum, initSS, testNum, snapshots, rs)
        print (f'#injectSS:{len(injectSS)}, #snapshots: {len(snapshots)}')

    # different data structure of graph 
    A = dict()
    dir_g = nx.DiGraph()
    sp_lil_M = lil_matrix((n, n))
    sp_lil_M_weight = lil_matrix((n, n))


    eg = 0
    injected = 0

    all_edges_list = []
    ################################################################
    ####    Iterate snapshot and calculate anomaly scores 
    ################################################################
    with tqdm(range(len(snapshots))) as t:
        for loop_i, ss in enumerate(t):
            # Incrementally update graph
            while edges[eg][2] < snapshots[ss]*timeStep:
                # If this edge happens before the time border, inject edges/update graph. 
                graph_data_utils.inject(A, edges[eg][0], edges[eg][1], 1, dir_g, sp_lil_M, sp_lil_M_weight)
                all_edges_list.append(edges[eg])# edges = [u,v,t,l]
                
                eg+=1 # eg only records the number of edge in the original graph, excluding augmented edges
                # reach to the end, break
                if eg == len(edges):
                    break

            # Artifically augmented anomalous edges in this snapshot.
            # injected_anomalous_edges = []
            if Inject_param is True and injectSS[injected] == snapshots[ss]:
                fake_edge_time = snapshots[ss]*timeStep - 1
                fake_edges = graph_data_utils.inject_anomaly_v2(Inject_mode, A, n, injectSize, rs)
                print (injected, fake_edges.shape[0])
                a = len(all_edges_list)
                for _ in range(fake_edges.shape[0]):
                    e = fake_edges[_,:]
                    t,u,v,l = fake_edge_time, e[0],e[1],1
                    all_edges_list.append([u,v,t,l])
                print ('added anomalous edges:', len(all_edges_list) - a )
                injected+=1
                if(injected == len(injectSS)):
                    Inject_param = False
            

    output_path = os.path.join(project_root, 'toy-data', f'{dataset_name}_inject_{inject_mode}.txt')
    with open (output_path,'w') as wf:
        for e in all_edges_list:
            print (f'{e[2]} {e[0]} {e[1]} {e[3]}', file = wf)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name", type=str, default='enron',
                        help="the testing dataset, could be {enron, darpa}")
    parser.add_argument("-timeStep", type=int, default=0,
                    help=f'timeStep = 60*60*24 for enron,  60 for darpa ')
    parser.add_argument("-initSS", type=int, default=256,
                    help=f' default 256')
    parser.add_argument("-make_inject_anomaly", 
                        help=" to inject artifical anoamly", action="store_true")
    parser.add_argument("-seed", type=int, default=621,
                    help=f'the randomseed')
    parser.add_argument("-inject_mode", type=int, default=1,
                    help=f'1: InjectS 2: InjectW 3: Inject-MultiW')
    
    args = parser.parse_args()
        # data related
    dataset_name = args.dataset_name #enron darpa
    assert dataset_name in ['enron', 'darpa','eucore']
    if args.dataset_name == 'enron':
        args.timeStep = 60*60*24 # seconds per day 
        args.initSS = 256 # 256 for enron
        data_path  = os_join(project_root,'toy-data', f'{dataset_name}.txt') #enron.txt  darpa.txt
    elif args.dataset_name == 'darpa':
        args.timeStep = 60 #500 #60 # seconds #87726
        args.initSS = 256 #50 # 256 for darpa
        data_path  = os_join(project_root,'toy-data', f'{dataset_name}.txt') #enron.txt  darpa.txt
    # added 
    elif args.dataset_name == 'eucore':
        args.timeStep = 3600*24*7 #total 114, effective 78
        args.initSS = 25 # ~first 1/4 snapshots
        data_path  = os_join(project_root,'toy-data', f'{dataset_name}.txt') 


    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))

    timeStep = args.timeStep
    initSS = args.initSS
    make_inject_anomaly = args.make_inject_anomaly
    inject_mode = args.inject_mode
    seed = args.seed
    
    test_graph_stream(  data_path, dataset_name, timeStep, initSS, \
                        make_inject_anomaly, inject_mode, seed )

# python test_graph_anomaly_creator.py -dataset_name eucore -make_inject_anomaly -inject_mode 1
# python test_graph_anomaly_creator.py -dataset_name eucore -make_inject_anomaly -inject_mode 2
# python test_graph_anomaly_creator.py -dataset_name eucore -make_inject_anomaly -inject_mode 3