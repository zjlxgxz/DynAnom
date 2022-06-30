import numpy as np 
import random 
import networkx as nx
import pandas as pd
from tqdm import tqdm as tqdm
import scipy as sp
import scipy.sparse  # call as sp.sparse
from scipy.sparse import csr_matrix, lil_matrix
from time import process_time as ptime
import itertools

def read_data(path, stepSize):
    """ Divide stream of edges (u,v,timestamp,label) into snapshots based on time interval (stepSize).
        Assume (u,v,timestamp,label) are all integers.

    Args:
        path (str): The edge stream file
        stepSize (int): The time period of each snapshot (e.g., 60 mins)

    Returns:
        n (int): #nodes
        m (int): #edges
        timeNum (int): number of unique timestamp
        sort_node (list): the sorted node list by raw node id (or node description)
        sort_edge (list): the sorted edge list by edge timestamp
        snapshots (list): the step boarder of each effective snapshot. 
                            e.g. 
                                snapshots[0] = 1, so that the edges of the timestamp before 1*stepSize are in the 1st snapshot
                                snapshots[1] = 30, so that the edges (in (1*stepSize, 30*stepSize) are in the 2nd effective snapshot (it means that 2-29 steps has no edge event).
    """
    edges = []
    nodes = []
    times = []
    snapshots = []
    
    anomalous_nodes_desc = set([])
    
    with open(path,'r') as f:
        for line in tqdm(f):
            tokens = line.strip().split(' ')
            #edges.push_back(timeEdge(tokens));
            times.append(int(tokens[0]))
            nodes.append(int(tokens[1]))
            nodes.append(int(tokens[2]))
            edges.append(
                            [
                                int(tokens[1]),# u
                                int(tokens[2]),# v
                                int(tokens[0]),# t
                                int(tokens[3]) # label
                            ]
                        )
            # get the anomalous nodes raw-description
            if int(tokens[3]) != 0:
                anomalous_nodes_desc.add(int(tokens[1]))
                anomalous_nodes_desc.add(int(tokens[2]))
    
    sort_time = sorted(times) #ascending order
    sort_node = sorted(nodes) # assume node are sort-able
    sort_edge = sorted(edges, key = lambda x:x[2]) # sort edges by time
    n = len(set(sort_node))
    m = len(sort_edge)
    print (f'min time: {sort_time[0]},  max time: { sort_time[-1]}')
    initial_time = sort_time[0] # the very first time
    timeNum = sort_time[-1] - initial_time + 1 # the total time span. timeNum//stepSize is the total steps.
    #reset edge time, let the earliest time to be 0
    for i in range(m):
        edges[i][2] = edges[i][2]-initial_time
    
    snapshot_t = 1
    for i in range(m):
        e = sort_edge[i]
        if e[2] > snapshot_t*stepSize:
            snapshots.append(snapshot_t)
            # two consecutive time may be greater than stepSize, 
            # so use t//stepSize +1, instead of t++,
            # thus len(snapshots) may not equal to total steps. 
            snapshot_t = e[2]//stepSize + 1
    if snapshot_t != snapshots[-1]: # handle residual for final steps
        snapshots.append(snapshot_t)
        
    return n, m, timeNum, sort_node, sort_edge, snapshots, anomalous_nodes_desc


def read_data_for_exp3(path, stepSize):
    """ Divide stream of edges (u,v,timestamp,label) into snapshots based on time interval (stepSize).
        Assume (u,v,timestamp,label) are all integers.

    Args:
        path (str): The edge stream file
        stepSize (int): The time period of each snapshot (e.g., 60 mins)

    Returns:
        n (int): #nodes
        m (int): #edges
        timeNum (int): number of unique timestamp
        sort_node (list): the sorted node list by raw node id (or node description)
        sort_edge (list): the sorted edge list by edge timestamp
        snapshots (list): the step boarder of each effective snapshot. 
                            e.g. 
                                snapshots[0] = 1, so that the edges of the timestamp before 1*stepSize are in the 1st snapshot
                                snapshots[1] = 30, so that the edges (in (1*stepSize, 30*stepSize) are in the 2nd effective snapshot (it means that 2-29 steps has no edge event).
    """
    edges = []
    nodes = []
    times = []
    snapshots = []
    
    anomalous_nodes_desc = set([])
    
    with open(path,'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            #edges.push_back(timeEdge(tokens));
            times.append(int(tokens[0]))
            nodes.append(int(tokens[1]))
            nodes.append(int(tokens[2]))
            edges.append(
                            [
                                int(tokens[1]),# u
                                int(tokens[2]),# v
                                int(tokens[0]),# t
                                int(tokens[3]) # label
                            ]
                        )
            # get the anomalous nodes raw-description
            if int(tokens[3]) != 0:
                anomalous_nodes_desc.add(int(tokens[1]))
                anomalous_nodes_desc.add(int(tokens[2]))
    
    sort_time = sorted(times) #ascending order
    sort_node = sorted(nodes) # assume node are sort-able
    sort_edge = sorted(edges, key = lambda x:x[2]) # sort edges by time
    n = len(set(sort_node))
    m = len(sort_edge)
    print (f'min time: {sort_time[0]},  max time: { sort_time[-1]}')
    initial_time = sort_time[0] # the very first time
    timeNum = sort_time[-1] - initial_time + 1 # the total time span. timeNum//stepSize is the total steps.
    #reset edge time, let the earliest time to be 0
    for i in range(m):
        edges[i][2] = edges[i][2]-initial_time
    
    snapshot_t = 1
    for i in range(m):
        e = sort_edge[i]
        if e[2] > snapshot_t*stepSize:
            snapshots.append(snapshot_t)
            # two consecutive time may be greater than stepSize, 
            # so use t//stepSize +1, instead of t++,
            # thus len(snapshots) may not equal to total steps. 
            snapshot_t = e[2]//stepSize + 1
    if snapshot_t != snapshots[-1]: # handle residual for final steps
        snapshots.append(snapshot_t)
        
    return n, m, timeNum, sort_node, sort_edge, snapshots, anomalous_nodes_desc, initial_time, sort_time

#injectNum=50, initSS, testNum, snapshots
def inject_snapshot(injectNum, initSS, testNum, snapshots, rs):
    """ Mark the injection snapshots in injectSS according to testNum(the range of injected snapshots) 
        and injectNum(how many injected snapshots)
    """
    injected = 0
    injectSS = []
    
    for _ in range(injectNum):
        while True:
            injected = initSS + rs.randint(0,testNum) # the injected anomly happening at initSS + random
            if injected not in injectSS:
                injectSS.append(injected)
                break
        if injected not in snapshots: # the random one is greater than the last one 
            snapshots.append(injected)
    sort_snapshot = sorted(snapshots)
    sort_injectSS = sorted(injectSS)
    #print (sort_snapshot, sort_injectSS)
    
    return sort_snapshot, sort_injectSS

class NeighborList():
    """Graph node class
    """
    def __init__(self, ppv_id):
        """Init a new node with unique node id

        Args:
            ppv_id (int): the node id of graph from [0,N)
        """
        self.neighbors = dict() # this is the out-neighbors
        self.total_w = 0
        self.ppv_id = ppv_id

def inject(A, u, v, num, dir_g, sp_lil_M, sp_lil_M_weight):
    """Add new edges, maintain graph for several data structure"""
    # what happen if v not in A? 
    
    if u not in A.keys():
        A[u] = NeighborList(ppv_id = len(A.keys()))
    
    if v not in A.keys():
        A[v] = NeighborList(ppv_id = len(A.keys()))
    
    # for self 
    if v in A[u].neighbors.keys():
        A[u].neighbors[v] += num
        A[u].total_w += num
    else:
        A[u].neighbors[v] = num
        A[u].total_w += num
    
    # for networkx
    #dir_g.add_edge(ppv_id_u, ppv_id_v, weight = 1)

        
    # update graph 
    # for key, node_info in A.items():
    #     u = node_info.ppv_id
    #     for v, _weight in node_info.neighbors.items():

    # for scipy pagerank
    ppv_id_v = A[v].ppv_id
    ppv_id_u = A[u].ppv_id
    # this is slow in python
    sp_lil_M_weight[ppv_id_v, ppv_id_u] += num
    sp_lil_M[ppv_id_v, ppv_id_u] = num
    

def inject_v2(A, u, v, num):
    """Add new edges, maintain graph for several data structure"""
    # what happen if v not in A? 
    
    if u not in A.keys():
        A[u] = NeighborList(ppv_id = len(A.keys()))
    
    if v not in A.keys():
        A[v] = NeighborList(ppv_id = len(A.keys()))
    
    # for self 
    if v in A[u].neighbors.keys():
        A[u].neighbors[v] += num
        A[u].total_w += num
    else:
        A[u].neighbors[v] = num
        A[u].total_w += num
    

def inject_anomaly(scenario, A, n, edgeNum, rs):
    if scenario == 1: 
        # add designated weight to one edge
        u = 0
        while True:
            u = rs.choice(tuple(A.keys()))
            if len(A[u].neighbors.keys()) > 0:
                break
        v = rs.random.choice(tuple(A[u].neighbors.keys()))
        inject(A, u, v, edgeNum)
        return edgeNum
    
    elif scenario == 2:
        #distribute whole weights to out-edges, every edge rec one or more
        u = 0
        while True:
            u = rs.choice(tuple(A.keys()))
            if len(A[u].neighbors.keys()) > 0:
                break
        
        if len(A[u].neighbors.keys())<= edgeNum:
            consumed = 0
            for e in A[u].neighbors.keys():
                inject(A, u, e, 1)
                consumed += 1
                if consumed == edgeNum:
                    break
        else:
            inj_per_edge = edgeNum//len(A[u].neighbors.keys())
            residual_num = edgeNum - inj_per_edge*len(A[u].neighbors.keys())
            for e in A[u].neighbors.keys():
                inject(A, u, e, inj_per_edge)
            # random pick one to distribute edges
            v = rs.choice(tuple(A[u].neighbors.keys()))
            inject(A, u, v, residual_num)
        return edgeNum
    
    elif scenario == 3:
        # distribute whole edges to unseen edges from one source
        u = rs.choice(tuple(A.keys()))
        # connect to u's unseen neighbors
        v = set([])
        u_ngbr = A[u].neighbors.keys()
        while len(v)<edgeNum:
            _v = rs.randint(0, n)
            if _v not in u_ngbr and _v not in v:
                v.add(_v)

        for e in v:
            inject(A, u, e, 1)
        
        return edgeNum
            
    elif scenario == 4:
        pass
    
    return None


def inject_anomaly_v2(scenario, A, n, edgeNum, rs):
    k = 10
    if scenario == 1:
        # simulating low degree nodes coonect to high degree nodes
        # node, degree list
        node_degree = [ [key, A[key].total_w ] for key in A.keys() ]
        sorted_node_degree = sorted(node_degree,key = lambda x: x[1])[::-1]
        sorted_node = np.array(sorted_node_degree)[:,0]
        sorted_nodes = sorted_node[:sorted_node.shape[0]//100]
        high_deg_node = rs.choice(sorted_nodes,1)[0]
        # its neighbors
        unconnected_nodes = set(A.keys()) - set(A[high_deg_node].neighbors.keys())
        u_s =  rs.choice(list(unconnected_nodes),k)
        unique_fake_edges = np.array([[high_deg_node, v] for v in u_s])
        number_of_rows = unique_fake_edges.shape[0]
        rand_edge_idx = rs.choice(number_of_rows, size = edgeNum)
        fake_edges = unique_fake_edges[rand_edge_idx, :]
        for _edge in fake_edges:
            u = _edge[0]
            v = _edge[1]
            inject_v2(A, u, v, 1)
        return fake_edges
    
    elif scenario == 2:
        # pick 2 nodes uniformaly at random, and connect them all
        u_s =  rs.choice(tuple(A.keys()),2)
        unique_fake_edges = np.array(list(itertools.permutations(u_s.tolist())))
        number_of_rows = unique_fake_edges.shape[0]
        rand_edge_idx = rs.choice(number_of_rows, size = edgeNum)
        fake_edges = unique_fake_edges[rand_edge_idx, :]
        for _edge in fake_edges:
            u = _edge[0]
            v = _edge[1]
            inject_v2(A, u, v, 1)
        return fake_edges

    elif scenario == 3:
        # pick k pairs of nodes uniformaly at random, and connect them all
        # repeat it for five times
        node_pairs = 5
        total_fake_edges = []
        for _ in range(node_pairs):
            u_s =  rs.choice(tuple(A.keys()),2)
            unique_fake_edges = np.array(list(itertools.permutations(u_s.tolist())))
            number_of_rows = unique_fake_edges.shape[0]
            rand_edge_idx = rs.choice(number_of_rows, size = edgeNum//node_pairs)
            fake_edges = unique_fake_edges[rand_edge_idx, :]
            for _edge in fake_edges:
                u = _edge[0]
                v = _edge[1]
                inject_v2(A, u, v, 1)
            total_fake_edges.append(fake_edges)
        return np.vstack(total_fake_edges)
    else:
        return None
        

