import sys
import os
import functools
from os.path import join as os_join 
from os.path import dirname as up_dir
project_root = functools.reduce(lambda x,y: up_dir(x), range(4) ,os.path.abspath(__file__) )
sys.path.append(os_join(project_root,'src'))

import numpy as np
import networkx as nx
import numba
import scipy.sparse as sp
from sklearn.utils import murmurhash3_32 as murmurhash
from tqdm import tqdm as tqdm

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

MAX_EPSILON_PRECISION = np.float32(1e-6)
MIN_EPSILON_PRECISION = np.float32(1e-10)

print (f'MAX_EPSILON_PRECISION:{MAX_EPSILON_PRECISION}')
print (f'MIN_EPSILON_PRECISION:{MIN_EPSILON_PRECISION}')

@numba.njit(cache =True, nogil = True)
def get_list_of_dict(n):
    list_dict = numba.typed.List()
    for _ in range(n):
        list_dict.append(
                numba.typed.Dict.empty(
                    key_type=numba.types.uint32,
                    value_type=numba.types.float32,
                    )
            )
    return list_dict

def get_hash_LUT(n,dim = 512):
    node_id_2_dim_id = np.zeros(n,dtype=np.int32)
    node_id_2_sign = np.zeros(n,dtype=np.int8)
    for _ in tqdm (range(n)):
        dim_id = murmurhash(_, seed = 0,positive=True)%dim
        sign = murmurhash(_, seed = 0,positive=True)%2
        node_id_2_dim_id[_] = dim_id
        node_id_2_sign[_] = 1 if sign == 1 else -1
    return node_id_2_dim_id, node_id_2_sign

@numba.njit(cache =True, parallel = True, fastmath=True, nogil = True)
def get_hash_embed(node_id_2_dim_id, node_id_2_sign, dim, n, indices, indptr, data):
    emb_mat = np.zeros((n,dim), dtype=np.float32)
    for i in numba.prange(n): # for all nodes.
        cols = indices[indptr[i]:indptr[i + 1]]
        vals = data[indptr[i]:indptr[i + 1]]
        emb_vec = emb_mat[i,:]
        for j, val in zip(cols,vals):
            emb_vec[node_id_2_dim_id[j]] += node_id_2_sign[j]* np.max( np.array([np.float32(0.0), np.float32(np.log(val*n))]) )
    return emb_mat

@numba.njit(cache =True)
def get_numba_dict_faster(numb_dict):
    # return all (k,v) pairs from numba.typed.Dict().
    # If it is under njit mode, key,val access is directed with minimal overhead.
    # If it is under python mode,  key (in python obj format) will be converted to numba key repr, causing sigificant conversion overhead
    # Reference: https://github.com/numba/numba/issues/6439#issuecomment-735959233 
    keys = []
    vals = []
    for key in numb_dict:
        val = numb_dict[key]
        keys.append(key)
        vals.append(val)
    return np.array(keys).astype(np.uint32), np.array(vals).astype(np.float32)

#@numba.njit(cache =True)
@numba.jit(cache =True)
def get_numba_nested_dict_faster(s_id_list, numb_list_dict):
    new_data = []
    new_row = []
    new_col = []
    for _key in s_id_list:
        p_numb_dict = numb_list_dict[_key] 
        #get_numba_dict_faster: iterate numba.typed.Dict in njit mode, reducing (k,v) access overhead.
        _new_col, _new_data = get_numba_dict_faster(p_numb_dict)
        _new_row = np.array([ _key for _ in _new_col]).astype(np.uint32)
        new_row.append(_new_row)
        new_col.append(_new_col)
        new_data.append(_new_data)
    return new_row, new_col, new_data


# @numba.jit(nopython =False)
def get_numba_nested_dict_faster_v2(s_id_list, updated_s_id_list, prev_sp_appr, numb_list_dict):
    new_data = []
    new_row = []
    new_col = []
    # get the previous ppv without accessing p_numb_dict
    if prev_sp_appr is not None:
        previous_sp_ppv = prev_sp_appr['sp_ppv']
        noupdated_s_id = np.array(list(set(s_id_list)-set(updated_s_id_list)))
        previous_sp_ppv_coo = previous_sp_ppv[noupdated_s_id,:].tocoo()
        non_update_row = previous_sp_ppv_coo.row
        non_update_col = previous_sp_ppv_coo.col
        non_update_data = previous_sp_ppv_coo.data
        non_update_row = noupdated_s_id[non_update_row]
        new_data+=non_update_data.tolist()
        new_row+=non_update_row.tolist()
        new_col+=non_update_col.tolist()
    for _key in updated_s_id_list:
        p_numb_dict = numb_list_dict[_key] 
        #get_numba_dict_faster: iterate numba.typed.Dict in njit mode, reducing (k,v) access overhead.
        _new_col, _new_data = get_numba_dict_faster(p_numb_dict)
        _new_row = np.array([ _key for _ in _new_col]).astype(np.uint32)
        new_row.append(_new_row)
        new_col.append(_new_col)
        new_data.append(_new_data)
    
    return new_row, new_col, new_data


@numba.njit(cache =True, parallel = True, fastmath=True, nogil = True)
def push_numba_parallel(N, query_list, indices, indptr, data, node_degree, p, r, alpha, beta, init_epsilon):
    """Based on Andersen's local push algorithm"""
    """With Numba acceleration"""
    eps_prime = np.float32(init_epsilon / node_degree.sum())
    for i in numba.prange(len(query_list)):
        ###########################################
        ## A thread under numba's hood . No GIL
        ###########################################
        
        ###################################
        # Adaptive push according to p_s
        ###################################
        s = query_list[i]
        adapt_epsilon = np.float32(eps_prime*node_degree[s]) # for high degree nodes?
        adapt_epsilon = np.max(np.array([adapt_epsilon, MIN_EPSILON_PRECISION])) # the lower bound 
        epsilon = np.min(np.array([adapt_epsilon, MAX_EPSILON_PRECISION])) # the upper bound
        #MAX_EPSILON_PRECISION = 1e-6
        #MIN_EPSILON_PRECISION = 1e-10
        # epsilon =  1e-6 # for debug only

        p_s = p[s]
        r_s = r[s]
        
        #################################### 
        ###### Positive FIFO Queue #########
        ## r_s[v] > epsilon*node_degree[v] #
        ####################################
        q_pos = numba.typed.List()
        q_pos_ptr = 0
        # q_pos_keys: The in-queue element.
        q_pos_marker = numba.typed.Dict.empty(
                    key_type=numba.types.uint32,
                    value_type=numba.types.boolean,
                    )
        # scan all residual in r_s, select init for pushing
        # Or only maintain the top p_s[v] nodes for pushing? since it's likely affect top-ppr
        for v in r_s:
            if r_s[v] > epsilon*node_degree[v]:
                q_pos.append(v)
                q_pos_marker[v] = True
    
        # Positive: pushing pushing!
        num_pushes_pos = 0
        while len(q_pos)!=q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr+=1
            q_pos_marker.pop(u)
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            
            if r_s_u > epsilon*deg_u: # for positive
                num_pushes_pos+=1
                if u not in p_s:
                    p_s[u] = np.float32(0.0)
                p_s[u] += alpha*r_s_u
                #push_residual = (1-alpha)*r_s_u*(1-beta)/deg_u
                push_residual = np.float32((1-alpha)*r_s_u/deg_u)
                _v = indices[indptr[u]:indptr[u + 1]]
                _w = data[indptr[u]:indptr[u + 1]]
                neighbor_push = np.column_stack((_v, _w)).astype(np.uint32)
                for _ in neighbor_push:
                    v = _[0]
                    w_u_v = _[1]
                    v = np.uint32(v)
                    # neighbor of the node u
                    if v not in r_s:
                        r_s[v] = np.float32(0.0)
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    #r_s[v] += np.float32(push_residual)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                #r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)
        #####################################
        ###### Negative FIFO Queue ##########
        ## r_s[v] < -epsilon*node_degree[v] #
        #####################################
        q_pos = numba.typed.List()
        q_pos_ptr = 0
        # q_pos_keys: The in-queue element. use dict as set.
        q_pos_marker = numba.typed.Dict.empty(
                    key_type=numba.types.uint32,
                    value_type=numba.types.boolean,
                    )
        # scan all residual in r_s, select init for pushing
        for v in r_s:
            if r_s[v] < -epsilon*node_degree[v]: ## for negative
                q_pos.append(v)
                q_pos_marker[v] = True
        num_pushes_neg = 0
        # Negative: pushing pushing!
        while len(q_pos)!=q_pos_ptr:
            u = q_pos[q_pos_ptr]
            q_pos_ptr+=1
            q_pos_marker.pop(u) # remove from the dummy set
            deg_u = node_degree[u]
            r_s_u = r_s[u]
            if r_s_u < -epsilon*deg_u: # for negative
                num_pushes_neg+=1
                if u not in p_s:
                    p_s[u] = np.float32(0.0)
                p_s[u] += alpha*r_s_u
                #push_residual = (1-alpha)*r_s_u*(1-beta)/deg_u-> beta =0
                push_residual = np.float32((1-alpha)*r_s_u/deg_u)
                _v = indices[indptr[u]:indptr[u + 1]]
                _w = data[indptr[u]:indptr[u + 1]]
                neighbor_push = np.column_stack((_v, _w)).astype(np.uint32)
                for _ in neighbor_push:
                    v = _[0]
                    w_u_v = _[1]
                    v = np.uint32(v)
                    # neighbor of the node u
                    if v not in r_s:
                        r_s[v] = np.float32(0.0)
                    # should multply edge weights.
                    r_s[v] += np.float32(push_residual * w_u_v)
                    if v not in q_pos_marker:
                        q_pos.append(v)
                        q_pos_marker[v] = True
                #r_s[u] = (1-alpha)*r_s[u]*beta # beta=0 --> r_s[u] = 0
                r_s[u] = np.float32(0.0)


@numba.njit(cache =True, fastmath=True, nogil = True)
def dynamic_adjust_numba_wrap(s_id_list, dict_p_arr, dict_r_arr, D, feed_new_uv_id, alpha, delta_s_r ):
    for edge in feed_new_uv_id:
        u_, v_, w_ = edge[0], edge[1], edge[3]
        D[u_] += w_
        dynamic_adjust(s_id_list, dict_p_arr, dict_r_arr, D, u_, v_, w_, alpha, delta_s_r)
        

@numba.njit(cache =True, parallel = True, fastmath=True, nogil = True)
def dynamic_adjust(s_id_set_pre, dict_p_arr, dict_r_arr, D, u_, v_, w_, alpha, delta_s_r):
    # Extended from Zhang 2016 dynamic push adjustment, but with weighted edges
    # delta_r for each 
    for i in numba.prange(len(s_id_set_pre)): # for every source node.
        s_id = np.uint32(s_id_set_pre[i])
        #s_id = s_id_set_pre[i]
        p = dict_p_arr[s_id] # e.g.: dict_p_arr[s_id][v] = 0.5
        r = dict_r_arr[s_id] 
        # Init residual to be 1.0 for further pushing: r_s[s] = 1.0 or self.dict_r_arr[s][s] = 1.0
        if s_id not in r: # if this source node was never seen before.
            r[s_id] = np.float32(1.0) # r_s = 1
            p[s_id] = np.float32(0.0) # p_s = 0
            delta_s_r[s_id] += np.float32(1.0) # the changed volumn.
        
        # init value for update
        u = u_
        v = v_
        w = w_ # this is the delta w_{(u,v)}
        if u not in p :
            p[u] = np.float32(0.0)
            r[u] = np.float32(0.0)
        
        if v not in p :
            p[v] = np.float32(0.0)
            r[v] = np.float32(0.0)
        
        if p[u]!=np.float32(0.0) and (D[u]-w)!=0:
            p[u] = np.float32(p[u]* D[u]/(D[u]-w))
            delta_r_u = w*p[u]/(D[u]*alpha)
            delta_r_v = (1-alpha)*delta_r_u
            #delta_r_v = (1-alpha)*w*p[u]/(D[u]*alpha)
            r[u] = np.float32(r[u] - delta_r_u) #(p[u]/(D[u] = prev_p[u]/D[u]-1)
            r[v] = np.float32(r[v] + delta_r_v)
            # record how much adjust happens for source node s
            delta_s_r[s_id] += np.float32(delta_r_u) # the changed volumn.
    # return delta_s_r

class GraphPPREstimator():
    """
    Incremental Personalized PageRank estimator based on the following works:
    [1].    Andersen, Reid, Fan Chung, and Kevin Lang. "Local graph partitioning using pagerank vectors." 
            IEEE Symposium on Foundations of Computer Science (FOCS) 2006.
    [2].    Zhang, Hongyang, Peter Lofgren, and Ashish Goel. "Approximate personalized pagerank on dynamic graphs." 
            ACM Knowledge Discovery and Data Mining (KDD)2016
    [3].    Guo X., Zhou B., Skiena S., Subset Node Representation Learning over Large Dynamic Graphs.
            ACM Knowledge Discovery and Data Mining (KDD)2021
    [4].    Guo X., Zhou B., Skiena S., Subset Node Anomaly Tracking over Large Dynamic Graphs.
            ACM Knowledge Discovery and Data Mining (KDD)2022 
    """

    def __init__(self, n, alpha = 0.1, epsilon=0.15, beta = 0, gamma = 0.0, selective_push_threshold = 0.01, dim = 512, update_networkx = False, is_multi_graph = False, is_directed_graph = False):
        # n is the max total number of nodes.
        self.n = n
        self.update_networkx = update_networkx
        """dictionary mapping raw node to internal id (Integer, starting from 0)"""
        self.dict_node2id = {} 
        """dictionary mapping internal id back to raw node"""
        self.dict_id2node = {} 
        
        """ the csr matrix of the graph """
        self.g = sp.csr_matrix(([], ([], [])), shape=(n, n))
        self.g_indices = self.g.indices
        self.g_indptr = self.g.indptr
        self.g_data = self.g.data
        
        self.node_degree = np.zeros(n, dtype=np.int64)
        self.delta_node_degree = np.zeros(n, dtype=np.int64)
        
        self.selective_push_threshold = selective_push_threshold
        self.tracked_node_id = set([])
        self.anomalous_node_id = set([]) # add the node ids if it involves in anmalous events.
        
        """nested dictionary for estimated PPR $\pi(s,t)$, both dictionaries are indexed by node_id dict[source_id][target_id], if not, then 0"""
        self.dict_p_arr = get_list_of_dict(n)
        #self.dict_p_arr = numba.typed.List()
        """nested dictionary for residual $r(s,t)$, both dictionaries are indexed by node_id dict[source_id][target_id], if not, then 0"""
        self.dict_r_arr = get_list_of_dict(n)
        
        self.delta_s_r = np.zeros(n, dtype=np.float32)
        
        """the hash embedding dimension/sign map"""
        node_id_2_dim_id, node_id_2_sign = get_hash_LUT(n, dim)
        self.hash_dim_arr = node_id_2_dim_id
        self.hash_sgn_arr = node_id_2_sign
        
        
        """the damping factor, \pi = (1-alpha) \pi P + \alpha e_s"""
        self.alpha = alpha
        """ensure r(u) < epsilon*D[u], min push val: epsilon"""
        self.epsilon = epsilon
        """lazy random walk, set beta to be 0 (no lazy random walk)"""
        self.beta = beta
        """ temporal-push bias factor.
            t_weight = np.power(self.gamma,t-min_year_offset)/Z
            gamma \in (0,1): push more residual along old edges, 
            gamma =  1: no temporal bias, 
            gamma \in (1,inf): push more residual along new edges, 
        """
        self.gamma = gamma # not used so far, should be zero
        assert self.gamma ==0, 'gamma should be zero'
        
        """ BOOL indictor for weighted multi-graph"""
        self.is_multi_graph = is_multi_graph
        self.is_directed_graph = is_directed_graph

        """the nx.Graph for exact ppr calculation as error measurement"""
        if self.update_networkx:
            self.G =  nx.MultiDiGraph()
        else:
            self.G = None
        self.edge_set = set()

        """BOOL indictor for temporal-push (not uniformally push residual, but biased by temporal edge weight)"""
        self.is_temporal_push = None
        
        print ("GraphPPREstimator init finished")
        

    def update_graph(self, e):
        """[Maintain graph edge by edge, the graph is directed mulit-graph]
        Pre-process should remove self-loop . 
        Args:
            E ([List]): [the list of pair of raw nodes with timestamp: (u, v , timestamp)]
        """
        u_id = e[0]
        v_id = e[1]
        t = e[2] # could convert to epoch time as int or double
        w = e[3]
        
        # update degree
        self.node_degree[u_id] += w
        
        # update graph in nx for exact ppr validation
        if self.update_networkx:
            # depends on how to define G.
            self.G.add_edge(u_id, v_id, t = t, w = w)
        
        # Edge set is maintained in previous 2-nd edge sweep.
        # self.edge_set.add((u_id, v_id))
        
        new_row = [u_id]
        new_col = [v_id]
        new_data = [w] # or t.
        #assert t>0 , 'with t=0 , the sparse matrix does not add it '
        return new_row, new_col, new_data

    def get_approx_ppv_with_new_edges_batch_multip_source(self, query_node_desc_list, E_t, bool_return_dict = True, track_mode ='all', top_index = 100):
        """ Update with newly inserted edge list E_t, 
            maintaining the p, r vector of each source node dynamically.
        """
        # for each edge in this batch, update graph and p,r for every source node.
        D = self.node_degree
        #old_D = np.copy(self.node_degree)
        # future_D is for temporarily recording how degree will actually change for this snapshot.
        # It is for selecting the top changed nodes in degree for tracking.
        future_D = np.copy(self.node_degree)
        feed_new_uv_id = []
        _edge_set = set([]) # remove duplicate in this edge batch
        
        # reset anomalous nodes in this snapshot
        self.anomalous_node_id = set([])
        
        # do first sweep, get the internal id, get the delta degree change of every nodes.
        for edge in E_t:
            _u = edge[0]
            _v = edge[1]
            _t = edge[2]
            _w = np.float32(edge[3]) #if len(edge) == 5 else np.float32(1.0)
            _l = np.float32(edge[4]) #if len(edge) == 5 else np.float32(0.0)
            
            # remove self loop
            if _u == _v:
                continue 
            
            # Suppose input graph is directed, multi-edge
            # If it is not multi-graph, then, filter duplicated edges.
            if self.is_multi_graph is False:
                if self.is_directed_graph:
                    if (_u, _v) in _edge_set: 
                        continue
                else: # for undirected graph.
                    if (_u, _v) in _edge_set or (_v, _u)  in _edge_set:
                        continue
                
            # check if u, v in the current graph..
            if _u not in self.dict_node2id.keys():
                u_id = np.uint32(len(self.dict_node2id.keys())) # as internal id
                self.dict_node2id[_u] = u_id
                self.dict_id2node[u_id] = _u  
            if _v not in self.dict_node2id.keys():
                v_id = np.uint32(len(self.dict_node2id.keys())) # as internal id
                self.dict_node2id[_v] = v_id
                self.dict_id2node[v_id] = _v
            # convert to internal id 
            u_id = self.dict_node2id[_u]
            v_id = self.dict_node2id[_v]
            
            # record the node pair if this edge (u,v ) involved in an anomalous events w.r.t. _l (label)
            if _l != np.float32(0.0):
                self.anomalous_node_id.add(u_id)
                self.anomalous_node_id.add(v_id)

            # suppose input graph is directed, multi-edge
            # If it is not multi-graph, then, filter duplicated edges.
            if self.is_multi_graph is False:
                _e_weight = np.float32(1.0)
                if self.is_directed_graph:
                    if (u_id, v_id) in self.edge_set :
                        continue
                    else:
                        _edge_set.add((_u, _v))
                        feed_new_uv_id.append((u_id, v_id, _t, _e_weight)) # not multi-graph, edge weight is fixed to be 1.
                        self.edge_set.add((u_id, v_id))
                        future_D[u_id]+=1 # not multi-graph, edge weight is fixed to be 1.
                else: # for undirected graph.
                    if (u_id, v_id) in self.edge_set  or  (v_id, u_id) in self.edge_set :
                        continue                    
                    _edge_set.add((_u, _v))
                    _edge_set.add((_v, _u))
                    self.edge_set.add((u_id, v_id))
                    self.edge_set.add((v_id, u_id))
                    feed_new_uv_id.append((u_id, v_id, _t, _e_weight)) # not multi-graph, edge weight is fixed to be 1.
                    feed_new_uv_id.append((v_id, u_id, _t, _e_weight))
                    future_D[u_id]+=1 # not multi-graph, edge weight is fixed to be 1.
                    future_D[v_id]+=1
                    pass
            else: # for multi-graph, do not de-duplicate.
                if self.is_directed_graph:
                    feed_new_uv_id.append((u_id, v_id, _t, _w))
                    future_D[u_id]+=_w
                else:
                    feed_new_uv_id.append((u_id, v_id, _t, _w))
                    feed_new_uv_id.append((v_id, u_id, _t, _w))
                    future_D[u_id]+=_w
                    future_D[v_id]+=_w
        
        # degree changes of each nodes after this snapshot.# O(n) memory
        self.delta_node_degree = future_D - D 
        # after first init sweep, find the right trakcing node set for push.
        if track_mode == 'definitive':
            # use input query_node_desc_list
            ## Cautions: query_node_desc_list may contain the node does not appear in the graph
            for node_desc in query_node_desc_list:
                if node_desc in self.dict_node2id.keys() :
                    self.tracked_node_id.add(self.dict_node2id[node_desc])
            #dummy_nodes = set(np.argpartition(future_D, -top_index)[-top_index:])
        elif track_mode == 'active':
            self.tracked_node_id |= set(np.argpartition(future_D, -top_index)[-top_index:])
            for _s_id in list(self.tracked_node_id):
                if _s_id not in self.dict_id2node.keys() :
                        self.tracked_node_id.remove(_s_id)
        elif track_mode == 'active-half':# the top half degree nodes
            half_index = self.n//2
            self.tracked_node_id |= set(np.argpartition(future_D, -half_index)[-half_index:]) # could optimize
            for _s_id in list(self.tracked_node_id):
                if _s_id not in self.dict_id2node.keys() :
                        self.tracked_node_id.remove(_s_id)
        elif track_mode == 'all': 
            self.tracked_node_id = set(self.dict_id2node.keys())
        elif track_mode == 'top-change':
            #self.tracked_node_id = set(self.dict_id2node.keys())
            self.tracked_node_id |= set(np.argpartition(self.delta_node_degree, -top_index)[-top_index:]) # could optimize
            for _s_id in list(self.tracked_node_id): # for several zeros (not yet exist, remove them)
                if _s_id not in self.dict_id2node.keys() :
                        self.tracked_node_id.remove(_s_id)
        else:
            print ("please assign node tracking strategy")
            exit()

        # if  len(self.anomalous_node_id.intersection(self.tracked_node_id)) > 0:
        #     print (self.anomalous_node_id.intersection(self.tracked_node_id))

        # convert target node ids to numba list for multi-thread adjusting and pushing
        s_id_list = numba.typed.List([np.uint32(0)])
        s_id_list.pop() # return the empty list with data type
        for valid_query_node_id in self.tracked_node_id:
            s_id_list.append(np.uint32(valid_query_node_id))
        # num_s_id_list = len(s_id_list)
        # s_id_list may not define.

        # update graph for all valid new edges
        new_row=[]
        new_col=[]
        new_data=[]
        # do second sweep, upate the edge into the graph, 
        for edge in tqdm(feed_new_uv_id): # this is one edge (undirected edge was augmented.)
            _new_row, _new_col, _new_data = self.update_graph(edge) # update this one edge
            if _new_row is not None:
                new_row += _new_row
                new_col += _new_col
                new_data += _new_data # this is edge weight
                u_ = _new_row[0]
                v_ = _new_col[0]
                w_ = _new_data[0]
                # adjust the p, r of tracking nodes wrt this edge change.
                # 1.4 update p,r vector of this inserted edge of ** every source node**
                #self.delta_s_r = dynamic_adjust(s_id_list, self.dict_p_arr, self.dict_r_arr, D, u_,v_,w_, self.alpha, self.delta_s_r)
                # if num_s_id_list > 0:
                dynamic_adjust(s_id_list, self.dict_p_arr, self.dict_r_arr, D, u_, v_, w_, self.alpha, self.delta_s_r)
        # print ('r after adjustment', self.dict_r_arr)
        # print ('p after adjustment', self.dict_p_arr)
        # for fast dynamic adjust
        # feed_new_uv_id
        #dynamic_adjust_numba_wrap(s_id_list, self.dict_p_arr, self.dict_r_arr, old_D, np.array(feed_new_uv_id, dtype=np.int64), self.alpha, self.delta_s_r)
        
        # update csr of graph after this edge batch
        # this is the bottleneck 
        self.g += sp.csr_matrix((new_data, (new_row, new_col)), shape=(self.n, self.n))
        self.g_indices = self.g.indices
        self.g_indptr = self.g.indptr
        self.g_data = self.g.data

        # 0. no selective
        # _s_id_list = s_id_list
        
        # 1. based on degree delta (does not work)
        # _s_id_list = numba.typed.List()
        # for _sid in self.tracked_node_id:
        #     if self.delta_node_degree[_sid] > 3:
        #         _s_id_list.append(_sid)
        
        # 2. based on accumulative r_s_u
        _s_id_list = numba.typed.List()
        for _sid in self.tracked_node_id:
            if self.delta_s_r[_sid] > np.float32(self.selective_push_threshold):
                _s_id_list.append(_sid)
                # reset to 0.
                self.delta_s_r[_sid] = np.float32(0.0)
        
        # updated list
        if len(_s_id_list) > 0:
            push_numba_parallel(self.n, _s_id_list, self.g_indices, self.g_indptr, self.g_data, self.node_degree, \
                                self.dict_p_arr, self.dict_r_arr, self.alpha, self.beta, self.epsilon)
        
        if bool_return_dict: # this is for debug only, access numba.typed.Dict in python mode is very slow.
            print ('========= Significant I/O Overhead   Warning =========================')
            print ('========= Returning dictionaries of all PPVs calls func. ===============')
            print ('========= __getitem__ over all element with great overhead =============')
            print ('== https://github.com/numba/numba/issues/6439#issuecomment-735959233 ===')
            return_dict = {}
            for s_id in s_id_list:
                return_dict[self.dict_id2node[s_id]] = {
                            "num_pos_push":None,
                            "num_neg_push":None,
                            "ppv":self.dict_p_arr[s_id], # this cause copy all element in this big self.dict_p_arr with memory big-O (n^2)!
                }
            return return_dict
        else:
            return s_id_list, _s_id_list


    def get_exact_ppv_dict(self, query_node_desc):
        # use networkx to calculate exact ppr matrix
        assert self.G is not None
        if query_node_desc not in  self.dict_node2id.keys():
            return {}
        query_node_id = self.dict_node2id[query_node_desc]
        #NOTE: The definition of alpha in nx.pagerank is opposite to ours: (nx.pagerank:= pi = alpha pi P + (1-alpha) e_s)
        ppr_dict = nx.pagerank(self.G, personalization={query_node_id:1}, alpha=1-self.alpha, weight = 'w' )
        #ppr_dict = nx.pagerank_scipy(self.G, personalization={query_node_id:1}, alpha=self.alpha )
        #ppr_dict = nx.pagerank_numpy(self.G, personalization={query_node_id:1}, alpha=self.alpha )
        return ppr_dict

def test_dynamic_ppr_batch_source(E, list_E_new):
    alpha = np.float32(0.15)
    beta = np.float32(0.0)
    epsilon = np.float32(1e-6)
    n = 5
    track_mode = 'all'
    top_index = n
    update_networkx = True
    is_multi_graph = True
    is_directed_graph = False

    estimator = GraphPPREstimator(n=n, alpha= alpha, epsilon = epsilon, beta = beta , 
                                    update_networkx = update_networkx, 
                                    is_multi_graph = is_multi_graph, 
                                    is_directed_graph = is_directed_graph)
    #query_node_desc_set = np.arange(0,10,1)
    query_node_desc_set = [0]
    dict_rmse = {}
    snapshot_ppv = []
    
    # add new edges list_E_new is a list of list
    for E_new in [E]+ list_E_new:
        print ("-------------------------------")
        print ("----------new-batch------------")
        print ("-------------------------------")
        print ('added edge:',E_new)
        return_dict = estimator.get_approx_ppv_with_new_edges_batch_multip_source(query_node_desc_set, E_new, bool_return_dict = True, track_mode =track_mode, top_index = top_index)
        #print ('re2', return_dict)
        appr = {}
        for key in return_dict.keys():
            _ppr = np.zeros(n)
            for ind,val in return_dict[key]['ppv'].items():
                _ppr[ind] = val
            appr[key] = _ppr
        #print ('appr', appr)
                
        eppr = {}
        return_dict = {}
        for query_node_desc in query_node_desc_set:
            return_dict = estimator.get_exact_ppv_dict(query_node_desc)
            for key in return_dict.keys():
                _ppv = np.zeros(n)
                for ind,ppr in return_dict.items():
                    _ppv[ind] = ppr
                eppr[query_node_desc] = _ppv
        
        for query_node_desc in query_node_desc_set:
            if query_node_desc not in appr.keys():
                continue
            print ('the query node', query_node_desc)
            # print (f'====== {query_node_desc} #snapshot:{ss} #new_edges:{len(new_edges)}  num_cc: {num_cc(estimator.G)}======')
            print ('appr. ppr ', np.round(appr[query_node_desc],3)[:])
            print ('exact ppr', np.round(eppr[query_node_desc],3)[:])
        print (" the graph:")
        print (" the networkx")
        print (nx.to_numpy_matrix(estimator.G, weight = 'w'))
        print (" the csr")
        print (estimator.g.toarray())
    return snapshot_ppv

if __name__ == "__main__":
    E = [
        (0,1,1,1,1),
        (0,2,1,1,1),
        (0,3,1,1,1),
        (0,4,1,1,1),
        ]
    # new snapshot...
    E_new = [
        [(0,2,1,1,1),(2,3,1,1,1)],
    ]
    snapshot_ppv = test_dynamic_ppr_batch_source(E, E_new)
    print (snapshot_ppv)
    print(f'Threading layer chosen: {numba.threading_layer()}' )
