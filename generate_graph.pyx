# encoding: utf-8
# cython: profile=True
# filename: generate_graph.pyx

from multiprocessing import Process, Queue
import cPickle
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict, Counter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from itertools import chain, combinations

cimport numpy as np
ctypedef np.int_t DTYPE_INT_t
ctypedef np.float_t DTYPE_t

# Create an ego network with num_friends friends.
# For each pair of friends
# sharing the same label, connect them with prob_interfriend_edge.
# Each friend shares a given label type with prob_friend_shares_label_type.
def create_ego_net(int num_friends=10,
                   float prob_interfriend_edge=0.1,
                   np.ndarray[DTYPE_t, ndim=2] p_L=np.kron([0.2, 0.8], [0.2, 0.3, 0.5]).reshape((2,3)),
                   np.ndarray[DTYPE_t] q_L=np.array([0.2, 0.8])):

  # Set ego's attributes
  cdef np.ndarray[DTYPE_INT_t] shape = np.array([p_L.shape[0], p_L.shape[1]])
  cdef long prod_shape = np.prod(shape)
  cdef np.ndarray p_L_1d = p_L.reshape(prod_shape)
  cdef list ego_attr = list(np.unravel_index(
      np.random.choice(prod_shape, p=p_L_1d),
      shape))
#  print "Ego index = " + str(ego_attr)

  cdef np.ndarray friends_share = np.random.choice(len(q_L), size=num_friends, p=q_L)
  cdef list friend_attr = []
  cdef dict by_attr_value = {}  # dict of friends by attribute value
  cdef set matching_pairs = set()  # Pairs of friends with some matching attr

  cdef int i, shared_index, shared_value, new_prod_shape, label_type, val
  cdef int x, y
  cdef list mask, this_friend_attr, all_attr, ego_edges
  cdef tuple z
  cdef dict ego_graph
  cdef np.ndarray[DTYPE_t, ndim=2] conditional_p
  cdef np.ndarray[DTYPE_t] conditional_p_1d
  cdef np.ndarray[DTYPE_INT_t] cond_shape

  for i in np.arange(num_friends):
    shared_index = friends_share[i]
    shared_value = ego_attr[shared_index]
    mask = [False] * shape[shared_index]
    mask[shared_value] = True
    new_prod_shape = prod_shape / shape[shared_index]
    conditional_p = np.compress(mask, p_L, axis=shared_index)
    conditional_p /= np.sum(conditional_p)  # Normalize
    cond_shape = np.array([conditional_p.shape[0], conditional_p.shape[1]])
    conditional_p_1d = conditional_p.flatten()
    this_friend_attr = list(np.unravel_index(
        np.random.choice(new_prod_shape, p=conditional_p_1d),
        cond_shape))
    this_friend_attr[shared_index] = shared_value
#    print "Friend %d: %s shared index %d" % (i+1, str(this_friend_attr), shared_index)
    
    friend_attr.append(this_friend_attr)
    for label_type, val in enumerate(this_friend_attr):
      key = "%d-%d" % (label_type, val)
      if key in by_attr_value:
        for f in by_attr_value[key]:
          matching_pairs.add((label_type, f, i+1))
        by_attr_value[key].append(i+1)
      else:
        by_attr_value[key] = [i+1]
  
  all_attr = [ego_attr] + friend_attr

  # Create ego graph
  ego_edges = [(z[1], z[2]) for z in matching_pairs if np.random.random() < prob_interfriend_edge * q_L[z[0]]]
  ego_edges.extend([(0, i) for i in xrange(1, num_friends + 1)])
  ego_graph = {}
  for (x, y) in ego_edges:
    if x in ego_graph:
      ego_graph[x].append(y)
    else:
      ego_graph[x] = [y]
    if y in ego_graph:
      ego_graph[y].append(x)
    else:
      ego_graph[y] = [x]

#  print ego_graph

  return (all_attr, ego_graph)

# Hide the attributes of ego and prob_hide_friend fraction of friends
def train_test(all_attr, prob_hide_friend=0.3):
  num_attr = len(all_attr[0])
  hidden_attr = [np.nan] * num_attr
  train_attr = [x if np.random.random() > prob_hide_friend else hidden_attr for x in all_attr]
  train_attr[0] = hidden_attr  # Ego attributes are missing
  return train_attr

###
### UPDATE FUNCTIONS
###

# Given a list of the (list of node prob dicts) for each neighbor,
# create a a list of node prob dicts for yourself.
# This function does the LP update
cpdef list update_node_probs_LP(
    list my_node_prob,
    list friend_node_probs,
    int iteration=0,
    dict options={}):

  cdef list probs_per_type = zip(*friend_node_probs)
  cdef list result = []
  cdef tuple list_of_dicts
  cdef int num_nonempty, label
  cdef float prob
  cdef dict d, one_dict
  for list_of_dicts in probs_per_type:
    d = {}
    num_nonempty = 0
    for one_dict in list_of_dicts:
      if bool(one_dict):
        num_nonempty += 1 
      for label, prob in one_dict.iteritems():
        if label in d:
          d[label] += prob
        else:
          d[label] = prob
    result.append({k: v / num_nonempty for k, v in d.iteritems()})
  return result

# This function does the dot_prod update
cpdef list update_node_probs_DOT(
    list my_node_probs,
    list friend_node_probs,
    int iteration=0,
    dict options={'alpha': 1.0, 'c': 0.0, 'stepsize': 100.0}):

  cdef list gradient = [{} for i in xrange(len(my_node_probs))]
  cdef list result = []
  cdef list f
  cdef float dotprod, sigmoid, factor, v, sum_label_type
  cdef int label_type, l, k
  cdef dict f_label_probs, m_label_probs, node_probs, d
  cdef set overlapping_labels

#  factors = []
  for f in friend_node_probs:
    dotprod = 0.0
    for label_type, f_label_probs in enumerate(f):
      m_label_probs = my_node_probs[label_type]
      overlapping_labels = m_label_probs.viewkeys() & f_label_probs.viewkeys()
      dotprod += sum([m_label_probs[l] * f_label_probs[l] for l in overlapping_labels])
    sigmoid = 1.0 / (1.0 + math.exp(-options['alpha'] * dotprod - options['c']))
    factor = (1.0 - sigmoid) * (options['stepsize'] / len(friend_node_probs))
#    factors.append(factor)

    for label_type, f_label_probs in enumerate(f):
      for k, v in f_label_probs.iteritems():
        if k in gradient[label_type]:
          gradient[label_type][k] += factor * v
        else:
          gradient[label_type][k] = factor * v

#  print "Counts of factor = {}".format(Counter(factors))

  # Add gradient to my_node_probs
  for label_type, node_probs in enumerate(my_node_probs):
    d = {}
    for k, v in node_probs.iteritems():
      d[k] = v
    for k, v in gradient[label_type].iteritems():
      if k in d:
        d[k] += v
      else:
        d[k] = v
    
    # Project to simplex
    sum_label_type = sum(d.itervalues())
    for k in d.iterkeys():
      d[k] /= sum_label_type
    result.append(d)

  return result


# This function does the variational update
def update_node_probs_VAR(
    list my_node_probs,
    list friend_node_probs,
    int iteration=0,
    dict options={'alpha': 1.0, 'c': 0.0}):

  cdef list result = []

  cdef int i, label_type, l, k
  cdef float phi_label_type
  cdef dict m_label_probs, f_label_probs
  cdef dict A, B

  cdef list f, dotproducts
  cdef np.ndarray[DTYPE_t] phi

  cdef int nnz_w
  cdef tuple s
  cdef list w, phi_w
  cdef float factor, g
  
  cdef dict d, probs
  cdef float v, maxval, denom, expval

  cdef int num_label_types = len(my_node_probs)
  cdef list new_node_logprobs = [{} for i in xrange(num_label_types)]
  
  for f in friend_node_probs:

    # Compute dotproducts
    dotproducts = []
    for label_type, A in enumerate(f):
      B = my_node_probs[label_type]
      if len(A) < len(B):
        dotproducts.append(sum([A[k]*B[k] for k in A if k in B]))
      else:
        dotproducts.append(sum([A[k]*B[k] for k in B if k in A]))

    # Compute phi
    phi = np.zeros(num_label_types)
    for nnz_w in xrange(num_label_types + 1):
      factor = math.log(1 + math.exp(-options['alpha'] * nnz_w - options['c']))
      for s in combinations(range(num_label_types), nnz_w):
        w = [False] * num_label_types
        for i in s:
          w[i] = True
        phi_w = [factor] * num_label_types
        for i in xrange(num_label_types):
          if w[i]:
            g = dotproducts[i]
          else:
            g = 1.0 - dotproducts[i]
          for j in xrange(num_label_types):
            if j != i:
              phi_w[j] *= g
        for j in s:
          phi_w[j] *= -1.0
        for i in xrange(num_label_types):
          phi[i] += phi_w[i]

    # Compute \sum \mu_{friend} * \phi
    for label_type, f_label_probs in enumerate(f):
      phi_label_type = phi[label_type]
      for k in f_label_probs:
        v = f_label_probs[k] * phi_label_type
        if k in new_node_logprobs[label_type]:
          new_node_logprobs[label_type][k] += v
        else:
          new_node_logprobs[label_type][k] = v

  # Compute the actual probabilities in a numerically stable manner
  result = []
  for label_type, probs in enumerate(new_node_logprobs):
    d = {}
    if probs:
      maxval = max(probs.itervalues())
      denom = 0.0
      for k, v in probs.iteritems():
        expval = math.exp(v - maxval)
        d[k] = expval
        denom += expval
      for k in d:
        d[k] /= denom
    result.append(d)

  return result


def update_node_probs_combo(
    list my_node_probs,
    list friend_node_probs,
    int iteration=0,
    dict options={'alpha': 1.0, 'c': 5.0, 'stepsize': 100.0, 'pretty_sure_each': 1.1, 'pretty_sure_all': 0.8}):

  # Find the fraction of neighbors who are "pretty sure" of their labels
  num_pretty_sure = 0
  num_nonempty_friends = 0
  for f in friend_node_probs:
    pretty_sure = True
    for label_type, f_label_probs in enumerate(f):
      if len(f_label_probs) > 0:
        num_nonempty_friends += 1
        maxval = max([v for v in f_label_probs.itervalues()])
        if maxval < options['pretty_sure_each'] / len(f_label_probs):
          pretty_sure = False
          break
      if pretty_sure:
        num_pretty_sure += 1

  if (num_nonempty_friends < 50) or (num_pretty_sure < options['pretty_sure_all'] * num_nonempty_friends):
    return update_node_probs_DOT(my_node_probs, friend_node_probs, iteration, options)
  else:
    return update_node_probs_VAR(my_node_probs, friend_node_probs, iteration, options)

#  if iteration <= options['DOT_iter']:
#    return update_node_probs_DOT(my_node_probs, friend_node_probs, iteration, options)
#  else:
#    return update_node_probs_VAR(my_node_probs, friend_node_probs, iteration, options)


###
### End of Update Functions
###
def print_extra(node_probs, hidden_nodes, all_attr):
  """Helper function for debugging"""
  hidden_correct = []
  if all_attr is not None:
    for i in hidden_nodes:
      if ((len(node_probs[i]) == 2) & (all_attr[i][0] in node_probs[i][0])):
        this_item = (node_probs[i][0][all_attr[i][0]] > 0.5) 
      else:
        this_item = False
      hidden_correct.append(this_item)
      
    is_ego_correct = hidden_correct[0]
    num_others_correct = sum(hidden_correct[1:])
    return "ego_correct={}, # other hidden correct={}/{}".format(is_ego_correct, num_others_correct, len(hidden_nodes)-1)
  return ""

def do_inference(list train_attr,
                 dict ego_graph,
                 process_func=update_node_probs_LP,
                 list all_attr=None,
                 int num_iterations=5,
                 options=None,
                 verbose=False):
  cdef np.ndarray[long] hidden_nodes = np.where([np.isnan(x[0]) for x in train_attr])[0]
  cdef list node_probs = [[{label: 1.0} for label in all_labels]
      if not np.isnan(all_labels).any() else [{}] * len(all_labels)
      for all_labels in train_attr]

  cdef int i, idx
  cdef long h, node
  cdef list friend_probs, result, node_probs_new

  for i in xrange(num_iterations):
    if verbose:
      print "At step {}, node_probs[ego]={} {}".format(
          i,
          node_probs[0],
          print_extra(node_probs, hidden_nodes, all_attr))
    node_probs_new = []
    for idx, node in enumerate(hidden_nodes):
      friend_probs = [node_probs[f] for f in ego_graph[node]]
      if options:
        result = process_func(node_probs[node], friend_probs, iteration=i, options=options)
      else:
        result = process_func(node_probs[node], friend_probs, iteration=i)
      node_probs_new.append(result)
    for idx, h in enumerate(hidden_nodes):
      node_probs[h] = node_probs_new[idx]

  if verbose:
#    print "At end, hidden node_probs={}".format([node_probs[j] for j in hidden_nodes])
    print "At end, node_probs[ego]={} {}".format(
        node_probs[0],
        print_extra(node_probs, hidden_nodes, all_attr))
  return hidden_nodes, node_probs

def evaluate(hidden_nodes, node_probs, all_attr):
  est_hidden = [node_probs[i] for i in hidden_nodes]
  act_hidden = [all_attr[i] for i in hidden_nodes]

  # list of nodes, with each item being a tuple of estimate and actual
  joined = zip(est_hidden, act_hidden)  
  
  # group together the estimates and actuals for each label type for each node
  joined_each_type = [zip(a,b) for a,b in joined]  

  recall_at_1 = [0.0] * len(joined_each_type[0])
  recall_at_1_ego = [0.0] * len(joined_each_type[0])
  for nodeid, node_est_act in enumerate(joined_each_type):
    for label_type, est_act in enumerate(node_est_act):
      est_dict, actual_label = est_act
      sorted_est = sorted(est_dict.iteritems(), key=itemgetter(1), reverse=True)
      best_labels_est = [x[0] for x in sorted_est]
      if best_labels_est[0] == actual_label:
        recall_at_1[label_type] += 1.0 / len(hidden_nodes)
        if nodeid == 0:  # ego is always node 0
          recall_at_1_ego[label_type] = 1.0

  return (recall_at_1, recall_at_1_ego)

def run_one_instance(
    num_friends = 10,
    num_iterations = 10,
    prob_interfriend_edge = 0.1,
    p_L=np.kron([0.2, 0.8], [0.2, 0.3, 0.5]).reshape((2,3)),
    q_L=np.array([0.2, 0.8]),
    prob_hide_friend=0.3,
    options=None):

  all_attr, ego_graph = create_ego_net(
      num_friends=num_friends,
      prob_interfriend_edge=prob_interfriend_edge,
      p_L=p_L,
      q_L=q_L)
  train_attr = train_test(all_attr, prob_hide_friend=prob_hide_friend)
  result = {}
  VAR_special_options={'alpha':1.0, 'c':1.0}
  for name, process_func, options in [("LP", update_node_probs_LP, None),
                                      ("DOT", update_node_probs_DOT, None),
                                      ("VAR", update_node_probs_VAR, None),
                                      ("Combo", update_node_probs_combo, None)]:
#                                      ("VAR_c", update_node_probs_VAR, VAR_special_options)]:
    hidden_nodes, node_probs = do_inference(train_attr,
                                            ego_graph,
                                            process_func=process_func,
                                            num_iterations=num_iterations,
                                            options=options)
    recall_at_1, recall_at_1_ego = evaluate(hidden_nodes, node_probs, all_attr)
    result[name]= {'all_attr':all_attr,
                   'ego_graph':ego_graph,
                   'train_attr':train_attr, 
                   'hidden_nodes':hidden_nodes,
                   'node_probs':node_probs,
                   'recall_at_1':recall_at_1,
                   'recall_at_1_ego':recall_at_1_ego}
  return result


# From http://nbviewer.ipython.org/gist/aflaxman/6871948
# Get a bootstrap resample of an array-like
def bootstrap_resample(X, n=None):
  if n == None:
    n = len(X)
  resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
  X_resample = [X[i] for i in resample_i]
  return X_resample

# Compute a confidence range of a function from bootstrap samples
def bootstrap_conf(X, func, num_resamples=100):
  result = []
  for i in xrange(num_resamples):
    result.append(func(bootstrap_resample(X)))
  mean = np.mean(result)
  std = np.std(result)
  return (mean, mean - std, mean + std)

def run_multiple_instances(
    num_friends = 10,
    num_iterations = 10,
    prob_interfriend_edge = 0.1,
    p_L=np.kron([0.55, 0.45], [0.3, 0.3, 0.4]).reshape((2,3)),
    q_L=np.array([0.2, 0.8]),
    prob_hide_friend = 0.3,
    num_instances = 45,
    num_processes = 45,
    verbose=True,
    save_full_result=False):

  # Define a worker for each process
  def worker(seed, num_instances_for_process, out_q):
    np.random.seed(seed)
    this_recall_list = []
    full_result_list = []
    for i in xrange(num_instances_for_process):
      result = run_one_instance(
          num_friends=num_friends,
          num_iterations=num_iterations,
          prob_interfriend_edge=prob_interfriend_edge,
          p_L=p_L,
          q_L=q_L,
          options={'alpha': 1.0, 'c': 0.0, 'stepsize': 100.0},
          prob_hide_friend=prob_hide_friend)
      result_dict = dict([(k, v['recall_at_1_ego']) for k, v in result.iteritems()])

      for alg1, alg2 in [('LP', 'DOT'),
                         ('VAR', 'DOT'),
                         ('LP', 'VAR'),
                         ('Combo', 'DOT'),
                         ('LP', 'Combo'),
                         ('VAR_c', 'DOT'),
                         ('LP', 'VAR_c')]:
        if alg1 in result and alg2 in result:
          result_dict['{}-{}'.format(alg2, alg1)] = [result[alg2]['recall_at_1_ego'][i] - result[alg1]['recall_at_1_ego'][i]
                                   for i in xrange(len(q_L))]
      this_recall_list.append(result_dict)
      if save_full_result:
        full_result_list.append(result)
    out_q.put((this_recall_list, full_result_list))

  out_q = Queue()
  procs = []
  num_instances_left = num_instances

  # Start the processes
  for j in xrange(num_processes):
    instances_for_this_process = num_instances_left / (num_processes - j)
    num_instances_left -= instances_for_this_process
    p = Process(
        target=worker,
        args=(j, instances_for_this_process, out_q))
    procs.append(p)
    p.start()

  # Collect the results
  recall_list = []
  full_result_list = []
  for j in xrange(num_processes):
    this_recall_list, full_result = out_q.get()
    recall_list.extend(this_recall_list)
    full_result_list.extend(full_result)

  # Wait for all workers to finish
  for p in procs:
    p.join()

#  if save_full_result:
#    cPickle.dump(full_result_list, open("save_full_result_list.pkl", "wb"))

  result = {}
  result_recall_zipped = {}
  for algo in recall_list[0].iterkeys():

    # Get all recall lists for a given algo
    recall_list_for_algo = [x[algo] for x in recall_list]

    # Group recalls for each label type
    recall_zipped = zip(*recall_list_for_algo)
    result_recall_zipped[algo] = recall_zipped

    aggregate_recall = [(np.mean(l), bootstrap_conf(l, np.mean)) for l in recall_zipped]
    result[algo] = {label_type: (mean, conf_mean, conf_left, conf_right) for label_type, (mean, (conf_mean, conf_left, conf_right)) in enumerate(aggregate_recall)}

    if verbose:
      for label_type, (mean, (conf_mean, conf_left, conf_right)) in enumerate(aggregate_recall):
        print "Recall@1 for %s for label type %d: mean=%2.2f, confidence interval mean=%2.2f, confidence interval=[%2.2f, %2.2f]" % (algo, label_type, mean, conf_mean, conf_left, conf_right)

  return (result, result_recall_zipped, full_result_list)


###
### EXPERIMENTS
###

# Given a dataframe with p, q, mean, plot the surface
def plot_compare_algos(df, compared_algos='DOT-LP', savefig=False):

  xs = df['p']; ys = df['q']; zs = df['{} conf_mean'.format(compared_algos)]
  cols = np.unique(xs).shape[0]
  X = xs.reshape(-1, cols); Y = ys.reshape(-1, cols); Z = zs.reshape(-1, cols)

#  fig = plt.figure()
#  ax = fig.add_subplot(111, projection='3d')
#  ax.set_xlabel('p')
#  ax.set_ylabel('q')
#  ax.set_zlabel('conf_mean of {}'.format(compared_algos))

#  surf = ax.plot_trisurf(df['p'], df['q'], df['{} mean'.format(compared_algos)])
#  fig.tight_layout()

  # Contour plot
  fig, ax = plt.subplots()
  ax.set_xlabel('p')
  ax.set_ylabel('q')
  cnt = ax.contour(Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), extent=[0, 1, 0, 1])
  if savefig:
    plt.savefig('EXPT.{}.png'.format(compared_algos), bbox_inches='tight')
  else:
    plt.show()


# For various values of p_L and q_L, get the results of various algos
def expt_compare_algos(prob_hide_friend_range=np.arange(0.0, 0.51, 0.05),
                       p_range=np.arange(0.1, 0.51, 0.025),
                       q_range=np.arange(0.1, 0.51, 0.025),
                       outputfile='EXPT.compare_algos.csv',
                       num_instances=200,
                       num_iterations=10):
  result = []
  columns = None
  prob_interfriend_edge = 0.3
  for prob_hide_friend in prob_hide_friend_range:
    for p in p_range:
      for q in q_range:
        print "prob_hide_friend={}, p={}, q={}".format(prob_hide_friend, p, q)
        p_L = np.kron([p, 1.0 - p], [0.3, 0.3, 0.4]).reshape((2,3))
        q_L = np.array([q, 1.0 - q])
        result_tuple = run_multiple_instances(
                           num_friends=1000,
                           p_L=p_L,
                           q_L=q_L,
                           prob_hide_friend=prob_hide_friend,
                           prob_interfriend_edge=prob_interfriend_edge,
                           num_instances=num_instances,
                           num_iterations=num_iterations,
                           verbose=False)
        result_one_setting = result_tuple[0]

        # Figure out the columns for the dataframe. We need this only once
        if columns is None:
          columns = ['p', 'q', 'prob_hide_friend', 'prob_interfriend_edge']
          columns.extend(['{} {}'.format(alg, coltype)
                          for alg in result_one_setting.keys()
                          for coltype in ('mean', 'conf_mean', 'conf_left', 'conf_right')])

        # We want to get results for label_type=0 (since we manipulated only
        # the first label type in p_L)
        row = [p, q, prob_hide_friend, prob_interfriend_edge]
        for x in result_one_setting.keys():
          if x in result_one_setting:
            algo_res = result_one_setting[x][0] 
          else:
            algo_res = [np.nan] * 4
          row.extend(algo_res)
        result.append(row)
        df = pd.DataFrame(result, columns=columns)
        df.to_csv(outputfile)
  
#  print df
  return df
