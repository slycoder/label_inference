import generate_graph
import gen0
import numpy as np
import cPickle
import pstats, cProfile

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm


def do_test():
  num_friends=1000
  num_iterations=10
  p_L=np.kron([0.25, 0.75], [0.3, 0.3, 0.4]).reshape((2,3))
  q_L=np.array([0.3, 0.7])
  prob_hide_friend=0.0
  prob_interfriend_edge=0.3

#  all_attr, ego_graph = generate_graph.create_ego_net(
#      num_friends=num_friends,
#      prob_interfriend_edge=prob_interfriend_edge,
#      p_L=p_L,
#      q_L=q_L)
#  all_attr, ego_graph = gen0.create_ego_net(
#      num_friends=num_friends,
#      prob_interfriend_edge=prob_interfriend_edge,
#      p_L=p_L,
#      q_L=q_L)

#  result = generate_graph.run_one_instance(
#      num_friends=num_friends,
#      num_iterations=num_iterations,
#      prob_interfriend_edge=prob_interfriend_edge,
#      p_L=p_L,
#      q_L=q_L,
#      options={'alpha': 1.0, 'c': 0.0, 'stepsize': 100.0},
#      prob_hide_friend=prob_hide_friend)
  
#  generate_graph.run_one_instance(num_friends=1000,
#                         p_L=np.kron([0.2, 0.8], [0.3, 0.3, 0.4]).reshape((2,3)),
#                         q_L=np.array([0.3, 0.7]),
#                         prob_hide_friend=0.0,
#                         prob_interfriend_edge=0.0,
#                         num_iterations=3)

  result_tuple = generate_graph.run_multiple_instances(num_friends=1000,
                         p_L=p_L,
                         q_L=q_L,
                         prob_hide_friend=prob_hide_friend,
                         prob_interfriend_edge=prob_interfriend_edge,
                         num_instances=50,
                         verbose=True,
                         save_full_result=False)
  return result_tuple

#  result = gen0.run_one_instance(
#      num_friends=num_friends,
#      num_iterations=num_iterations,
#      prob_interfriend_edge=prob_interfriend_edge,
#      p_L=p_L,
#      q_L=q_L,
#      options={'alpha': 1.0, 'c': 0.0, 'stepsize': 100.0},
#      prob_hide_friend=prob_hide_friend)

#  generate_graph.run_multiple_instances(
#                         num_friends=num_friends,
#                         num_iterations=num_iterations,
#                         p_L=p_L,
#                         q_L=q_L,
#                         prob_hide_friend=prob_hide_friend,
#                         prob_interfriend_edge=prob_interfriend_edge,
#                         num_instances=100,
#                         verbose=True)

#  generate_graph.expt_compare_algos()

def run_expt():
  df = generate_graph.expt_compare_algos(
      prob_hide_friend_range=np.arange(0.0, 0.51, 0.1),
      p_range=np.arange(0.2, 0.4, 0.05),
      q_range=np.arange(0.2, 0.4, 0.05),
      outputfile='EXPT.combo.csv',
      num_instances=50)
  return df

def identify_bad_cases():
  num_friends=1000
  num_iterations=5
  p_L=np.kron([0.25, 0.75], [0.3, 0.3, 0.4]).reshape((2,3))
  q_L=np.array([0.3, 0.7])
  prob_hide_friend=0.35
  prob_interfriend_edge=0.3

  result_tuple = generate_graph.run_multiple_instances(num_friends=1000,
                         p_L=p_L,
                         q_L=q_L,
                         prob_hide_friend=prob_hide_friend,
                         prob_interfriend_edge=prob_interfriend_edge,
                         num_instances=50,
                         verbose=True,
                         save_full_result=True)
  result, result_recall_zipped, full_result_list = result_tuple
  bad_indices = [i for i, x in enumerate(result_recall_zipped['DOT-VAR'][0]) if x > 0.1]
  bad_cases = [full_result_list[i] for i in bad_indices]
  return bad_cases

def do_one_bad_case(bad_cases, index=0):
  bad_case = bad_cases[index]['VAR']
  all_attr = bad_case['all_attr']
  ego_graph = bad_case['ego_graph']
  train_attr = bad_case['train_attr']
  for name, process_func in [("DOT", generate_graph.update_node_probs_DOT),
                             ("VAR", generate_graph.update_node_probs_VAR)]:
    print "Processing {}:".format(name)
    hidden_nodes, node_probs = generate_graph.do_inference(
                                            train_attr,
                                            ego_graph,
                                            process_func=process_func,
                                            num_iterations=num_iterations,
                                            verbose=True)
    print ""


def do_big_plot(df):
  prob_hide_friend_series = [0.0, 0.2, 0.4]
  df1 = df[['p', 'q', 'prob_hide_friend', 'prob_interfriend_edge', 'DOT-LP conf_mean', 'DOT-VAR conf_mean', 'DOT-Combo conf_mean']]
  df2 = df1[['DOT-LP conf_mean', 'DOT-VAR conf_mean', 'DOT-Combo conf_mean']]
  min_p = df['p'].min(); max_p = df['p'].max()
  min_q = df['q'].min(); max_q = df['q'].max()
  
  min_x = min_p - 0.5 * (max_p - min_p); max_x = max_p + 0.5 * (max_p - min_p)
  min_y = min_q - 0.5 * (max_q - min_q); max_y = max_q + 0.5 * (max_q - min_q)

  toplot_list = ['DOT-LP', 'DOT-VAR', 'DOT-Combo']
  fig = plt.figure(figsize=plt.figaspect(1.0 * len(toplot_list) / len(prob_hide_friend_series)))
  plot_idx = 0
  for toplot in toplot_list:
    for i, prob_hide_friend in enumerate(prob_hide_friend_series):
      plot_idx += 1
      df1 = df[abs(df['prob_hide_friend'] - prob_hide_friend) < 0.001]
      df2 = df1['{} conf_mean'.format(toplot)]
      min_val = df2.min(); max_val = df2.max()
      min_z = min_val - 0.5 * (max_val - min_val); max_z = max_val + 0.5 * (max_val - min_val)

      ax = fig.add_subplot(*(len(toplot_list), len(prob_hide_friend_series), plot_idx), projection='3d');
      xs = df1['p']; ys = df1['q']; zs = df1['{} conf_mean'.format(toplot)]
      cols = np.unique(xs).shape[0]; X = xs.reshape(-1, cols); Y = ys.reshape(-1, cols); Z = zs.reshape(-1, cols)
      ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
      cset = ax.contour(X, Y, Z, zdir='z', offset=min_z, cmap=cm.coolwarm)
      
      ax.set_xlim(min_x, max_x)
      ax.set_ylim(min_y, max_y)
      ax.set_zlim(min_z, max_z)

      ax.set_xlabel('p')
      ax.set_ylabel('q')
      ax.set_title(toplot)

      ax.set_xticks(np.linspace(min_x, max_x, 3))
      ax.set_yticks(np.linspace(min_y, max_y, 3))
      ax.set_zticks(np.linspace(min_z, max_z, 3))

  plt.tight_layout()
  plt.show()


def do_plot(df, prob_hide_friend=0.0, toplot='Combo'):
  dfp = df[abs(df['prob_hide_friend'] - prob_hide_friend) < 0.001]
  xs = dfp['p']; ys = dfp['q']; zs = dfp['{} conf_mean'.format(toplot)]
  min_x = min(xs) - 0.5 * (max(xs) - min(xs)); max_x = max(xs) + 0.5 * (max(xs) - min(xs))
  min_y = min(ys) - 0.5 * (max(ys) - min(ys)); max_y = max(ys) + 0.5 * (max(ys) - min(ys))
  min_z = min(zs) - 0.5 * (max(zs) - min(zs)); max_z = max(zs) + 0.5 * (max(zs) - min(zs))

  cols = np.unique(xs).shape[0]; X = xs.reshape(-1, cols); Y = ys.reshape(-1, cols); Z = zs.reshape(-1, cols)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d'); ax.set_xlabel('p'); ax.set_ylabel('q');
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
  cset = ax.contour(X, Y, Z, zdir='z', offset=min_z, cmap=cm.coolwarm)
#  cset = ax.contour(X, Y, Z, zdir='x', offset=min_x, cmap=cm.coolwarm)
#  cset = ax.contour(X, Y, Z, zdir='y', offset=max_y, cmap=cm.coolwarm)

  ax.set_xlabel('X')
  ax.set_xlim(min_x, max_x)
  ax.set_ylabel('Y')
  ax.set_ylim(min_y, max_y)
  ax.set_zlabel('Z')
  ax.set_zlim(min_z, max_z)

  ax.set_xlabel('p')
  ax.set_ylabel('q')
  ax.set_zlabel(toplot)

  plt.show()




#cProfile.runctx("do_test()", globals(), locals(), "Profile.prof")
#s = pstats.Stats("Profile.prof")
#s.strip_dirs().sort_stats("time").print_stats()
