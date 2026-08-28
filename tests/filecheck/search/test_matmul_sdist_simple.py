# RUN: python %s 2>&1 | filecheck %s
"""
Test strategy SDist Simple (no distribution) on matmul
"""
import utils
from xtc.search.sdist_strategies import Strategy_SDist_Simple as Strategy

graph = utils.get_graph_matmul()
backend = utils.get_backend(graph)
strategy = Strategy(graph)

utils.print_all_opt_schedules(backend, strategy)
utils.print_exhaustive_samples(backend, strategy, 100)
