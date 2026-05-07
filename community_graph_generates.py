import networkx as nx
import numpy as np
from itertools import chain
import os

from typing import Iterable, Tuple, Hashable
import random
import os
import pickle
from tqdm import tqdm
from collections import defaultdict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from typing import Any, Dict, List, Optional, Tuple, Hashable, Set
import math

def gen_strong_sbm(n=1000, k=8, p_in=0.08, p_out=0.002, directed=True, seed=42):
    rng = np.random.default_rng(seed)
    sizes = [n // k] * k
    sizes[0] += n - sum(sizes) 

    B = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]

    G = nx.stochastic_block_model(sizes, B, seed=seed, directed=directed)
    node2comm = {}
    start = 0
    for cid, sz in enumerate(sizes):
        for v in range(start, start+sz):
            node2comm[v] = cid
        start += sz
    return G, node2comm

def sbm_example(n, k, p_in, p_out, directed):
    G, node2comm = gen_strong_sbm(n=n, k=k, p_in=p_in, p_out=p_out, directed=directed, seed=1)
    density = nx.density(G)
    print(f"Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges, density: {density}")
    return G




def check_all_node_pairs(G):
    nodes = list(G.nodes())
    max_length = 0
    node_pairs = []
    for n1 in nodes:
        for n2 in nodes:
            if nx.has_path(G, n1, n2) and n1!=n2:
                length = nx.shortest_path_length(G, source=n1, target=n2)
                if length<=2: continue
                if length > max_length:
                    max_length = length
                node_pairs.append([n1,n2])
    return node_pairs

def canon(path, undirected=False):
    t = tuple(path)
    if undirected:
        r = tuple(reversed(t))
        return min(t, r)
    return t
    

    
def collect_shortest_paths_by_k(G, path_lists, undirected=False, min_path = 2):
    paths_by_k = defaultdict(set)      
    tuple2string = {}                 
    for p in tqdm(path_lists, total=len(path_lists)):
        s, e = p[0], p[-1]
        for path in nx.all_shortest_paths(G, source=s, target=e):
            if len(path) < min_path:
                continue
            k = len(path) - 1  # hops
            key = canon(path, undirected=undirected)
            paths_by_k[k].add(key)
            s_, e_ = path[0], path[-1]
            string = f"S {s_} E {e_} PATH " + " ".join(map(str, path)) + " END_P <END>"
            tuple2string.setdefault(key, string)
    paths_by_k = {k: list(v) for k, v in paths_by_k.items()}
    return paths_by_k, tuple2string

def subpaths_of_hops(path_tuple, sub_hops):
    L = len(path_tuple) - 1  # hops
    k = sub_hops
    if k > L:
        return []
    win = k + 1
    return [path_tuple[i:i+win] for i in range(0, len(path_tuple) - win + 1)]


def split_paths_disjoint_by_subpath(paths_by_k, test_ratio=0.2, seed=42):
    random.seed(seed)
    cand_train_by_k, cand_test_by_k = {}, {}
    for k, plist in paths_by_k.items():
        items = list(plist)
        random.shuffle(items)
        cut = int(len(items) * (1 - test_ratio))
        cand_train_by_k[k] = items[:cut]
        cand_test_by_k[k]  = items[cut:]

    ban = defaultdict(set)
    for k2, train_paths in cand_train_by_k.items():
        for pt in train_paths:
            for k1 in range(0, k2 + 1):
                for sp in subpaths_of_hops(pt, k1):
                    ban[k1].add(sp)

    test_by_k = {}
    for k1, test_paths in cand_test_by_k.items():
        test_by_k[k1] = [p for p in test_paths if p not in ban[k1]]

    ban_train = False
    test_ban = defaultdict(set)
    for k1, tpaths in test_by_k.items():
        for pt in tpaths:
            test_ban[k1].add(pt)
    train_by_k = {}
    for k2, tr_paths in cand_train_by_k.items():
        keep = []
        for pt in tr_paths:
            if ban_train:
                bad = False
                for k1 in range(0, k2 + 1):
                    if bad: break
                    for sp in subpaths_of_hops(pt, k1):
                        if sp in test_ban[k1]:
                            bad = True
                            break
                if not bad:
                    keep.append(pt)
            else:
                keep.append(pt)
        train_by_k[k2] = keep

    return train_by_k, test_by_k

def export_string_sets(train_by_k, test_by_k, tuple2string):
    train_strings_by_k = {k: [tuple2string[t] for t in v] for k, v in train_by_k.items()}
    test_strings_by_k  = {k: [tuple2string[t] for t in v] for k, v in test_by_k.items()}
    train_strings = [s for k in sorted(train_strings_by_k) for s in train_strings_by_k[k]]
    test_strings  = [s for k in sorted(test_strings_by_k)  for s in test_strings_by_k[k]]
    return train_strings, test_strings, train_strings_by_k, test_strings_by_k

def build_datasets_with_subpath_guard(G, path_lists, test_ratio=0.2, seed=42, undirected=False, min_path=2):
    paths_by_k, tuple2string = collect_shortest_paths_by_k(G, path_lists, undirected=undirected, min_path=min_path)
    train_by_k, test_by_k = split_paths_disjoint_by_subpath(paths_by_k, test_ratio=test_ratio, seed=seed)
    return export_string_sets(train_by_k, test_by_k, tuple2string)



def all_n_node_subgraphs(
    G: nx.Graph,
    n: int,
    *,
    induced: bool = True,
    weakly_connected_only: bool = False,
    remove_k_edges: int = 0,         
    prune_isolates: bool = True,
    skip_if_not_enough_edges: bool = True,
) -> List[nx.Graph]:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > G.number_of_nodes():
        return []
    if remove_k_edges < 0:
        raise ValueError(f"remove_k_edges must be >= 0, got {remove_k_edges}")

    nodes = list(G.nodes())
    out: List[nx.Graph] = []
    seen: Set[Tuple[frozenset, frozenset]] = set()  # (node_set, edge_set)
    total = math.comb(len(nodes), n)

    def _prune_isolates_inplace(H: nx.Graph) -> None:
        if not prune_isolates:
            return
        iso = list(nx.isolates(H))  # DiGraph: total degree=0
        if iso:
            H.remove_nodes_from(iso)

    def _signature(H: nx.Graph) -> Tuple[frozenset, frozenset]:
        ns = frozenset(H.nodes())
        if H.is_multigraph():
            es = frozenset(H.edges(keys=True))  # (u,v,k)
        else:
            es = frozenset(H.edges())      
        return (ns, es)

    def _remove_edge_inplace(H: nx.Graph, e) -> None:
        if H.is_multigraph():
            u, v, k = e
            if H.has_edge(u, v, k):
                H.remove_edge(u, v, k)
        else:
            u, v = e
            if H.has_edge(u, v):
                H.remove_edge(u, v)

    for node_set in tqdm(
        itertools.combinations(nodes, n),
        total=total,
        desc=f"Enumerating {n}-node subgraphs",
    ):
        H_view = G.subgraph(node_set)

        if weakly_connected_only:
            if H_view.is_directed():
                if not nx.is_weakly_connected(H_view):
                    continue
            else:
                if not nx.is_connected(H_view):
                    continue

        H0 = H_view.copy()

        _prune_isolates_inplace(H0)

        if remove_k_edges == 0:
            sig = _signature(H0)
            if sig not in seen:
                seen.add(sig)
                out.append(H0)
            continue

        edges = list(H0.edges(keys=True)) if H0.is_multigraph() else list(H0.edges())
        m = len(edges)

        if m == 0:
            if skip_if_not_enough_edges:
                continue
            sig = _signature(H0)
            if sig not in seen:
                seen.add(sig)
                out.append(H0)
            continue

        k = remove_k_edges
        if k > m:
            if skip_if_not_enough_edges:
                continue
            k = m 

        for edges_to_remove in itertools.combinations(edges, k):
            H = H0.copy()
            for e in edges_to_remove:
                _remove_edge_inplace(H, e)

            _prune_isolates_inplace(H)

            sig = _signature(H)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(H)

    return out



def all_pairs_shortest_path_list_with_pairs(
    G: nx.Graph,
    *,
    weight: Optional[str] = None,
    min_num_nodes: int = 3,
    dedup: bool = False,
) -> List[Tuple[Hashable, Hashable, List[Hashable]]]:

    out: List[Tuple[Hashable, Hashable, List[Hashable]]] = []
    seen = set()

    if weight is None:
        paths_iter = nx.all_pairs_shortest_path(G)
    else:
        paths_iter = nx.all_pairs_dijkstra_path(G, weight=weight)

    for u, path_map in paths_iter:
        for v, path in path_map.items():
            if len(path) < min_num_nodes:
                continue

            if dedup:
                sig = tuple(path)
                if sig in seen:
                    continue
                seen.add(sig)

            out.append(path)

    return out

def generate_edge_list(g):
    txt=str(g.edges())[1:-1].replace(', ',' ')
    txt=txt.replace(') (',' | ')
    txt=txt.replace('(','')
    txt=txt.replace(')','')
    return txt
def path_txt_generate(path):
    question = f' S {path[0]} E {path[-1]} '
    path_txt = question + 'PATH '
    for node in path:
        path_txt += str(node) + ' '
    path_txt += 'END_P'
    return path_txt

def build_condition_data(G, nodes_range = [4,15]):
    subgraphs_list = []
    for i in range(nodes_range[0], nodes_range[1]):
        for j in range(4):
            subgraphs = all_n_node_subgraphs(G, i, remove_k_edges=j)
            subgraphs_list.extend(subgraphs)
    subgraphs = subgraphs_list
    print(f'Total {len(subgraphs)} subgraphs generated.')
    random.shuffle(subgraphs)
    test_num = int(len(subgraphs) * 0.1)
    test_graph = subgraphs[:test_num]
    train_graph = subgraphs[test_num:]
    train_set = []
    test_set = []
    for g in tqdm(train_graph, total=len(train_graph)):
        txt = generate_edge_list(g)
        paths = all_pairs_shortest_path_list_with_pairs(g)
        for path in paths:
            path_txt = path_txt_generate(path)
            string = '<START> ' + txt + path_txt +' <END>'
            train_set.append(string)
    for g in tqdm(test_graph, total=len(test_graph)):
        txt = generate_edge_list(g)
        paths = all_pairs_shortest_path_list_with_pairs(g)
        for path in paths:
            path_txt = path_txt_generate(path)
            string = '<START> ' + txt + path_txt + ' <END>'
            test_set.append(string)
    print(train_set[:2])
    print(test_set[:2])
    print(len(train_set), len(test_set))
    return train_set, test_set

def data_prepare(G, base_model_path, tokenizer_path=None, min_path=2,max_nodes=4, condition = False):
    path_lists = check_all_node_pairs(G)
    print(path_lists[:5])
    random.shuffle(path_lists)
    dataset = []
    base_model_path = tokenizer_path
    if condition:
        train_data, test_data = build_condition_data(G, nodes_range = [4,max_nodes - 1])
        print(len(train_data))
        print(len(test_data))
        if os.path.exists(base_model_path) == False:
            os.makedirs(base_model_path)
        with open(os.path.join(base_model_path, f'condition_train.pkl'),'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(base_model_path, f'condition_test.pkl'),'wb') as f:
            pickle.dump(test_data, f)
        dataset = train_data + test_data
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["<UNK>"])
        tokenizer.train_from_iterator(dataset, trainer)
        tokenizer = PreTrainedTokenizerFast(min_frequency=2,
                                            tokenizer_object=tokenizer)
        tokenizer.add_special_tokens({'pad_token': '<PAD>',
                                    'eos_token': '<END>',
                                    'unk_token': '<UNK>',
                                    'bos_token': '<START>'})
        
        
        save_tokenizer_path = os.path.join(base_model_path, f"condition_baby_tokenizer.json")
        tokenizer.save_pretrained(save_tokenizer_path)
    else:
        train_data, test_data, train_by_k_str, test_by_k_str = build_datasets_with_subpath_guard(G, path_lists, test_ratio=0.1, seed=0, undirected=False, min_path=min_path)
        print(len(train_data))
        print(len(test_data))
        if os.path.exists(base_model_path) == False:
            os.makedirs(base_model_path)
        with open(os.path.join(base_model_path, f'path_soft_train.pkl'),'wb') as f:
            pickle.dump(train_data, f)
        with open(os.path.join(base_model_path, f'path_soft_test.pkl'),'wb') as f:
            pickle.dump(test_data, f)
        dataset = train_data + test_data
        tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(special_tokens=["<UNK>"])
        tokenizer.train_from_iterator(dataset, trainer)
        tokenizer = PreTrainedTokenizerFast(min_frequency=2,
                                            tokenizer_object=tokenizer)
        tokenizer.add_special_tokens({'pad_token': '<PAD>',
                                    'eos_token': '<END>',
                                    'unk_token': '<UNK>',
                                    'bos_token': '<START>'})
        
        
        save_tokenizer_path = os.path.join(base_model_path, f"baby_tokenizer.json")
        tokenizer.save_pretrained(save_tokenizer_path)
    
import itertools
from typing import List

import networkx as nx
import itertools
from typing import List, Literal, Set, Tuple


if __name__ == "__main__":
    n = 10 
    
    min_path = 2    
    method = 'path'
    types = 'er' # er for intrinsic reasoning, comm for extrinsic reasoning
    
    contition = True
    if types == 'er':
        for p in [ 0.4]:
            base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_er'
            if os.path.exists(base_model_path) == False:
                os.makedirs(base_model_path)
            
            graph_file = os.path.join(base_model_path,f"{n}_{p}.pkl")
            if os.path.exists(graph_file):
                with open(os.path.join(base_model_path,f"{n}_{p}.pkl"),'rb') as f:
                    G = pickle.load(f)
            else:
                if os.path.exists(base_model_path) == False:
                    os.makedirs(base_model_path)
                G = nx.gnp_random_graph(n, p, directed=True)
                with open(os.path.join(base_model_path,f"{n}_{p}.pkl"),'wb') as f:
                    pickle.dump(G, f)
            tokenizer_path = os.path.join(base_model_path,f"{n}_{p}")
            data_prepare(G, base_model_path, tokenizer_path, min_path=min_path, condition=contition, max_nodes=n)
                
    elif types == 'comm':
        for k_ratio in [ 0.1]:
            k = int(n*k_ratio)
            for p_in in [0.1]:
                for p_out in [0.01]:
                    print(n, k, p_in, p_out)
                    base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com'
                    if os.path.exists(base_model_path) == False:
                        os.makedirs(base_model_path)
                    
                    graph_file = os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}.pkl")
                    if os.path.exists(graph_file):
                        with open(os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}.pkl"),'rb') as f:
                            G = pickle.load(f)
                    else:
                        if os.path.exists(base_model_path) == False:
                            os.makedirs(base_model_path)
                        G = sbm_example(n=n, k=k, p_in=p_in, p_out=p_out, directed=True)
                        with open(os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}.pkl"),'wb') as f:
                            pickle.dump(G, f)
                    if method!= 'path':
                        method_dicts_save_path = os.path.join(base_model_path,f"{method}_{n}_{k_ratio}_{p_in}_{p_out}.pkl")
                        if os.path.exists(method_dicts_save_path):
                            with open(method_dicts_save_path, 'rb') as f:
                                pickle.load(G)
                        
                    tokenizer_path = os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}")
                    
                    data_prepare(G, base_model_path, tokenizer_path, min_path=min_path, condition=contition,max_nodes=n)
    else:
        print("no such graph type")