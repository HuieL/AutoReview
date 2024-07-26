import os
import glob
import torch
from torch_geometric.data import Data

def load_graphs(folder_path):
    graph_files = glob.glob(os.path.join(folder_path, '*.pt'))
    graphs = []
    for file in graph_files:
        graphs.append(torch.load(file))
    return graphs

def merge_graphs(graphs):
    merged_data = Data()
    merged_data.x = []
    merged_data.edge_index = []
    merged_data.arxiv_id = []
    merged_data.content = []
    merged_data.abstract = []
    merged_data.title = []
    ### Update codes here for reviewd papers
    merge_data.comment = [] # str
    merge_data.score = []
    ###

    arxiv_id_to_index = {}
    current_index = 0
    original_nodes = 0

    for graph in graphs:
        original_nodes += len(graph.x[0])
        for i, arxiv_id in enumerate(graph.arxiv_ids):
            if arxiv_id not in arxiv_id_to_index:
                arxiv_id_to_index[arxiv_id] = current_index
                merged_data.arxiv_id.append(arxiv_id)
                merged_data.content.append(graph.content[i])
                merged_data.abstract.append(graph.abstract[i])
                merged_data.title.append(graph.title[i])
                merged_data.comment.append(###)
                merged_data.score.append(###)
                current_index += 1

        for edge in graph.edge_index.t():
            source = arxiv_id_to_index[graph.arxiv_ids[edge[0]]]
            target = arxiv_id_to_index[graph.arxiv_ids[edge[1]]]
            merged_data.edge_index.append([source, target])

    merged_data.edge_index = torch.tensor(merged_data.edge_index).t().contiguous()
    return merged_data

def print_arxiv_id_titles(graphs):
    descs = []
    for graph in graphs:
        descs.append(f"{graph.arxiv_ids[0]}: {graph.title[0]}")
    return "\n".join(descs)

def merge(folder_path, output_path)
  graphs = load_graphs(folder_path)
  merged_graph = merge_graphs(graphs)
  
  print(merged_graph)
  torch.save(merged_graph, output_path)
