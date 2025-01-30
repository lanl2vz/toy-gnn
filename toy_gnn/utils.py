import torch
from torch_geometric.data import Data
import random

def compute_edge_labels(edge_index, num_nodes):
    """Compute binary labels: 1 for nodes with multiple edges, 0 for single edge"""
    edge_count = torch.zeros(num_nodes, dtype=torch.long)  # Change to long dtype
    unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
    edge_count[unique_nodes] = counts
    
    # 1 if node has multiple edges, 0 if single edge
    return (edge_count > 1).long()

def create_sample_graph():
    """Create a sample graph with mix of single and multiple edge nodes"""
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 4],  # Source nodes
        [1, 0, 2, 1, 3, 2, 3],  # Target nodes
    ], dtype=torch.long)
    
    num_nodes = 5
    x = torch.ones((num_nodes, 1))  # Simple feature: all ones
    y = compute_edge_labels(edge_index, num_nodes)
    
    return Data(x=x, edge_index=edge_index, y=y)

def generate_random_graph(min_nodes=5, max_nodes=10, edge_probability=0.3):
    """Generate a random graph with varying number of nodes and edges"""
    num_nodes = random.randint(min_nodes, max_nodes)
    
    # Create possible edges (excluding self-loops)
    all_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    
    # Randomly select edges
    selected_edges = random.sample(all_edges, 
                                 k=int(len(all_edges) * edge_probability))
    
    # Convert to edge index format
    edge_index = torch.tensor(list(zip(*selected_edges)), dtype=torch.long)
    
    # Create node features (using degree as feature)
    x = torch.ones((num_nodes, 1))  # Simple feature: all ones
    
    # Compute labels
    y = compute_edge_labels(edge_index, num_nodes)
    
    return Data(x=x, edge_index=edge_index, y=y)

def create_dataset(num_graphs=50):
    """Create a dataset of multiple random graphs"""
    train_graphs = [generate_random_graph() for _ in range(num_graphs)]
    val_graphs = [generate_random_graph() for _ in range(num_graphs // 5)]
    test_graphs = [generate_random_graph() for _ in range(num_graphs // 5)]
    return train_graphs, val_graphs, test_graphs
