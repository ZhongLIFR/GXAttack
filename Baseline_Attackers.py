#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:39:15 2024

@author: Anonymous

Baseline Attack functions
"""


import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

device = "cuda" if torch.cuda.is_available() else "cpu"


class RandomAttacks:
    
    def random_flipping(self, data, budget_val):
        """
        Randomly flip edges in the graph contained within the data object of PyTorch Geometric,
        within the constraints of the specified budget.

        Parameters:
        - data (Data): A PyTorch Geometric Data object.
        - budget_val (int): The number of edge flips allowed.

        Returns:
        - perturbed_data (Data): A new Data object with flipped edges.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Check if edge_attr exists, if not, create it with all ones (indicating default weights)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weights = data.edge_attr
        else:
            edge_weights = torch.ones((num_edges,), dtype=torch.float)

        # Generate random indices to flip
        flip_indices = np.random.choice(num_edges, budget_val, replace=False)

        # Set to track existing edges for quick lookup
        existing_edges = set()
        for i in range(num_edges):
            node_a, node_b = edge_index[:, i]
            existing_edges.add((node_a.item(), node_b.item()))

        # Edges to be added or removed
        new_edges = []
        new_weights = []

        for i in range(num_edges):
            node_a, node_b = edge_index[:, i]
            if i in flip_indices:
                # Flip the edge: if it exists, attempt to remove it
                if (node_a.item(), node_b.item()) in existing_edges:
                    existing_edges.remove((node_a.item(), node_b.item()))
                else:
                    # If it does not exist, add it with weight 1
                    existing_edges.add((node_a.item(), node_b.item()))
                    new_edges.append([node_a, node_b])
                    new_weights.append(1)  # Weight for new edges is 1
            else:
                new_edges.append([node_a, node_b])
                new_weights.append(edge_weights[i])

        # Convert new edges and weights to tensors
        perturbed_edge_index = torch.tensor(new_edges).t().contiguous()
        perturbed_edge_weights = torch.tensor(new_weights, dtype=torch.float)

        # Create a new Data object
        perturbed_data = Data(x=data.x, edge_index=perturbed_edge_index, edge_attr=perturbed_edge_weights)
        
        # Copy all other data attributes
        for key, value in data:
            if key not in ['x', 'edge_index', 'edge_attr']:
                setattr(perturbed_data, key, value)

        return perturbed_data
    
    @staticmethod
    def random_rewiring(data, node_idx, k, budget_val):
        # Obtain k-hop subgraph around the target node
        subgraph_nodes, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, k, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        
        num_edges = sub_edge_index.size(1)
        num_sub_nodes = subgraph_nodes.size(0)
        
        # Check if the current budget_val is feasible
        feasible_budget = min(budget_val, num_edges, edge_mask.numel())
        if feasible_budget < budget_val:
            print(f"Requested budget ({budget_val}) is not feasible. Adjusted to {feasible_budget} based on available edges.")
        
        # Generate random edge indices to rewire
        rewiring_indices = np.random.choice(num_edges, feasible_budget, replace=False)
        
        # Collect new edges ensuring no duplicates or self-loops
        new_edges = []
        existing_edges = set(map(tuple, sub_edge_index.t().numpy().tolist()))
        attempts = 0
        max_attempts = feasible_budget * 10  # Adjust max attempts based on the feasible budget
        
        while len(new_edges) < feasible_budget and attempts < max_attempts:
            rand_nodes = np.random.choice(num_sub_nodes, 2, replace=False)
            new_edge = (rand_nodes[0], rand_nodes[1])
            reverse_edge = (rand_nodes[1], rand_nodes[0])
        
            if new_edge not in existing_edges and reverse_edge not in existing_edges:
                new_edges.append(new_edge)
                existing_edges.add(new_edge)
                existing_edges.add(reverse_edge)
            attempts += 1
        
        # Update subgraph edges by removing selected ones and adding new ones
        sub_edge_index = np.delete(sub_edge_index.numpy(), rewiring_indices, axis=1)
        
        # Ensure new_edges_array is 2D
        new_edges_array = np.array(new_edges).T
        if new_edges_array.ndim == 1:  # This happens if new_edges is empty or only contains one edge
            new_edges_array = new_edges_array.reshape(2, -1)  # Reshape to 2D if necessary
        
        sub_edge_index = np.concatenate([sub_edge_index, new_edges_array], axis=1)
        
        # Convert numpy array back to tensor
        sub_edge_index_tensor = torch.tensor(sub_edge_index, dtype=torch.long)
        
        # Replace in the full edge_index tensor
        perturbed_edge_index = data.edge_index.clone()
        try:
            perturbed_edge_index[:, edge_mask.bool()] = sub_edge_index_tensor
        except RuntimeError as e:
            print("Failed to insert due to mismatched sizes:", str(e))
            return data  # Optionally, return the original data on failure
        
        # Create a new Data object with updated edges
        perturbed_data = Data(x=data.x, edge_index=perturbed_edge_index)
        # Copy other attributes
        for key, value in data:
            if key not in ['x', 'edge_index']:
                setattr(perturbed_data, key, value)
        
        return perturbed_data

