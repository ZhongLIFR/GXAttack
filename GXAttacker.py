#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:04:04 2024

@author: Anonymous

Attack function
"""

# =============================================================================
# Define utility functions
# =============================================================================
import math
import copy
import torch
import networkx as nx
from functools import partial
from collections import Counter
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
from typing import Callable
from torch_geometric.utils.num_nodes import maybe_num_nodes



# =============================================================================
# Define the explainer PGExplainer()
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import tqdm
import time

from typing import Optional
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv

##------------------------------------------------------
##We can directly use the utility functions from GraphXAI
##------------------------------------------------------

from graphxai.explainers._base import _BaseExplainer
##It contains the following functions:
    # __set_embedding_layer
    # _get_embedding
    # _set_masks
    # _clear_masks
    # _flow
    # _predict
    # _prob_score_func_graph
    # _prob_score_func_node
    # _get_activation
    # _get_k_hop_subgraph
    # get_explanation_node
    # get_explanation_graph
    # get_explanation_link
    
from graphxai.utils import Explanation, node_mask_from_edge_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

class PGExplainerAttack(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG, and GraphXAI
    """
    
    # =============================================================================
    # Step1. Initialise Parameters and Explantion Model
    # =============================================================================
    def __init__(self, 
                 model: nn.Module, 
                 explanation_model: nn.Module,
                 emb_layer_name: str = None,
                 explain_graph: bool = False,
                 coeff_ent: float = 5e-4,
                 lr: float = 0.003, 
                 max_epochs: int = 20, 
                 eps: float = 1e-3,
                 num_hops: int = None, 
                 in_channels = None):
        """
        Args:
            ---model (torch.nn.Module): model on which to make predictions
                    The output of the model should be unnormalized class score.
                    For example, last layer = CNConv or Linear.
            ---explanation_model (torch.nn.Module): explanation model tp attack
            ---emb_layer_name (str, optional): name of the embedding layer
                    If not specified, use the last but one layer by default.
            ---explain_graph (bool): whether the explanation is graph-level (True) or node-level (False)
            --- coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            ---lr (float): learning rate to train the explanation network
            ---max_epochs (int): number of epochs to train the explanation network
            ---num_hops (int): number of hops to consider for node-level explanation
        """
        
        super().__init__(model, emb_layer_name)
        self.explanation_model = explanation_model
        
        # =============================================================================
        # Step1.1 Initialise Parameters for PGExplainer
        # =============================================================================
        self.explain_graph = explain_graph
        self.coeff_ent = coeff_ent
        self.lr = lr
        self.eps = eps
        self.max_epochs = max_epochs
        self.num_hops = self.L if num_hops is None else num_hops


        # =============================================================================
        # Step1.2 Explanation model in PGExplainer
        # =============================================================================

        ##Define how many layers to construt in GNN or MLP
        mult = 2 # if self.explain_graph else 3

        if in_channels is None:
            if isinstance(self.emb_layer, GCNConv):
                in_channels = mult * self.emb_layer.out_channels
            elif isinstance(self.emb_layer, GINConv):
                in_channels = mult * self.emb_layer.nn.out_features
            elif isinstance(self.emb_layer, torch.nn.Linear):
                in_channels = mult * self.emb_layer.out_features
            else:
                fmt_string = 'PGExplainer not implemented for embedding layer of type {}, please provide in_channels directly.'
                raise NotImplementedError(fmt_string.format(type(self.emb_layer)))
                

    # =============================================================================
    # Define a function to get the GNN predictor's prediction
    # =============================================================================
    def DIY_predict(self, x: torch.Tensor, 
                    edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
                    training: bool = False,
                    return_type: str = 'label', 
                    forward_kwargs: dict = {}):
        """
        Get the model's prediction.

        Args:
            x (torch.Tensor, [n x d]): node features
            edge_index (torch.Tensor, [2 x m]): edge index of the graph
            return_type (str): one of ['label', 'prob', 'log_prob']
            forward_kwargs (dict, optional): additional arguments to model.forward
                beyond x and edge_index

        Returns:
            pred (torch.Tensor, [n x ...]): model prediction
        """
        # Compute unnormalized class score
        import torch.nn.functional as F
        
        with torch.set_grad_enabled(training):
            out = self.model.to(device)(x, edge_index, edge_weight, **forward_kwargs)
            if return_type == 'label':
                out = out.argmax(dim=-1)
            elif return_type == 'prob':
                out = F.softmax(out, dim=-1)
            elif return_type == 'log_prob':
                out = F.log_softmax(out, dim=-1)
            else:
                raise ValueError("return_type must be 'label', 'prob', or 'log_prob'")

            if self.explain_graph:
                out = out.squeeze()

            return out
    # =============================================================================
    # Define a function to get the embedding: 1
    # =============================================================================
    def DIY_get_activation(self, layer: nn.Module, 
                           x: torch.Tensor,
                           edge_index: torch.Tensor, 
                           edge_weight: torch.Tensor,
                           training: bool = False,
                           forward_kwargs: dict = {}):
        """
        Get the activation of the layer.
        """
        activation = {}
        def get_activation():
            def hook(model, inp, out):
                activation['layer'] = out.detach()
            return hook

        layer.register_forward_hook(get_activation())

        with torch.set_grad_enabled(training):
            _ = self.model(x, edge_index, edge_weight, **forward_kwargs)

        return activation['layer']
    
    # =============================================================================
    # Define a function to get the embedding: 2
    # =============================================================================
    def DIY_get_embedding(self, x: torch.Tensor, 
                          edge_index: torch.Tensor,
                          edge_weight: torch.Tensor,
                          training: bool = False,
                          forward_kwargs: dict = {}):
        """
        Get the embedding.
        """
        emb = self.DIY_get_activation(self.emb_layer, x, edge_index, edge_weight, training, forward_kwargs)
        return emb

    
    # =============================================================================
    # Step3. (tomodify) given a n-hop subgraph, we should generate the perturbed n-hop subgraph
    #                   the perturbed n-hop subgraph can be used for prediction
    # =============================================================================
    def get_perturbed_full_graph(self, 
                                  x: torch.Tensor, 
                                  edge_index: torch.Tensor,
                                  edge_index_in_whole_hop: torch.Tensor,
                                  edge_flipping_matrix: torch.Tensor):
        """
        The get_perturbed_full_graph function is designed to generate a perturbed graph

        Parameters
        ----------
        x : torch.Tensor [n x d]
              the node features of original graph.
        edge_index : torch.Tensor [2 x m]
            edge indices of original n-hop graph.
        edge_index_in_whole_hop : torch.Tensor [2 x (n*n-n)]
            all possible edge indices of whole graph except the diagonal.
        edge_flipping_matrix : torch.Tensor [2 x m]
            edge indices to flip.
        Returns
        -------
        x : torch.Tensor
            the node features of perturbed graph.
        edge_index_perturbed : torch.Tensor
            edge indices of perturbed graph.
        edge_weights_perturbed : torch.Tensor
            edge indices of perturbed graph.
        """
        
        x_perturbed = x
                
        num_nodes= x_perturbed.size(0)
        
        #Original graph: A
        original_adj_mat = torch.zeros(num_nodes, num_nodes)
        
        for i in range(edge_index.size(1)):
            source, target = edge_index[:, i]
            original_adj_mat[source, target] = 1
            original_adj_mat[target, source] = 1
        
        #Learnable perturbation matrix: P
        flipping_adj_mat = edge_flipping_matrix
            
        # All-ones matrix: 1
        all_one_mat = torch.ones(num_nodes, num_nodes)
                                
        ## Perturbed whole: A+P(1-2A): pay attention to the diagnoal 
        edge_weights_perturbed = original_adj_mat + flipping_adj_mat*(all_one_mat-2*original_adj_mat)
                
        ##only keep non-diagnoal elements
        ##Zhong: question? is this correct?
        edge_weights_perturbed = edge_weights_perturbed[~torch.eye(num_nodes).bool()].flatten()
                
        ##consider all possible edges in whole graph
        edge_index_perturbed = edge_index_in_whole_hop
        
        
        return x_perturbed, edge_index_perturbed, edge_weights_perturbed
            
    

    # =============================================================================
    # Step5. This function trains the GNN explaination function
    # =============================================================================

    def node_explanation_attack_model(self, 
                                      dataset: Data, 
                                      node_idx_to_attack: int,
                                      forward_kwargs: dict = {}):
        """
        Train the explanation attack model.
        """
        

        # ----------------------------------------------------------------
        # Step5.1. Define the loss function based on cross-entropy
        # ----------------------------------------------------------------
        def loss_fn1(prob: torch.Tensor,
                    ori_pred: int,
                    perturbation_matrix):
            
            ## Term 1: Maximize the probability of predicting the label (cross entropy)
            loss = -torch.log(prob[ori_pred] + 1e-6)
            
            ## Term 2: element-wise entropy regularization, where a low entropy implies the edge flipping matrix is close to binary
            perturbation_matrix = perturbation_matrix * 0.99 + 0.005 ##why this?
            entropy = - perturbation_matrix * torch.log(perturbation_matrix) - (1 - perturbation_matrix) * torch.log(1 - perturbation_matrix)
            loss += self.coeff_ent * torch.mean(entropy) ## coeff_ent is the trade-off HP

            return loss

        
        # ----------------------------------------------------------------
        # Step5.2. Define the loss function based on cos sim
        # ----------------------------------------------------------------      
        def loss_fn2_cos(node_nums, 
                         edge_weights_original, 
                         edge_weights_perturbed,
                         edge_index_original, 
                         edge_index_perturbed,
                         if_vis = False,
                         node_idx = None,
                         epoch = None):
     
            # Create sparse adjacency matrices
            adj_matrix_original = torch.sparse_coo_tensor(edge_index_original, edge_weights_original, (node_nums, node_nums))
            adj_matrix_perturbed = torch.sparse_coo_tensor(edge_index_perturbed, edge_weights_perturbed, (node_nums, node_nums))
        
            # Convert to dense for subtraction and ensure same size
            adj_matrix_original_dense = adj_matrix_original.to_dense()
            adj_matrix_perturbed_dense = adj_matrix_perturbed.to_dense()
            
            
            # Flatten the dense matrices to compute cosine similarity as if they are vectors
            flat_adj_original = adj_matrix_original_dense.flatten()
            flat_adj_perturbed = adj_matrix_perturbed_dense.flatten()
    
            # Compute the difference, square it, and then take the mean for MSE loss
            import torch.nn.functional as F
            cos_sim = F.cosine_similarity(flat_adj_original.unsqueeze(0), flat_adj_perturbed.unsqueeze(0))
            cos_loss = 1-cos_sim
            
            return cos_loss
        

        # ----------------------------------------------------------------
        # Step5.2. Attack explanations of node-level predictions
        # ----------------------------------------------------------------
        if not self.explain_graph:  
            """
            If self.explain_graph is False, the function focuses on explaining node-level predictions within a single graph. 
            It identifies the nodes for which explanations are required (typically those in the training set) 
            and pre-computes some necessary information for each, such as the relevant subgraph around each node and 
            the corresponding embeddings.
            """
            data = dataset.to(device)
            X = data.x
            EIDX = data.edge_index
            EATTR = torch.ones(data.edge_index.size(1), dtype=torch.float)   
            EMBS  = self.DIY_get_embedding(X, EIDX, EATTR, training= False, forward_kwargs=forward_kwargs)
            
            # ----------------------------------------------------------------
            # Step5.2.1: Get the predicted label of node on original graph
            # (To do): we can move this outside the attack
            # ----------------------------------------------------------------
            with torch.set_grad_enabled(True):
                
                ## indicate no gradient computation for GNN predictor 
                self.model.eval()
                
                ## get the list of nodes to predict
                explain_node_index_list = [node_idx_to_attack]
                                
                ##get the labels of these nodes to use cross entropy
                label = self.DIY_predict(X, EIDX, EATTR, training=True, forward_kwargs=forward_kwargs)
                
                ##get the probs of these nodes to use in advanced cross entropy
                probs = self.DIY_predict(X, EIDX, EATTR, training=True, forward_kwargs=forward_kwargs, return_type='prob')
                                
                ##store the pairs of nodes and labels in a dict
                original_pred_label_dict = dict(zip(explain_node_index_list, label[explain_node_index_list]))
                
                ##store the pairs of nodes and probs in a dict
                original_pred_prob_dict = dict(zip(explain_node_index_list, probs[explain_node_index_list]))
                
                
            # ----------------------------------------------------------------
            # Step5.2.2: Get the explanation edge mask of node on original graph
            # (To do): we can move this outside the attack
            # ----------------------------------------------------------------
            with torch.set_grad_enabled(True):
                
                ## indicate no gradient computation: do not needed 
                #self.explanation_model.eval()
                
                ## get the list of nodes to explain
                explain_node_index_list = [node_idx_to_attack]
                                
                ##a list to store explanation edge masks of these nodes
                explanation_list = []

                for explanation_node_idx in explain_node_index_list:
                    
                    ##get the explanation edge masks of these nodes
                    _, explanation_edge_mask = self.explanation_model.get_explanation_node(dataset = data,
                                                                                           node_idx = explanation_node_idx, 
                                                                                           x = X, 
                                                                                           edge_index = EIDX,
                                                                                           edge_weights = EATTR,
                                                                                           training = True)
                    
                    explanation_list.append(explanation_edge_mask)
                    
                    ##this function is defined in GraphXAI: _BaseExplainer
                    self._clear_masks()
                
                ##store the pairs of nodes and explanations in a dict
                original_explanation_dict = dict(zip(explain_node_index_list, explanation_list))              
                              

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Perform attacks
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        

            # ----------------------------------------------------------------
            # Step5.2.3: Initialize perturbation matrix to zeros
            # ----------------------------------------------------------------
            
            num_nodes = data.x.size(0)
            all_zero_mat = torch.zeros(num_nodes, num_nodes)
            
            perturbation_matrix = torch.tensor(all_zero_mat, requires_grad=True) ##Ensure Gradient Tracking
                   
            duration = 0.0
            loss = 0.0
            tic = time.perf_counter()
            
            # ----------------------------------------------------------------
            # Prepare repeatedly used data
            # ----------------------------------------------------------------
            
            def adjacency_matrix_to_edge_index(n):
                
                # Step 1: Create the adjacency matrix
                adj_matrix = np.ones((n, n)) - np.eye(n)
                
                # Step 2: Convert the adjacency matrix to edge indices
                # Get the indices where the adjacency matrix is 1
                edge_indices = np.where(adj_matrix == 1)
                
                # Convert to tensor
                edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long)
                
                return edge_index_tensor
            
        
            x_full_original = X
            edge_index_full_original = EIDX
            edge_index_in_whole_graph = adjacency_matrix_to_edge_index(num_nodes) #Consider all possible edges (existing and non-existing) in full graph
            
            vis_num = 0
            
            for epoch in range(self.max_epochs):
                                
                ##Zero the gradients to  avoid accumulating them over epochs
                if perturbation_matrix.grad is not None:
                    perturbation_matrix.grad.zero_()
                
                # ----------------------------------------------------------------
                # Step5.2.4: Perform perturbations on adj matrix
                # ----------------------------------------------------------------                           
                ## Use pre-defined get_perturbed_full_graph() function 
                x_full_perturbed, edge_index_full_perturbed, edge_weights_full_perturbed = \
                    self.get_perturbed_full_graph(x_full_original, 
                                                  edge_index_full_original, 
                                                  edge_index_in_whole_graph, 
                                                  perturbation_matrix) 


                # ----------------------------------------------------------------
                # Step5.2.5: Get the predicted label of node on perturbed graph
                # ----------------------------------------------------------------
                ## Use pre-defined DIY_predict() function
                perturbed_prediction_prob = self.DIY_predict(x_full_perturbed, 
                                                             edge_index_full_perturbed, 
                                                             edge_weights_full_perturbed,
                                                             training = True,
                                                             forward_kwargs=forward_kwargs, 
                                                             return_type='prob')   
    
    
                # ----------------------------------------------------------------
                # Step5.2.6: Get the explanation edge mask of node on perturbed graph
                # ----------------------------------------------------------------
                ## Use existing explanation_model.get_explanation_node() function
                _, perturbed_explanation_edge_mask = self.explanation_model.get_explanation_node(dataset = data,
                                                                                                 node_idx = node_idx_to_attack,
                                                                                                 x = x_full_perturbed,
                                                                                                 edge_index = edge_index_full_perturbed,
                                                                                                 edge_weights = edge_weights_full_perturbed,
                                                                                                 training = True)    
                
                ##this function is defined in GraphXAI: _BaseExplainer
                self._clear_masks()
                # ----------------------------------------------------------------
                # Step5.2.7: Compute loss function values
                # ----------------------------------------------------------------               
                
                ##----------------------------------------------
                ##First loss term: keep the prediction unchanged.
                ##-----------------------------------------------
                # print(perturbed_prediction_prob[node_idx_to_attack])
                # print(original_pred_label_dict[node_idx_to_attack])
                # print(original_pred_prob_dict[node_idx_to_attack])
                original_max_prob = torch.max(original_pred_prob_dict[node_idx_to_attack]).item()
                # print(original_max_prob)
                
                loss_tmp1 = loss_fn1(perturbed_prediction_prob[node_idx_to_attack], 
                                      original_pred_label_dict[node_idx_to_attack],
                                      perturbation_matrix)
                
                
                ##----------------------------------------------
                ##Secon loss term: change the explanations
                ##----------------------------------------------
                
                
                loss_tmp2 = loss_fn2_cos(node_nums = X.size(0), 
                                         edge_weights_original = original_explanation_dict[node_idx_to_attack], 
                                         edge_weights_perturbed = perturbed_explanation_edge_mask,
                                         edge_index_original = edge_index_full_original, 
                                         edge_index_perturbed=  edge_index_full_perturbed
                                         )
                
                
                ##----------------------------------------------
                ##Combine loss terms
                ##----------------------------------------------                
                beta = 0.1
                loss_tmp = -loss_tmp1 + beta*loss_tmp2
            
                # ----------------------------------------------------------------
                # Step5.2.8: Use gradients to update perturbation_matrix
                # ----------------------------------------------------------------              
                
                ##Compute gradients
                loss_tmp.backward(retain_graph=True)
        
                
                # Perform an in-place update 
                ## Use add_() so that we can still track the gradient after updating
                ## Use ".data" to perform the update so as to not interfere with the gradient computation graph
                alpha_lr = 0.01
                perturbation_matrix.data.add_(alpha_lr * perturbation_matrix.grad.data)
                
                # ----------------------------------------------------------------
                # Step5.2.9: Perform projection using code from
                # ##https://github.com/pyg-team/pytorch_geometric/blob/82aad03c0c2d4afbf5bd4622378ab060bfe91b3b/torch_geometric/contrib/nn/models/rbcd_attack.py#L18
                # ----------------------------------------------------------------                     
                
                def _bisection(edge_weights: torch.Tensor, 
                               a: float, 
                               b: float, 
                               n_pert: int,
                               eps=1e-5,
                               max_iter=1e3) -> torch.Tensor:
                    """Bisection search for projection."""
                    def shift(offset: float):
                        return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)
            
                    miu = a
                    for _ in range(int(max_iter)):
                        miu = (a + b) / 2
                        # Check if middle point is root
                        if (shift(miu) == 0.0):
                            break
                        # Decide the side to repeat the steps
                        if (shift(miu) * shift(a) < 0):
                            b = miu
                        else:
                            a = miu
                        if ((b - a) <= eps):
                            break
                    return miu
                
                def _project(budget: int, 
                             values,
                             eps: float = 1e-7) -> torch.Tensor:
                    r"""Project :obj:`values`:
                    :math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`.
                    """
                    if torch.clamp(values, 0, 1).sum() > budget:
                        left = (values - 1).min()
                        right = values.max()
                        miu = _bisection(values, left, right, budget)
                        values = values - miu
                    return torch.clamp(values, min=eps, max=1 - eps)
                
                B=15
                perturbation_matrix.data = _project(B, 
                                                    perturbation_matrix.data,
                                                    1e-7)
                
                # ----------------------------------------------------------------
                # ----------------------------------------------------------------
                loss += loss_tmp.item() ##this was commented by GraphXAI
                
                duration += time.perf_counter() - tic
                
                # print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')
                
                
            # ----------------------------------------------------------------
            # Step5.2.10: Generate binary perturbation matrix: symmetric and less than budget B
            # ----------------------------------------------------------------  
            
            def generate_symmetric_binary(perturbation_matrix, B):
                
                n = perturbation_matrix.size(0)
                max_attempts = 2000  # Set a limit to prevent infinite loops
                
                for _ in range(max_attempts):
                    # Sample the upper triangular part, including the diagonal
                    upper_tri = torch.triu(torch.bernoulli(perturbation_matrix), diagonal=0)
            
                    # Make the matrix symmetric
                    binary_perturbation_matrix = upper_tri + upper_tri.transpose(0, 1) - torch.diag(upper_tri.diagonal())
            
                    # Check if the sum satisfies the condition
                    if binary_perturbation_matrix.sum() < 2 * B and binary_perturbation_matrix.sum() > 10 :
                        return binary_perturbation_matrix
            
                # If we reach here, we didn't find a matrix that satisfies the condition in the given attempts
                print(f"Failed to generate a binary matrix satisfying the 5< sum < 2B after {max_attempts} attempts.")
                return binary_perturbation_matrix
            
            print("perturbation_matrix budget:" + str(perturbation_matrix.sum().tolist()))     
            print("adjusted perturbation_matrix budget:" + str(perturbation_matrix.sum().tolist()))
                
            binary_perturbation_matrix = generate_symmetric_binary(perturbation_matrix, B)
            
            
            used_budget = binary_perturbation_matrix.sum().tolist()
            
            print("binary_perturbation_matrix budget:" + str(used_budget))
            
            print(f"attack time is {duration:.5}s")

            # ----------------------------------------------------------------
            # Step5.2.11: Return perturbed graph: edge_index, edge_weights
            # ----------------------------------------------------------------  
            
            _, edge_index_res, edge_weights_res = \
                self.get_perturbed_full_graph(x_full_original, 
                                              edge_index_full_original, 
                                              edge_index_in_whole_graph, 
                                              binary_perturbation_matrix) 
                    
            
            return edge_index_res, edge_weights_res, used_budget
    
    
    
