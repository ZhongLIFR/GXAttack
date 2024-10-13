#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:25:10 2024

@author: Anonymous

Implemented/repurposed based on GraphXAI: https://github.com/mims-harvard/GraphXAI
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


def k_hop_subgraph_with_default_whole_graph(edge_index, 
                                            node_idx=None, 
                                            num_hops=3, 
                                            relabel_nodes=False,
                                            num_nodes=None,
                                            flow='source_to_target'):
    
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node :attr:`node_idx`.
    Args:
        ---edge_index (LongTensor): The edge indices.
        ---node_idx (int, list, tuple or :obj:`torch.Tensor`): The central node(s).
        ---num_hops: (int): The number of hops :math:`k`.
        ---relabel_nodes (bool, optional): If set to :obj:`True`, 
                the resulting :obj:`edge_index` will be relabeled to hold consecutive indices starting from zero. 
                (default: :obj:`False`)
        --- num_nodes (int, optional): The number of nodes, *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
               (default: :obj:`None`)
        ---flow (string, optional): The flow direction of :math:`k`-hop aggregation (:obj:`"source_to_target"` or
               :obj:`"target_to_source"`). 
              (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,:class:`BoolTensor`)

    It returns 
        (1) the nodes involved in the subgraph, 
        (2) the filtered :obj:`edge_index` connectivity, 
        (3) the mapping from node indices in :obj:`node_idx` to their new location, 
        (4) the edge mask indicating which edges were preserved.
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx], device=row.device)
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


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

class PGExplainer(_BaseExplainer):
    """
    PGExplainer

    Code adapted from DIG, and GraphXAI
    """
    print("\n $$$$$$$This is our DIY PGExplainer$$$$$$$ \n")
    
    # =============================================================================
    # Step1. Initialise Parameters and Explantion Model
    # =============================================================================
    def __init__(self, 
                 model: nn.Module, 
                 emb_layer_name: str = None,
                 explain_graph: bool = False,
                 coeff_size: float = 0.01, 
                 coeff_ent: float = 5e-4,
                 t0: float = 5.0, 
                 t1: float = 2.0,
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
            ---emb_layer_name (str, optional): name of the embedding layer
                    If not specified, use the last but one layer by default.
            ---explain_graph (bool): whether the explanation is graph-level (True) or node-level (False)
            ---coeff_size (float): size regularization to constrain the explanation size
            --- coeff_ent (float): entropy regularization to constrain the connectivity of explanation
            ---t0 (float): the temperature at the first epoch
            ---t1 (float): the temperature at the final epoch
            ---lr (float): learning rate to train the explanation network
            ---max_epochs (int): number of epochs to train the explanation network
            ---num_hops (int): number of hops to consider for node-level explanation
        """
        
        super().__init__(model, emb_layer_name)

        # =============================================================================
        # Step1.1 Initialise Parameters for PGExplainer
        # =============================================================================
        self.explain_graph = explain_graph
        self.coeff_size = coeff_size
        self.coeff_ent = coeff_ent
        self.t0 = t0
        self.t1 = t1
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
                
        ##Define the layers in the model: Linear + ReLU + Linear
        self.elayers = nn.ModuleList([nn.Sequential(nn.Linear(in_channels, 64),nn.ReLU()),nn.Linear(64, 1)]).to(device)

    # =============================================================================
    # Define a function to get the GNN predictor's prediction
    # =============================================================================
    def DIY_predict(self, x: torch.Tensor, 
                    edge_index: torch.Tensor,
                    edge_weight: torch.Tensor,
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
        
        with torch.no_grad():
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

        with torch.no_grad():
            _ = self.model(x, edge_index, edge_weight, **forward_kwargs)

        return activation['layer']
    
    # =============================================================================
    # Define a function to get the embedding: 2
    # =============================================================================
    def DIY_get_embedding(self, x: torch.Tensor, 
                          edge_index: torch.Tensor,
                          edge_weight: torch.Tensor,
                          forward_kwargs: dict = {}):
        """
        Get the embedding.
        """
        emb = self.DIY_get_activation(self.emb_layer, x, edge_index, edge_weight, forward_kwargs)
        return emb
    
    # =============================================================================
    # Step2. This function is used to sample from a Concrete (or Gumbel-Softmax) distribution
    # =============================================================================
    def __concrete_sample(self, 
                          log_alpha: torch.Tensor,
                          beta: float = 1.0, 
                          training: bool = True):
        """
        Sample from the instantiation of concrete distribution when training (Gumel-Softmax reparameterization trick):
    
        1. If training is True, the function introduces randomness into the sampling process to facilitate exploration 
          and gradient backpropagation. It generates random noise with the same shape as log_alpha, using a logistic 
          distribution (derived from uniform random noise). This noise is added to log_alpha and scaled by beta. 
          The sigmoid function is then applied to the result, yielding a sample from the Concrete distribution.
          
        2. If training is False, indicating inference or evaluation mode, the function bypasses the addition of noise 
          and simply applies the sigmoid function to log_alpha. This deterministic step is often used during evaluation
          to pick the most probable discrete choice directly.
        
        Args:
            ---mlog_alpha: it represents the log-odds of the probabilities associated with the discrete variables
            ---beta: the temperature parameter that controls the "sharpness" of the distribution.
                     A lower value makes the distribution closer to discrete (sharper), 
                     and a higher value makes it more continuous (softer)
            ---training: A boolean indicating whether the model is in training mode

        Returns:
            --- sigmoid((log_alpha + noise) / beta) if training == True
            --- sigmoid(log_alpha) if training == False
            
            More concretely, it returns a tensor of the same shape as log_alpha, with values between 0 and 1 representing 
            "soft" discrete choices. During training, these choices are "softened" by the noise and temperature to 
            allow gradient-based optimization. During inference, the choices are more deterministic.
        """
        if training:
            random_noise = torch.rand(log_alpha.shape).to(device)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    # =============================================================================
    # Step3. This function generates the edge mask based on node embeddings 
    # =============================================================================
    def __emb_to_edge_mask(self, 
                           emb: torch.Tensor,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           node_idx: int = None,
                           forward_kwargs: dict = {},
                           tmp: float = 1.0, 
                           training: bool = False):
        """
        Compute the edge mask based on embedding.

        Returns:
            prob_with_mask (torch.Tensor): the predicted probability with edge_mask
            edge_mask (torch.Tensor): the mask for graph edges with values in [0, 1]
        """

        with torch.set_grad_enabled(training):
            
            # ----------------------------------------------------------------
            # Step3.1. Concat relevant node embeddings
            # ----------------------------------------------------------------
            U, V = edge_index  # edge (u, v), U = (u), V = (v)
            h1 = emb[U]
            h2 = emb[V]
            
            if self.explain_graph: 
                ##graph-level explaination
                h = torch.cat([h1, h2], dim=1)
            else:                  
                ##node-level explaination
                h3 = emb.repeat(h1.shape[0], 1) ##this line is never used.
                h = torch.cat([h1, h2], dim=1)

            # ----------------------------------------------------------------
            # Step3.2. Calculate the edge weights and generate the edge mask
            # ----------------------------------------------------------------
            
            ##make the embedding iterate through a series of layers
            for elayer in self.elayers:
                h = elayer.to(device)(h)
                
            h = h.squeeze()
            
            ## Sample from a Concrete (or Gumbel-Softmax) distribution based on embedding h
            edge_weights = self.__concrete_sample(h, tmp, training)
            
            ## Generate a sparse edge mask then
            n = emb.shape[0]  # number of nodes
            mask_sparse = torch.sparse_coo_tensor(edge_index, edge_weights, (n, n))
            
            # Convert sparse mask to dense mask matrix (Note that this is not scalable)
            self.mask_sigmoid = mask_sparse.to_dense()
            
            # Convert the mask matrix to synmetric matrix so that it generates undirected graph
            sym_mask = (self.mask_sigmoid + self.mask_sigmoid.transpose(0, 1)) / 2
            edge_mask = sym_mask[edge_index[0], edge_index[1]] ##Apply it on U and V
            
            # print(edge_mask)
            
            # Apply the computed edge_mask to the graph defined by x (node features) and edge_index
            self._set_masks(x, edge_index, edge_mask) ##this function is defined in GraphXAI: _BaseExplainer
            
            edge_weights = torch.ones(edge_index.size(1), dtype=torch.float)

        
        # ----------------------------------------------------------------
        # Step3.3. Compute the predicted probability of a graph after applying the so-generated edge_mask on it
        # ----------------------------------------------------------------
                
        prob_with_mask = self.DIY_predict(x, edge_index, edge_weights, forward_kwargs=forward_kwargs, return_type='prob') ##this function is defined in GraphXAI: _BaseExplainer
        
        self._clear_masks() ##this function is defined in GraphXAI: _BaseExplainer

        return prob_with_mask, edge_mask


    # =============================================================================
    # Step4. This function trains the GNN explaination function
    # =============================================================================

    def train_explanation_model(self, 
                                dataset: Data, 
                                forward_kwargs: dict = {}):
        """
        Train the explanation model.
        """
        
        ##Specifiy the optimizer
        optimizer = torch.optim.Adam(self.elayers.parameters(), lr=self.lr)

        # ----------------------------------------------------------------
        # Step4.1. Define the loss function
        # ----------------------------------------------------------------
        def loss_fn(prob: torch.Tensor, ori_pred: int):
            
            ## Term 1: Maximize the probability of predicting the label (cross entropy)
            loss = -torch.log(prob[ori_pred] + 1e-6)
            
            ## Term 2: size regularization for edge mask
            edge_mask = self.mask_sigmoid
            loss += self.coeff_size * torch.sum(edge_mask) ##less change is perferred, coeff_size is the trade-off HP
            
            ## Term 3: element-wise entropy regularization, where a low entropy implies the mask is close to binary
            edge_mask = edge_mask * 0.99 + 0.005 ##why this?
            entropy = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)
            loss += self.coeff_ent * torch.mean(entropy) ## coeff_ent is the trade-off HP

            return loss
        
        # ----------------------------------------------------------------
        # Step4.2. Explain node-level predictions of a graph
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
            
            # Generate meaningful weights for original graph
            EATTR = torch.ones(data.edge_index.size(1), dtype=torch.float)


            # ----------------------------------------------------------------
            # Step4.2.1: Get the predicted labels for training nodes
            # ----------------------------------------------------------------
            with torch.no_grad():
                
                ## indicate no gradient computation 
                self.model.eval()
                
                ## get the list of nodes to train
                explain_node_index_list = torch.where(data.train_mask)[0].tolist()
                
                ##get the labels of these nodes
                label = self.DIY_predict(X, EIDX, EATTR, forward_kwargs=forward_kwargs)
                
                ##store the pairs of nodes and labels in a dict
                pred_dict = dict(zip(explain_node_index_list, label[explain_node_index_list]))


            # ----------------------------------------------------------------
            # Step4.2.2: Train the mask generator
            # ----------------------------------------------------------------
            """
            the function then trains the explanation model to generate edge masks that effectively explain the predictions 
            for each node. The training loop updates the model based on the computed loss, which again includes terms 
            for prediction accuracy, mask size, and mask entropy.
            """
            duration = 0.0
            last_loss = 0.0
            x_dict = {}
            edge_index_dict = {}
            node_idx_dict = {}
            emb_dict = {}
            
            ##iterate over the nodes to explain and obatin the necessary information
            for iter_idx, node_idx in tqdm.tqdm(enumerate(explain_node_index_list)):
                
                ##get the corresponding subgraph
                subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, self.num_hops, EIDX, relabel_nodes=True, num_nodes=data.x.shape[0])
                
                ##get the corresponding node features
                x_dict[node_idx] = X[subset].to(device)
                
                ##get the corresponding edges
                edge_index_dict[node_idx] = sub_edge_index.to(device)
                
                # Generate meaningful weights for original graph
                sub_edge_weights = torch.ones(sub_edge_index.size(1), dtype=torch.float)
                
                ##get the corresponding node embeddings
                emb = self.DIY_get_embedding(X[subset], sub_edge_index, sub_edge_weights, forward_kwargs=forward_kwargs)
                emb_dict[node_idx] = emb.to(device)
                
                node_idx_dict[node_idx] = int(torch.where(subset==node_idx)[0])

            for epoch in range(self.max_epochs):
                
                ##initialize values and model
                loss = 0.0
                optimizer.zero_grad()
                
                ##temperature parameter that is adjusted at each epoch to control the sampling process involved 
                ##  in generating the edge masks. This parameter affects the softness of the decision boundaries 
                ##  in the mask, allowing for more gradual transitions between included and excluded edges in the early 
                ##  stages of training.
                tmp = float(self.t0 * np.power(self.t1/self.t0, epoch/self.max_epochs))
                
                self.elayers.train()
                tic = time.perf_counter()

                ##iterate over the nodes to explain and obtain their prediction to train the explainer
                for iter_idx, node_idx in tqdm.tqdm(enumerate(x_dict.keys())):
                    
                    ##get the prediction probabilities after masking the original graph
                    prob_with_mask, _ = self.__emb_to_edge_mask(emb_dict[node_idx], 
                                                                x = x_dict[node_idx], 
                                                                edge_index = edge_index_dict[node_idx], 
                                                                node_idx = node_idx,
                                                                forward_kwargs=forward_kwargs,
                                                                tmp=tmp, 
                                                                training=True)
                    ##compute loss function values
                    loss_tmp = loss_fn(prob_with_mask[node_idx_dict[node_idx]], pred_dict[node_idx])
                    loss_tmp.backward()
                    
                    loss += loss_tmp.item() ##this was commented by GraphXAI

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss/len(explain_node_index_list)}')

            print(f"training time is {duration:.5}s")


    # =============================================================================
    # Step5. This function explain a node prediction in the inference stage
    # =============================================================================
    
    def get_explanation_node(self, 
                             dataset: Data,
                             node_idx: int, 
                             x: torch.Tensor,
                             edge_index: torch.Tensor,
                             edge_weights: torch.Tensor, 
                             training: bool = False,
                             label: torch.Tensor = None,
                             y = None,
                             forward_kwargs: dict = {}, **_):
        """
        Explain a node prediction.

        Args:
            node_idx (int): 
                index of the node to be explained
            x (torch.Tensor, [n x d]): 
                node features
            edge_index (torch.Tensor, [2 x m]): 
                edge index of the graph
            label (torch.Tensor, optional, [n x ...]): 
                labels to explain. If not provided, we use the output of the model.
            forward_kwargs (dict, optional): 
                additional arguments to model.forward beyond x and edge_index

        Returns:
            ---exp (dict):
                -exp['feature_imp'] (torch.Tensor, [d]): 
                    feature mask explanation
                -exp['edge_imp'] (torch.Tensor, [m]): 
                    k-hop edge importance
                -exp['node_imp'] (torch.Tensor, [n]): 
                    k-hop node importance
            ---khop_info (4-tuple of torch.Tensor):
                0. the nodes involved in the subgraph
                1. the filtered `edge_index`
                2. the mapping from node indices in `node_idx` to their new location
                3. the `edge_index` mask indicating which edges were preserved
        """
        if self.explain_graph:
            raise Exception('For graph-level explanations use `get_explanation_graph`.')
        
        ##get labels of nodes        
        label = self.DIY_predict(x, edge_index, edge_weights) if label is None else label

        ##get the k-hop information
        khop_info = _, _, _, sub_edge_mask = k_hop_subgraph(node_idx,
                                                            self.num_hops,
                                                            edge_index,
                                                            relabel_nodes=False,
                                                            num_nodes=x.shape[0])
        
        ##get the embeddings that will be used to compute edge_mask
        emb = self.DIY_get_embedding(x, 
                                     edge_index, 
                                     edge_weights,
                                     forward_kwargs=forward_kwargs)
        
        ##get the edge mask
        _, edge_mask = self.__emb_to_edge_mask(emb, 
                                               x, 
                                               edge_index, 
                                               node_idx, 
                                               forward_kwargs=forward_kwargs,
                                               tmp=2, 
                                               training=training)
        
        ##compute the edge importance
        edge_imp = edge_mask[sub_edge_mask]
                
        ##use pre-defined function Explanation() in graph.utils to generate explanations
        exp = Explanation(node_imp = node_mask_from_edge_mask(khop_info[0], khop_info[1], edge_imp.bool()),
                          edge_imp = edge_imp,
                          node_idx = node_idx,
                          graph = dataset)

        exp.set_enclosing_subgraph(khop_info)

        return exp, edge_mask
    
