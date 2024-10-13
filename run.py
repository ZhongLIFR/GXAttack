#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 10:30:44 2024

@author: Anonymous

This testbed is used to test the attack performances for GXAttack and other baselines

things to save:
    1. dataset
    2. attack_results_prob5
    3. attack_results_prob5
    4. attack_results_prob5
    5. attack_results_prob5
    6. attack_results_prob5
    7. analysis_results
"""


# =============================================================================
# Add this code block if there is a fault related to "_centered"
# =============================================================================
import numpy as np
import  scipy.signal.signaltools

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered

import graphxai


import torch
import matplotlib.pyplot as plt
from graphxai.datasets import ShapeGGen


# =============================================================================
# Step1. Load the dataset and run a quick plot of the dataset.
# =============================================================================

#dataset = ShapeGGen(
#    model_layers = 2,
#    shape = 'house',
#    num_subgraphs = 150,
#    subgraph_size = 10,
#    prob_connection = 0.03,
#    add_sensitive_feature = False
#)


#--------------------------------------------
# Save the dataset to a file
#--------------------------------------------


#save_path = './exp_results/dataset_Syn8.pth'

#torch.save(dataset, save_path)  # Save the dataset object

# --------------------------------------------
# Load the dataset from the file
# --------------------------------------------
import argparse

parser = argparse.ArgumentParser(description='Save a dataset with a specified filename from command line.')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset file to save.')
args = parser.parse_args()

save_path = f'./exp_results/{args.dataset_name}.pth'
dataset = torch.load(save_path)

# --------------------------------------------
# set random seed for reproducible results: not working for generating dataset
# --------------------------------------------
import torch
import random
import os
import numpy as np

seed_number = 2

## Set random seed
np.random.seed(seed_number)
torch.manual_seed(seed_number)
torch.cuda.manual_seed(seed_number)
torch.cuda.manual_seed_all(seed_number)
random.seed(seed_number)
os.environ['PYTHONHASHSEED'] = str(seed_number)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
# Step2:  Build a GNN predictor using GCNConv, which can accept edge weight
# =============================================================================    
from torch_geometric.nn import GCNConv


class MyGNN(torch.nn.Module):
    def __init__(self, 
                 input_feat, 
                 hidden_channels, 
                 classes=2):
        
        super(MyGNN, self).__init__()
        # Initialize the first GCN layer
        self.gcn1 = GCNConv(in_channels=input_feat, out_channels=hidden_channels)
        # Initialize the second GCN layer
        self.gcn2 = GCNConv(in_channels=hidden_channels, out_channels=classes)
            
    def forward(self, x, edge_index, edge_weight):
        # Pass the input through the first GCN layer with edge weights and apply a ReLU activation
        x = self.gcn1(x, edge_index, edge_weight)
        x = torch.relu(x)

        # Pass the result through the second GCN layer with edge weights
        x = self.gcn2(x, edge_index, edge_weight)
        return x


# =============================================================================
# Step3: Split the data and train the GNN predictor
# =============================================================================

from torch_geometric.data import Data
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

##define a function to train the GNN predictor
def predictor_train(model: torch.nn.Module, 
                    optimizer,
                    criterion, 
                    data):
    model.train()
    
    # Clear gradients.
    optimizer.zero_grad() 
    
    # Perform a single forward pass.
    out = model(data.x, data.edge_index, data.edge_attr)  
    
    # Compute the loss solely based on the training nodes.
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) 
    
    # Derive gradients.
    loss.backward()
    
    # Update parameters based on gradients.
    optimizer.step()  
    
    return loss
 
def predictor_test(model: torch.nn.Module, data, num_classes=2, get_auc=False):
    model.eval()
    
    # Perform a single forward pass.
    out = model(data.x, data.edge_index, data.edge_attr)
    
    # Use the class with the highest probability.
    pred = out.argmax(dim=1)
    
    # Get the predicted classes by finding the indices of the max logit for each example
    _, predicted_classes = out.max(dim=1)
    
    # Gather the probabilities of the predicted classes
    probabilities = out.softmax(dim=1)
    
    # Initialize sets for correctly and wrongly predicted nodes and for the specific probability ranges.
    correct_nodes, wrong_nodes = set(), set()
    proba_ranges = {f'range_{i}': set() for i in range(5, 10)}

    # Extract test indices to match predictions with original indices
    test_indices = data.test_mask.nonzero(as_tuple=False).squeeze()

    # Select the probabilities of the predicted classes for the test dataset
    probas_pred = probabilities[test_indices, predicted_classes[test_indices]].detach().numpy()
    
    # Get true labels for the test dataset using the test indices
    true_Y = data.y[test_indices].numpy()

    # Calculate accuracy.
    acc = accuracy_score(true_Y, pred[test_indices].numpy())

    # Fill in the sets based on the predictions and probability ranges for original indices
    for idx, (true_label, predicted_label, proba) in zip(test_indices, zip(true_Y, pred[test_indices].numpy(), probas_pred)):
        original_idx = idx.item()  # Get the original node index
        if true_label == predicted_label:
            correct_nodes.add(original_idx)
        else:
            wrong_nodes.add(original_idx)
        
        # Assign nodes to the specific probability range sets based on probability
        if 0.5 <= proba < 0.6:
            proba_ranges['range_5'].add(original_idx)
        elif 0.6 <= proba < 0.7:
            proba_ranges['range_6'].add(original_idx)
        elif 0.7 <= proba < 0.8:
            proba_ranges['range_7'].add(original_idx)
        elif 0.8 <= proba < 0.9:
            proba_ranges['range_8'].add(original_idx)
        elif 0.9 <= proba <= 1.0:
            proba_ranges['range_9'].add(original_idx)

    # Calculate metrics for binary classification if applicable.
    if num_classes == 2:
        test_score = f1_score(true_Y, pred[test_indices].numpy())
        precision = precision_score(true_Y, pred[test_indices].numpy())
        recall = recall_score(true_Y, pred[test_indices].numpy())

        # Calculate AUROC and AUPRC if required.
        if get_auc:
            auprc = metrics.average_precision_score(true_Y, probas_pred, pos_label=1)
            auroc = metrics.roc_auc_score(true_Y, probas_pred)

            return acc, test_score, precision, recall, auprc, auroc, correct_nodes, wrong_nodes, proba_ranges['range_5'], proba_ranges['range_6'], proba_ranges['range_7'], proba_ranges['range_8'], proba_ranges['range_9']
    
    return acc, test_score, correct_nodes, wrong_nodes, proba_ranges['range_5'], proba_ranges['range_6'], proba_ranges['range_7'], proba_ranges['range_8'], proba_ranges['range_9']


data = dataset.get_graph(use_fixed_split=True)

prediction_model = MyGNN(dataset.n_features, 32)
optimizer = torch.optim.Adam(prediction_model.parameters(), lr = 0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()

#--------------------------------------------
# Train the GNN classifer 
#--------------------------------------------

# Generate meaningful weights for original graph
edge_weights = torch.ones(data.edge_index.size(1), dtype=torch.float)

# Add edge weights to the Data object
data.edge_attr = edge_weights

for _ in range(300):
    loss = predictor_train(prediction_model, optimizer, criterion, data)

#--------------------------------------------
# Test the GNN classifer
#--------------------------------------------

##we use all nodes as test nodes here (normaly we should not do this)
data.test_mask =  torch.tensor([True] * data.x.shape[0])

##Zhong: modifications here, the order of acc and test_score
acc, f1, prec, rec, auprc, auroc, correct_nodes_set, wrong_nodes_set, prob5, prob6, prob7, prob8, prob9\
    = predictor_test(prediction_model, data, num_classes = 2, get_auc = True)

print('Test Accuracy: {:.4f}'.format(acc))
print('Test F1 score: {:.4f}'.format(f1))
print('Test AUROC: {:.4f}'.format(auroc))


# =============================================================================
# Step4: Train the GNN explainers: PGExplainer DIY by us
# =============================================================================


from torch_geometric.data import Data
from PGExplainer_WB import PGExplainer


# --------------------------------------------------------------
# Step4.1: Train PGExplainer
# --------------------------------------------------------------

# Embedding layer name is final GNN embedding layer in the model
explanation_model = PGExplainer(prediction_model, emb_layer_name = 'gcn2', max_epochs = 10, lr = 0.1)

# Required to first train PGExplainer on the dataset: feed the entire data
explanation_model.train_explanation_model(data)

# =============================================================================
# Step5: Attack the PGExplainer
# =============================================================================

from GXAttacker import PGExplainerAttack

# --------------------------------------------------------------
# Step5.1: Specify PGExplainerAttack
# --------------------------------------------------------------

# Embedding layer name is final GNN embedding layer in the prediction_model
attack_model = PGExplainerAttack(prediction_model, explanation_model, emb_layer_name = 'gcn2', max_epochs = 100, lr = 0.1)


# # =============================================================================
# # Step5.2: Test PGExplainerAttack and other baseline attackers
# # =============================================================================

from graphxai.metrics import graph_exp_acc
                 



# ==============
# All Groups
# ==============

group_cnt = 0

for group2attack in [prob5,prob6,prob7,prob8,prob9]:
    
    attack_results_prob = []

    for node_idx in group2attack:
        
        ground_truth_exp_subgraph = dataset.explanations[node_idx]
        
        print("\n node_idx to attack: " + str(node_idx))
        
        # --------------------------------------------------------------
        # Step5.2.1: Compute explanation accuracy change after perturbations
        # -------------------------------------------------------------- 
        
        # --------------------------------
        # Step5.2.1.1: before perturbation
        # --------------------------------
        
        # Get explanations from PGEx
        pgex_exp_subgraph_original, original_edge_mask = explanation_model.get_explanation_node(dataset = data,
                                                                                                node_idx = node_idx,
                                                                                                x = data.x,
                                                                                                edge_index = data.edge_index,
                                                                                                edge_weights = data.edge_attr)
        original_edge_mask_copy = original_edge_mask
        
        ##set the topk to 1 while the remaining to zeros
        topk_edges = torch.count_nonzero(ground_truth_exp_subgraph[0].edge_imp)
        topk_percent = int(0.25*data.edge_index.size(1))
        
        pgex_exp_subgraph_original.edge_imp = torch.where(torch.isin(torch.arange(len(pgex_exp_subgraph_original.edge_imp)), torch.topk(pgex_exp_subgraph_original.edge_imp, topk_edges).indices), torch.tensor(1.0), torch.tensor(0.0))
        original_edge_mask = torch.where(torch.isin(torch.arange(len(original_edge_mask)), torch.topk(original_edge_mask, topk_percent).indices), torch.tensor(1.0), torch.tensor(0.0))
        
        
        ## Accuracy of PGExplainer before perturbation
        pg_acc_original = graph_exp_acc(gt_exp = ground_truth_exp_subgraph[0], generated_exp = pgex_exp_subgraph_original)
        
        
        # --------------------------------
        # Step5.2.1.2:  after perturbation of GXAttacker
        # --------------------------------
        
        # Run PGExplainerAttack to perform attack
        perturbed_result = attack_model.node_explanation_attack_model(dataset = data, node_idx_to_attack = node_idx)
            
        used_budget = perturbed_result[2]
        
        #Create a mask for non-zero weights
        non_zero_mask = perturbed_result[1] != 0
        
        # Apply the mask to edge_index to filter out edges with zero weights
        perturbed_edge_index = perturbed_result[0][:, non_zero_mask]
        
        # Apply the same mask to edge_weights to retain corresponding non-zero weights
        perturbed_edge_weights = perturbed_result[1][non_zero_mask]
        
        
        from copy import deepcopy
        perturbed_data = deepcopy(data)
        perturbed_data.edge_index = perturbed_edge_index
        perturbed_data.edge_attr =  perturbed_edge_weights
        
        
        # # Get explanations from PGEx
        pgex_exp_subgraph_perturbed, perturbed_edge_mask = explanation_model.get_explanation_node(dataset = perturbed_data,
                                                                                                  node_idx = node_idx,
                                                                                                  x = data.x,
                                                                                                  edge_index = perturbed_edge_index,
                                                                                                  edge_weights = perturbed_edge_weights)
        
        
        perturbed_edge_mask_copy = perturbed_edge_mask
           
        
        ## Accuracy of PGExplainer after perturbation
        pg_acc_perturbed_GXA = graph_exp_acc(gt_exp = ground_truth_exp_subgraph[0], generated_exp = pgex_exp_subgraph_perturbed)
        
        
        print('Explanation Accuracy Change for GXAttacker:\n {:.4f} -> {:.4f}'.format(pg_acc_original,pg_acc_perturbed_GXA))
        
    
        # --------------------------------
        # Step5.2.1.3:  after perturbation of random_flipping
        # --------------------------------
        from Baseline_Attackers import RandomAttacks
        
        # Create an instance of the RandomAttacks class
        ra = RandomAttacks()
        
        perturbed_data_rf = ra.random_flipping(data = data, budget_val = 15)
        
    
        # Get explanations from PGEx
        pgex_exp_subgraph_rf, rf_edge_mask = explanation_model.get_explanation_node(dataset = perturbed_data_rf,
                                                                                    node_idx = node_idx,
                                                                                    x = perturbed_data_rf.x,
                                                                                    edge_index = perturbed_data_rf.edge_index,
                                                                                    edge_weights = perturbed_data_rf.edge_attr)    
        ##set the topk to 1 while the remaining to zeros
        topk_edges = torch.count_nonzero(pgex_exp_subgraph_rf.edge_imp)
        
        pgex_exp_subgraph_rf.edge_imp = torch.where(torch.isin(torch.arange(len(pgex_exp_subgraph_rf.edge_imp)), torch.topk(pgex_exp_subgraph_rf.edge_imp, topk_edges).indices), torch.tensor(1.0), torch.tensor(0.0))
        
        ## Accuracy of PGExplainer after random_flipping perturbation
        pg_acc_perturbed_rf = graph_exp_acc(gt_exp = ground_truth_exp_subgraph[0], generated_exp = pgex_exp_subgraph_rf)
        
        print('Explanation Accuracy Change for random_flipping:\n {:.4f} -> {:.4f}'.format(pg_acc_original, pg_acc_perturbed_rf))
        
       # --------------------------------
        # Step5.2.1.4:  after perturbation of random_rewiring
        # --------------------------------
            
        perturbed_data_rw = ra.random_rewiring(data = data, node_idx = node_idx, k=2, budget_val = 8)
        
        # Get explanations from PGEx
        pgex_exp_subgraph_rw, rw_edge_mask = explanation_model.get_explanation_node(dataset = perturbed_data_rw,
                                                                                    node_idx = node_idx,
                                                                                    x = perturbed_data_rw.x,
                                                                                    edge_index = perturbed_data_rw.edge_index,
                                                                                    edge_weights = perturbed_data_rw.edge_attr)    
        ##set the topk to 1 while the remaining to zeros
        topk_edges = torch.count_nonzero(pgex_exp_subgraph_rw.edge_imp)
        
        pgex_exp_subgraph_rw.edge_imp = torch.where(torch.isin(torch.arange(len(pgex_exp_subgraph_rw.edge_imp)), torch.topk(pgex_exp_subgraph_rw.edge_imp, topk_edges).indices), torch.tensor(1.0), torch.tensor(0.0))
        
        ## Accuracy of PGExplainer after random_flipping perturbation
        pg_acc_perturbed_rw = graph_exp_acc(gt_exp = ground_truth_exp_subgraph[0], generated_exp = pgex_exp_subgraph_rw)
        
        print('Explanation Accuracy Change for random_rewiring:\n {:.4f} -> {:.4f}'.format(pg_acc_original, pg_acc_perturbed_rw))
        
        # --------------------------------------------------------------
        # Step5.2.2: Compute prediction accuracy change after perturbations
        # --------------------------------------------------------------
        
        # --------------------------------
        # Step5.2.2.1: before perturbation
        # --------------------------------
        
        # Set the model to evaluation mode
        prediction_model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            
            # Get the prediction for the entire graph
            original_out = prediction_model(data.x, data.edge_index, data.edge_attr)
        
            # Get the prediction for the specific node
            original_node_pred = original_out[node_idx]
        
            # The raw output is logits, you can apply softmax if you need probabilities
            original_node_prob = torch.softmax(original_node_pred, dim=0)
            
         
        # --------------------------------
        # Step5.2.2.2: after perturbation of GXAttacker
        # --------------------------------
        
        # Set the model to evaluation mode
        prediction_model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            
            # Get the prediction for the entire graph
            perturbed_out = prediction_model(data.x, perturbed_edge_index, perturbed_edge_weights)
        
            # Get the prediction for the specific node
            perturbed_node_pred = perturbed_out[node_idx]
        
            # The raw output is logits, you can apply softmax if you need probabilities
            perturbed_node_prob = torch.softmax(perturbed_node_pred, dim=0)
            
        original_node_prob_short =  [f"{num:.3f}" for num in original_node_prob.tolist()]
        perturbed_node_prob_short_GXA =  [f"{num:.3f}" for num in perturbed_node_prob.tolist()]
        
        print("Prediction Probabilities Change for GXAttacker:\n", original_node_prob_short, "->", perturbed_node_prob_short_GXA)
    
        # --------------------------------
        # Step5.2.2.2: after perturbation of random_flipping
        # --------------------------------
        
        # Set the model to evaluation mode
        prediction_model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            
            # Get the prediction for the entire graph
            perturbed_out_rf = prediction_model(perturbed_data_rf.x, perturbed_data_rf.edge_index, perturbed_data_rf.edge_attr)
        
            # Get the prediction for the specific node
            perturbed_node_pred_rf = perturbed_out_rf[node_idx]
        
            # The raw output is logits, you can apply softmax if you need probabilities
            perturbed_node_prob_rf = torch.softmax(perturbed_node_pred_rf, dim=0)
            
            
        perturbed_node_prob_short_rf =  [f"{num:.3f}" for num in perturbed_node_prob_rf.tolist()]
        
        print("Prediction Probabilities Change for random_flipping:\n", original_node_prob_short, "->", perturbed_node_prob_short_rf)
    
    
        # --------------------------------
        # Step5.2.2.3: after perturbation of random_rewiring
        # --------------------------------
        
        # Set the model to evaluation mode
        prediction_model.eval()
        
        # Disable gradient computation for inference
        with torch.no_grad():
            
            # Get the prediction for the entire graph
            perturbed_out_rw = prediction_model(perturbed_data_rw.x, perturbed_data_rw.edge_index, perturbed_data_rw.edge_attr)
        
            # Get the prediction for the specific node
            perturbed_node_pred_rw = perturbed_out_rw[node_idx]
        
            # The raw output is logits, you can apply softmax if you need probabilities
            perturbed_node_prob_rw = torch.softmax(perturbed_node_pred_rw, dim=0)
            
            
        perturbed_node_prob_short_rw =  [f"{num:.3f}" for num in perturbed_node_prob_rw.tolist()]
        
        print("Prediction Probabilities Change for random_rewiring:\n", original_node_prob_short, "->", perturbed_node_prob_short_rw)
    
    
        # --------------------------------------------------------------
        # Step5.2.3: Compute cosine similarity between original explanation and perturbed explanation
        #            In particular, we set the top 25% important edges as 1, while the rest as zeros.
        # --------------------------------------------------------------
        
        import torch.nn.functional as F
        
        node_nums = data.x.size(0)
        
        # # Create sparse adjacency matrices using continuous edge mask
        # adj_matrix_original = torch.sparse_coo_tensor(data.edge_index, original_edge_mask_copy, (node_nums, node_nums))
        # adj_matrix_perturbed = torch.sparse_coo_tensor(perturbed_edge_index, perturbed_edge_mask_copy, (node_nums, node_nums))
        
        # --------------------------------
        # Step5.2.3.1: cos similarity for GXAttacker
        # --------------------------------
    
        ##set the topk_percent to 1 while the remaining to zeros
        perturbed_edge_mask = torch.where(torch.isin(torch.arange(len(perturbed_edge_mask)), torch.topk(perturbed_edge_mask, topk_percent).indices), torch.tensor(1.0), torch.tensor(0.0))
      
        ## Create sparse adjacency matrices using discretised edge mask
        adj_matrix_original = torch.sparse_coo_tensor(data.edge_index, original_edge_mask, (node_nums, node_nums))
        adj_matrix_perturbed_GXA = torch.sparse_coo_tensor(perturbed_edge_index, perturbed_edge_mask, (node_nums, node_nums))
        
        
        # Convert sparse adjacency matrices to dense format
        adj_matrix_original_dense = adj_matrix_original.to_dense()
        adj_matrix_perturbed_dense_GXA = adj_matrix_perturbed_GXA.to_dense()
        
        
        # Flatten the dense matrices to compute cosine similarity as if they are vectors
        flat_adj_original = adj_matrix_original_dense.flatten()
        flat_adj_perturbed_GXA = adj_matrix_perturbed_dense_GXA.flatten()
        
        # Add a batch dimension and compute cosine similarity
        cos_sim_GXA = F.cosine_similarity(flat_adj_original.unsqueeze(0), flat_adj_perturbed_GXA.unsqueeze(0))
        
        print("Cosine Similarity for GXA:", cos_sim_GXA.item())
        
        # --------------------------------
        # Step5.2.3.2: cos similarity for random_flipping
        # --------------------------------
    
        ##set the topk_percent to 1 while the remaining to zeros
        rf_edge_mask = torch.where(torch.isin(torch.arange(len(rf_edge_mask)), torch.topk(rf_edge_mask, topk_percent).indices), torch.tensor(1.0), torch.tensor(0.0))
    
        ## Create sparse adjacency matrices using discretised edge mask
        adj_matrix_perturbed_rf = torch.sparse_coo_tensor(perturbed_data_rf.edge_index, rf_edge_mask, (node_nums, node_nums))
        
        # Convert sparse adjacency matrices to dense format
        adj_matrix_perturbed_dense_rf = adj_matrix_perturbed_rf.to_dense()
        
        # Flatten the dense matrices to compute cosine similarity as if they are vectors
        flat_adj_perturbed_rf = adj_matrix_perturbed_dense_rf.flatten()
        
        # Add a batch dimension and compute cosine similarity
        cos_sim_rf = F.cosine_similarity(flat_adj_original.unsqueeze(0), flat_adj_perturbed_rf.unsqueeze(0))
        
        print("Cosine Similarity for random_flipping:", cos_sim_rf.item())  
        
        # --------------------------------
        # Step5.2.3.3: cos similarity for random_wiring
        # --------------------------------
        
        ##set the topk_percent to 1 while the remaining to zeros
        rw_edge_mask = torch.where(torch.isin(torch.arange(len(rw_edge_mask)), torch.topk(rw_edge_mask, topk_percent).indices), torch.tensor(1.0), torch.tensor(0.0))
    
        ## Create sparse adjacency matrices using discretised edge mask
        adj_matrix_perturbed_rw = torch.sparse_coo_tensor(perturbed_data_rw.edge_index, rw_edge_mask, (node_nums, node_nums))
        
        # Convert sparse adjacency matrices to dense format
        adj_matrix_perturbed_dense_rw = adj_matrix_perturbed_rw.to_dense()
        
        # Flatten the dense matrices to compute cosine similarity as if they are vectors
        flat_adj_perturbed_rw = adj_matrix_perturbed_dense_rw.flatten()
        
        # Add a batch dimension and compute cosine similarity
        cos_sim_rw = F.cosine_similarity(flat_adj_original.unsqueeze(0), flat_adj_perturbed_rw.unsqueeze(0))
        
        print("Cosine Similarity for random_wiring:", cos_sim_rw.item()) 
        
        
        # At the end of each iteration, collect the results for the current node
        attack_results_prob.append({
                    'node_idx': node_idx,
                    'used_budget': used_budget,
                    'original_node_prob': original_node_prob_short,
                    'perturbed_node_prob_GXA': perturbed_node_prob_short_GXA,
                    'perturbed_node_prob_rf': perturbed_node_prob_short_rf,
                    'perturbed_node_prob_rw': perturbed_node_prob_short_rw,
                    'pg_acc_original': pg_acc_original,
                    'pg_acc_perturbed_GXA': pg_acc_perturbed_GXA,
                    'pg_acc_perturbed_rf': pg_acc_perturbed_rf,
                    'pg_acc_perturbed_rw': pg_acc_perturbed_rw,
                    'cosine_similarity_GXA': cos_sim_GXA.item(),
                    'cosine_similarity_rf': cos_sim_rf.item(),
                    'cosine_similarity_rw': cos_sim_rw.item()
                    })
        
        
    import pandas as pd
    # Convert the list of results into a DataFrame
    save_path = './exp_results'
    if group_cnt == 0:
        attack_results_prob5 = pd.DataFrame(attack_results_prob)
        attack_results_prob5.to_csv(f"{save_path}/attack_results_prob5_{args.dataset_name}_s{seed_number}.csv", index=False)
    elif group_cnt == 1:
        attack_results_prob6 = pd.DataFrame(attack_results_prob)
        attack_results_prob6.to_csv(f"{save_path}/attack_results_prob6_{args.dataset_name}_s{seed_number}.csv", index=False)
    elif group_cnt == 2:
        attack_results_prob7 = pd.DataFrame(attack_results_prob)
        attack_results_prob7.to_csv(f"{save_path}/attack_results_prob7_{args.dataset_name}_s{seed_number}.csv", index=False)
    elif group_cnt == 3:
        attack_results_prob8 = pd.DataFrame(attack_results_prob)
        attack_results_prob8.to_csv(f"{save_path}/attack_results_prob8_{args.dataset_name}_s{seed_number}.csv", index=False)
    elif group_cnt == 4:
        attack_results_prob9 = pd.DataFrame(attack_results_prob)
        attack_results_prob9.to_csv(f"{save_path}/attack_results_prob9_{args.dataset_name}_s{seed_number}.csv", index=False)
    
    group_cnt += 1
    
    
# =============================================================================
# Some results analysis for Table 3
# =============================================================================
import ast
import pandas as pd

df1 = attack_results_prob5
df2 = attack_results_prob6
df3 = attack_results_prob7
df4 = attack_results_prob8
df5 = attack_results_prob9

df_all = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)

mydata = df_all

# Assuming mydata has already been loaded with your data

# Function to parse the string representation of lists and convert to floats
def parse_and_convert_to_floats(list_str):
    return [float(num) for num in ast.literal_eval(list_str)]

# Convert the columns from string representations of lists to actual lists of floats, if they are not already lists
def convert_if_needed(item):
    if isinstance(item, str):
        return parse_and_convert_to_floats(item)
    return item

mydata['original_node_prob'] = mydata['original_node_prob'].apply(convert_if_needed)
mydata['perturbed_node_prob_GXA'] = mydata['perturbed_node_prob_GXA'].apply(convert_if_needed)
mydata['perturbed_node_prob_rf'] = mydata['perturbed_node_prob_rf'].apply(convert_if_needed)
mydata['perturbed_node_prob_rw'] = mydata['perturbed_node_prob_rw'].apply(convert_if_needed)

# Calculate the original and perturbed labels by finding the index of the max value
mydata['original_label'] = mydata['original_node_prob'].apply(lambda probs: probs.index(max(probs)))
mydata['perturbed_label_GXA'] = mydata['perturbed_node_prob_GXA'].apply(lambda probs: probs.index(max(probs)))
mydata['perturbed_label_rf'] = mydata['perturbed_node_prob_rf'].apply(lambda probs: probs.index(max(probs)))
mydata['perturbed_label_rw'] = mydata['perturbed_node_prob_rw'].apply(lambda probs: probs.index(max(probs)))

# # Calculate the absolute changes in the probabilities for the originally predicted class

def safe_indexing(prob_list, index):
    try:
        # Convert the string at the specified index to a float
        return float(prob_list[int(index)])
    except (IndexError, ValueError, TypeError):
        return 0.0  # Return 0.0 or an appropriate default value if there's an error

# Apply this updated function in your DataFrame operation
mydata['prob_change_GXA'] = mydata.apply(lambda row: abs(
    safe_indexing(row['perturbed_node_prob_GXA'], row['original_label']) - 
    safe_indexing(row['original_node_prob'], row['original_label'])
), axis=1)

mydata['prob_change_rf'] = mydata.apply(lambda row: abs(
    safe_indexing(row['perturbed_node_prob_rf'], row['original_label']) - 
    safe_indexing(row['original_node_prob'], row['original_label'])
), axis=1)

mydata['prob_change_rw'] = mydata.apply(lambda row: abs(
    safe_indexing(row['perturbed_node_prob_rw'], row['original_label']) - 
    safe_indexing(row['original_node_prob'], row['original_label'])
), axis=1)


# Calculate various metrics
original_exp_acc = mydata["pg_acc_original"].mean()  ## Original Explanation Accuracy

perturbed_exp_acc_GXA = mydata["pg_acc_perturbed_GXA"].mean()  ## Perturbed Explanation Accuracy
exp_acc_change_GXA = original_exp_acc - perturbed_exp_acc_GXA  ## Explanation Accuracy Change

perturbed_exp_acc_rf = mydata["pg_acc_perturbed_rf"].mean()  ## Perturbed Explanation Accuracy
exp_acc_change_rf = original_exp_acc - perturbed_exp_acc_rf  ## Explanation Accuracy Change

perturbed_exp_acc_rw = mydata["pg_acc_perturbed_rw"].mean()  ## Perturbed Explanation Accuracy
exp_acc_change_rw = original_exp_acc - perturbed_exp_acc_rw  ## Explanation Accuracy Change


percentage_label_change_GXA = 100 * (mydata['original_label'] != mydata['perturbed_label_GXA']).mean()  ## Percentage of Label Changes
percentage_label_change_rf = 100 * (mydata['original_label'] != mydata['perturbed_label_rf']).mean()  ## Percentage of Label Changes
percentage_label_change_rw = 100 * (mydata['original_label'] != mydata['perturbed_label_rw']).mean()  ## Percentage of Label Changes

average_prob_change_GXA = mydata['prob_change_GXA'].mean()  ## Average Absolute Probability Change for the originally predicted class
average_prob_change_rf = mydata['prob_change_rf'].mean()  ## Average Absolute Probability Change for the originally predicted class
average_prob_change_rw = mydata['prob_change_rw'].mean()  ## Average Absolute Probability Change for the originally predicted class



cosine_similarity_GXA = mydata["cosine_similarity_GXA"].mean()  ## Cosine Similarity
cosine_similarity_rf = mydata["cosine_similarity_rf"].mean()  ## Cosine Similarity
cosine_similarity_rw = mydata["cosine_similarity_rw"].mean()  ## Cosine Similarity

used_budget = mydata["used_budget"].mean() / 2  ## Used Budget (assumed to halve as per your instruction)


# Print all results formatted as percentages only in the print function
print(f"Original Explanation Accuracy: {original_exp_acc * 100:.1f}%")

print(f"Perturbed Explanation Accuracy for GXA: {perturbed_exp_acc_GXA * 100:.1f}%")
print(f"Perturbed Explanation Accuracy for rf: {perturbed_exp_acc_rf * 100:.1f}%")
print(f"Perturbed Explanation Accuracy for rw: {perturbed_exp_acc_rw * 100:.1f}%")

print(f"Explanation Accuracy Change for GXA: {exp_acc_change_GXA * 100:.1f}%")
print(f"Explanation Accuracy Change for rf: {exp_acc_change_rf * 100:.1f}%")
print(f"Explanation Accuracy Change for rw: {exp_acc_change_rw * 100:.1f}%")


print(f"Percentage of Label Changes for GXA: {percentage_label_change_GXA:.1f}%")
print(f"Percentage of Label Changes for rf: {percentage_label_change_rf:.1f}%")
print(f"Percentage of Label Changes for rw: {percentage_label_change_rw:.1f}%")


print(f"Average Absolute Probability Change for GXA: {average_prob_change_GXA * 100:.1f}%")
print(f"Average Absolute Probability Change for rf: {average_prob_change_rf * 100:.1f}%")
print(f"Average Absolute Probability Change for rw: {average_prob_change_rw * 100:.1f}%")


print(f"Cosine Similarity for GXA: {cosine_similarity_GXA * 100:.1f}%")
print(f"Cosine Similarity for rf: {cosine_similarity_rf * 100:.1f}%")
print(f"Cosine Similarity for rw: {cosine_similarity_rw * 100:.1f}%")

print(f"Used Budget for GXA: {used_budget * 1:.1f}")

# =============================================================================
# Save results into dataframe
# =============================================================================

# Collect data for each metric in a dictionary with the variable names as keys
results_data = {
    "original_exp_acc": f"{original_exp_acc * 100:.1f}%",
    "ptb_exp_acc_GXA": f"{perturbed_exp_acc_GXA * 100:.1f}%",
    "ptb_exp_acc_rf": f"{perturbed_exp_acc_rf * 100:.1f}%",
    "ptb_exp_acc_rw": f"{perturbed_exp_acc_rw * 100:.1f}%",
    "exp_acc_change_GXA": f"{exp_acc_change_GXA * 100:.1f}%",
    "exp_acc_change_rf": f"{exp_acc_change_rf * 100:.1f}%",
    "exp_acc_change_rw": f"{exp_acc_change_rw * 100:.1f}%",
    "pc_label_change_GXA": f"{percentage_label_change_GXA:.1f}%",
    "pc_label_change_rf": f"{percentage_label_change_rf:.1f}%",
    "pc_label_change_rw": f"{percentage_label_change_rw:.1f}%",
    "avg_prob_change_GXA": f"{average_prob_change_GXA * 100:.1f}%",
    "avg_prob_change_rf": f"{average_prob_change_rf * 100:.1f}%",
    "avg_prob_change_rw": f"{average_prob_change_rw * 100:.1f}%",
    "cos_sim_GXA": f"{cosine_similarity_GXA * 100:.1f}%",
    "cos_sim_rf": f"{cosine_similarity_rf * 100:.1f}%",
    "cos_sim_rw": f"{cosine_similarity_rw * 100:.1f}%",
    "used_budget_GXA": f"{used_budget * 1:.1f}"
}

# Convert the dictionary into a DataFrame
results_df = pd.DataFrame([results_data])

# Optionally, you can transpose the DataFrame to have metrics as rows and a single column for values
results_df = results_df.T
results_df.columns = ['Value']  # Rename the column

# # Optionally, save the DataFrame to a CSV file
results_df.to_csv(f"analysis_results_{args.dataset_name}_seed{seed_number}.csv", header=True, index_label="Metric")





