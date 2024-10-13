import torch
from graphxai.datasets import ShapeGGen
import os 
import argparse

# =============================================================================
# Step0. Settings of Dataset to be generated
# =============================================================================

parser = argparse.ArgumentParser(description='Save a dataset with a specified filename from command line.')
parser.add_argument('--dataset_name', type=str, help='Name of the dataset file to save.')
parser.add_argument('--shape', type=str, help='Shape of the dataset file to save.')
parser.add_argument('--num_subgraphs', type=int, help='Num of subgraphs of the dataset file to save.')
parser.add_argument('--prob_connection', type=float, help='Prob_connection of the dataset file to save.')
parser.add_argument('--subgraph_size', type=int, help='subgraph_size of the dataset file to save.')


args = parser.parse_args()


# =============================================================================
# Step1. Generate synthetic dataset 
# =============================================================================

dataset = ShapeGGen(
   model_layers = 2,
   shape = args.shape,
   num_subgraphs = args.num_subgraphs,
   subgraph_size = args.subgraph_size,
   prob_connection = args.prob_connection,
   add_sensitive_feature = False
)

##Example to use the code to generate new synthetic dataset:
# dataset = ShapeGGen(
#    model_layers = 2,
#    shape = 'house',
#    num_subgraphs = 150,
#    subgraph_size = 10,
#    prob_connection = 0.03,
#    add_sensitive_feature = False
# )

# --------------------------------------------
# Step2. Save the dataset to a file
# --------------------------------------------

save_path = f'./exp_results/{args.dataset_name}'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

torch.save(dataset, save_path)  # Save the dataset object
