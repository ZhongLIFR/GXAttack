
#!/bin/bash

# Loop through dataset names
i=1

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 10 --subgraph_size 6 --prob_connection 0.3 

i=2

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 50 --subgraph_size 6 --prob_connection 0.06 


i=3

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 50 --subgraph_size 10 --prob_connection 0.06

i=4

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "Circle" --num_subgraphs 50 --subgraph_size 10 --prob_connection 0.06

i=5

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 50 --subgraph_size 10 --prob_connection 0.12

i=6

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 50 --subgraph_size 10 --prob_connection 0.20

i=7

echo "Running Python script for generating dataset_Syn${i}.pth"
python Generate_Dataset.py --dataset_name "dataset_Syn${i}.pth" --shape "house" --num_subgraphs 100 --subgraph_size 10 --prob_connection 0.06
