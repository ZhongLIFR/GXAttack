
#!/bin/bash

# Loop through dataset names
for i in {1..7}
do
   echo "Running Python script for dataset_Syn${i}.pth"
   python run.py --dataset_name "dataset_Syn${i}"
done

