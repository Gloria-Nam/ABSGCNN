import os
run_train = 'python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 --dp dg-3-4-0.01 --lr 0.001 --task classification -b 32 --epochs 20 data/maz-classification'
os.system(run_train)
