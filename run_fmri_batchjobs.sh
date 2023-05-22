for ini in 1 2 3 4 5 6 7 8 9 10
do
for dataset in hcp decnef
do
for threshold_annealing in True False
do
    sed -i '$ d' submit_big.sh
    echo "python3 main.py --dataset $dataset --n_rois 100 --noc 30 --model_type nonparametric --threshold_annealing $threshold_annealing" >> submit_big.sh
    bsub < submit_big.sh
done
done
done