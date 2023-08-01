for ini in 1 2 3 4 5 6 7 8 9 10
do
for n_rois in 100 200 300
do
    sed -i '$ d' submit_big.sh
    echo "python3 main.py --dataset hcp --n_rois $n_rois --noc 2 --model_type nonparametric --threshold_annealing False " >> submit_big.sh
    bsub < submit_big.sh
done
done