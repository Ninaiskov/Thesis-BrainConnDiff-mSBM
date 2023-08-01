for ini in 1 2 3 4 5 6 7 8 9 10
do
for K in 2 5 10
do
for S1 in 5 10
do
for Nc_type in 'balanced' 'unbalanced'
do
for eta_similarity in 'same' 'comp_diff' 'part_diff'
do
for model_type in 'parametric' 'nonparametric'
do
for noc in 2 5 10
do
    sed -i '$ d' submit_big.sh
    echo "python3 main.py --dataset synthetic --maxiter_gibbs 100 --K $K --S1 $S1 --Nc_type $Nc_type --eta_similarity $eta_similarity --model_type $model_type --noc $noc" >> submit_big.sh
    bsub < submit_big.sh
done
done
done
done
done
done
done