conda create -n DynAnomKDD python=3.7
conda activate DynAnomKDD
pip install tqdm networkx numba==0.54.1 numpy scipy tbb pandas matplotlib scikit-learn --user 
echo '################################################'
echo '###### I: Please switch to conda environment ###'
echo '###### $ conda activate DynAnomKDD #############'
echo '################################################'
echo '###### II: Sanity check ########################'
echo '###### running DynamicPPE.py ###################'
echo '###### Should see the followings: ##############'
echo '################################################'
echo 'appr. ppr  [0.474 0.081 0.222 0.143 0.081]'
echo 'exact ppr [0.474 0.081 0.222 0.143 0.081]'
echo '################################################'
echo '############ Running ###########################'
python ./src/algorithms/DynAnom/DynamicPPE.py