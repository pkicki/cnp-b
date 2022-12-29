cd utils/hitting_point_optimization
git submodule update --init
rm -rf build/
bash build.sh
cd ..
bash get_hpo.sh