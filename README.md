Constrained Manifold Neural Motion Planning with B-splines (CNP-B)
--

Implementation of [P. Kicki et al., "Fast Kinodynamic Planning on the Constraint Manifold With Deep Neural Networks," in IEEE Transactions on Robotics, vol. 40, pp. 277-297, 2024](https://ieeexplore.ieee.org/document/10292912).

![main image](media/hitting.gif)

See also:\
[paper](https://ieeexplore.ieee.org/document/10292912) \
[website](https://sites.google.com/view/constrained-neural-planning/) \
[preprint](https://arxiv.org/abs/2301.04330)

## Dependencies
General
* Tensorflow `pip install tensorflow`
* Tensorflow-graphics `pip install tensorflow-graphics`
* NumPy `pip install numpy`
* Matplotlib `pip install matplotlib`
* Pinocchio `sudo apt install ros-noetic-pinocchio`

For Air-Hockey hitting
* Nlopt `sudo apt install libnlopt-cxx-dev libnlopt-dev`
* Coin-or-CLP `sudo apt install coinor-libclp-dev`

For demonstration of motion planning in ROS
* [docker](https://docs.docker.com/engine/install/ubuntu/)

For results plotting and statistical analysis
* SciPy ``
* statsmodels `pip install statsmodels`

## Usage
Download data
```
bash download_datasets.sh
```
Download pre-trained models
```
bash download_models.sh
```
Build python bindings (for Air Hockey hitting only)
```
bash build.sh
```
Make an inference of the model on a sample Air Hockey hitting problem
```
python examples/air_hockey_hitting.py
```
or moving a vertically oriented heavy object
```
python examples/heavy_object.py
```

## Use for motion planning in ROS
Run docker container
```
cd docker && ./run.sh
```
Run demo
```
bash demo/demo.sh
```

## Cite
```
@ARTICLE{kicki2024kinodynamic,
  author={Kicki, Piotr and Liu, Puze and Tateo, Davide and Bou-Ammar, Haitham and Walas, Krzysztof and Skrzypczyński, Piotr and Peters, Jan},
  journal={IEEE Transactions on Robotics}, 
  title={Fast Kinodynamic Planning on the Constraint Manifold With Deep Neural Networks}, 
  year={2024},
  volume={40},
  number={},
  pages={277-297},
  doi={10.1109/TRO.2023.3326922}}
```




