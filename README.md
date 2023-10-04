Constrained Manifold Neural Motion Planning with B-splines (CNP-B)
--

Implementation of "Fast Kinodynamic Planning on the Constraint
Manifold with Deep Neural Networks".

![main image](media/hitting.gif)

See also:\
[website](https://sites.google.com/view/constrained-neural-planning/) \
[paper](https://arxiv.org/abs/2301.04330)

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




