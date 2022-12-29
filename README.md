Constrained Manifold Neural Motion Planning with B-splines (CNP-B)
--

Implementation of "Fast Kinodynamic Planning on the Constraint
Manifold with Deep Neural Networks".

![main image](media/hitting.gif)

See also:\
[website](https://sites.google.com/view/constrained-neural-planning/) \
[paper](link_to_arxiv)

## Dependencies
General
* Tensorflow `pip install tensorflow`
* Tensorflow-graphics `pip install tensorflow-graphics`
* Pinocchio `sudo apt install ros-noetic-pinocchio`

For Air-Hockey hitting
* Nlopt `sudo apt install libnlopt-dev`
* Coin-or-CLP `sudo apt install coinor-libclp-dev`

For demonstration of motion planning in ROS
* nvidia-docker

## Usage
Download data
```
bash download_data.sh
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
python exmaples/air_hockey_hitting.py
```
or moving a vertically oriented heavy object
```
python exmaples/heavy_object.py
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




