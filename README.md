Constrained Manifold Neural Motion Planning with B-splines (CNP-B)
--

Implementation of "Fast Kinodynamic Planning on the Constraint
Manifold with Deep Neural Networks".

![main image](media/hitting.gif)

See also:\
[website](https://sites.google.com/view/constrained-neural-planning/) \
[paper](link_to_arxiv)

## Usage
Download data
```
bash download_data.sh
```
Download pre-trained models
```
bash download_models.sh
```
Make an inference of the model on a sample problem
```
python demo.py
```

## Use for motion planning in ROS
Run docker container
```
bash docker/run.sh
```
Run demo
```
bash demo/demo.sh
```




