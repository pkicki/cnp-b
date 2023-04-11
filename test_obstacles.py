import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

from losses.obstacles2D import Obstacles2DLoss
from models.obstacles2D_planner_boundaries import Obstacles2DPlannerBoundaries
from utils.data import unpack_data_obstacles2D
from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.constants import Limits, TableConstraint, UrdfModels
import matplotlib.pyplot as plt

class args:
    batch_size = 128
    log_interval = 100
    #dataset_path = "./data/paper/obstacles2D_simple/train/data.tsv"
    dataset_path = "./data/paper/obstacles2D_boundaries/train/data.tsv"

plot = True

train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

test_data = np.loadtxt(args.dataset_path.replace("train", "test"), delimiter='\t').astype(np.float32)
test_size = test_data.shape[0]
test_ds = tf.data.Dataset.from_tensor_slices(test_data)

ds = test_ds
ds_size = test_size

N = 15
loss = Obstacles2DLoss(N)
model = Obstacles2DPlannerBoundaries(N, 3, 3, loss.bsp, loss.bsp_t)

ckpt_striker = tf.train.Checkpoint(model=model)
#ckpt_striker.restore("./trainings/test_obs2D/checkpoints/best-122")

train_step = 0
val_step = 0
dataset_epoch = ds.shuffle(ds_size)
dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
epoch_loss = []
unscaled_epoch_loss = []
xy_dot_losses = []
xy_ddot_losses = []
obstacle_losses = []
for i, d in _ds('Test', dataset_epoch, train_size, 0, args.batch_size):
    with tf.GradientTape(persistent=True) as tape:
        xy_cps, t_cps = model(d)
        model_loss, xy_dot_loss, xy_ddot_loss, obstacle_loss, \
        xy, xy_dot, xy_ddot, t, t_cumsum, t_loss, dt, unscaled_model_loss = loss(xy_cps, t_cps, d)
        a = 0

        if plot:
            xy0, xyk, dxy0, dxyk, obstacles = unpack_data_obstacles2D(d)
            obs = obstacles.numpy().reshape((-1, 10, 3))
            for i in range(d.shape[0]):
                plt.clf()
                plt.plot(xy0[i, 0], xy0[i, 1], 'go')
                plt.plot(xyk[i, 0], xyk[i, 1], 'rx')
                plt.plot(xy[i, :, 0], xy[i, :, 1])
                #[plt.Circle((obs[i, 0], obs[i, 1]), obs[i, 2], color='r') for k in range(obs.shape[1])]
                circles = [plt.Circle((o[0], o[1]), o[2], color='r') for o in obs[i]]
                [plt.gca().add_patch(c) for c in circles]
                plt.xlim(0., 1.)
                plt.ylim(0., 1.)
                plt.show()


    xy_dot_losses.append(xy_dot_loss)
    xy_ddot_losses.append(xy_ddot_loss)
    obstacle_losses.append(obstacle_loss)
    epoch_loss.append(model_loss)
    unscaled_epoch_loss.append(unscaled_model_loss)
    train_step += 1

xy_dot_losses = tf.reduce_mean(tf.concat(xy_dot_losses, 0))
xy_ddot_losses = tf.reduce_mean(tf.concat(xy_ddot_losses, 0))
obstacle_losses = tf.reduce_mean(tf.concat(obstacle_losses, 0))
loss.alpha_update(xy_dot_losses, xy_ddot_losses, obstacle_losses)
epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))