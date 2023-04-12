import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from losses.obstacles2D import Obstacles2DLoss, Obstacles2DPathLoss
from models.obstacles2D_planner_boundaries import Obstacles2DPlannerBoundaries, Obstacles2DPathPlannerBoundaries, \
    Obstacles2DInvPathPlannerBoundaries
from utils.data import unpack_data_obstacles2D
from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.constants import Limits, TableConstraint, UrdfModels

class args:
    #batch_size = 128
    batch_size = 2
    working_dir = './trainings'
    out_name = 'test_obs2D_boundaries_alpha000_path_n10_bs2_lr5em6_bar1em5_inv'
    log_interval = 100
    learning_rate = 5e-6
    #dataset_path = "./data/paper/obstacles2D_simple/train/data.tsv"
    dataset_path = "./data/paper/obstacles2D_boundaries/train/data.tsv"

n = 10

train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)[:n]
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)[:n]
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = Obstacles2DPathLoss(N)
model = Obstacles2DInvPathPlannerBoundaries(N)

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

train_step = 0
val_step = 0
best_epoch_loss = 1e10
best_unscaled_epoch_loss = 1e10
for epoch in range(30000):
    # training
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_training()
    obstacle_losses = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            xy_cps = model(d)
            model_loss, s_loss, obstacle_loss, xy, unscaled_model_loss = loss(xy_cps, d)

        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        xy0, xyk, dxy0, dxyk, obstacles = unpack_data_obstacles2D(d)
        obs = obstacles.numpy().reshape((-1, 10, 3))
        for i in range(d.shape[0]):
            plt.clf()
            plt.plot(xy0[i, 0], xy0[i, 1], 'go')
            plt.plot(xyk[i, 0], xyk[i, 1], 'rx')
            plt.plot(xy[i, :, 0], xy[i, :, 1])
            # [plt.Circle((obs[i, 0], obs[i, 1]), obs[i, 2], color='r') for k in range(obs.shape[1])]
            circles = [plt.Circle((o[0], o[1]), o[2], color='r') for o in obs[i]]
            [plt.gca().add_patch(c) for c in circles]
            plt.xlim(0., 1.)
            plt.ylim(0., 1.)
            plt.savefig(f"imgs/{epoch:05d}_{i:05d}.png")

        obstacle_losses.append(obstacle_loss)
        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=train_step)
            tf.summary.scalar('metrics/obstacle_loss', tf.reduce_mean(obstacle_loss), step=train_step)
            tf.summary.scalar('metrics/s_loss', tf.reduce_mean(s_loss), step=train_step)
        train_step += 1

    obstacle_losses = tf.reduce_mean(tf.concat(obstacle_losses, 0))
    loss.alpha_update(obstacle_losses)
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_obstacle', loss.alpha_obstacle, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        xy_cps = model(d)
        model_loss, s_loss, obstacle_loss, xy, unscaled_model_loss = loss(xy_cps, d)

        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=val_step)
            tf.summary.scalar('metrics/obstacle_loss', tf.reduce_mean(obstacle_loss), step=val_step)
            tf.summary.scalar('metrics/s_loss', tf.reduce_mean(s_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)

    #w = 25
    #if epoch % w == w - 1:
    #    experiment_handler.save_last()
    #if best_unscaled_epoch_loss > unscaled_epoch_loss:
    #    best_unscaled_epoch_loss = unscaled_epoch_loss
    #    experiment_handler.save_best()
    #else:
    #    if best_epoch_loss > epoch_loss:
    #        best_epoch_loss = epoch_loss
    #        experiment_handler.save_best()
