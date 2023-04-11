import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

from losses.obstacles2D import Obstacles2DLoss
from models.obstacles2D_planner_boundaries import Obstacles2DPlannerBoundaries
from utils.dataset import _ds
from utils.execution import ExperimentHandler
from utils.constants import Limits, TableConstraint, UrdfModels

class args:
    batch_size = 128
    working_dir = './trainings'
    out_name = 'test_obs2D_boundaries_alpha000'
    log_interval = 100
    learning_rate = 5e-5
    #dataset_path = "./data/paper/obstacles2D_simple/train/data.tsv"
    dataset_path = "./data/paper/obstacles2D_boundaries/train/data.tsv"


train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.striker)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = Obstacles2DLoss(N)
model = Obstacles2DPlannerBoundaries(N, 3, 3, loss.bsp, loss.bsp_t)

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
    xy_dot_losses = []
    xy_ddot_losses = []
    obstacle_losses = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            xy_cps, t_cps = model(d)
            model_loss, xy_dot_loss, xy_ddot_loss, obstacle_loss, \
            xy, xy_dot, xy_ddot, t, t_cumsum, t_loss, dt, unscaled_model_loss = loss(xy_cps, t_cps, d)
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        xy_dot_losses.append(xy_dot_loss)
        xy_ddot_losses.append(xy_ddot_loss)
        obstacle_losses.append(obstacle_loss)
        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=train_step)
            tf.summary.scalar('metrics/obstacle_loss', tf.reduce_mean(obstacle_loss), step=train_step)
            tf.summary.scalar('metrics/xy_dot_loss', tf.reduce_mean(xy_dot_loss), step=train_step)
            tf.summary.scalar('metrics/xy_ddot_loss', tf.reduce_mean(xy_ddot_loss), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
        train_step += 1

    xy_dot_losses = tf.reduce_mean(tf.concat(xy_dot_losses, 0))
    xy_ddot_losses = tf.reduce_mean(tf.concat(xy_ddot_losses, 0))
    obstacle_losses = tf.reduce_mean(tf.concat(obstacle_losses, 0))
    loss.alpha_update(xy_dot_losses, xy_ddot_losses, obstacle_losses)
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_xy_dot', loss.alpha_xy_dot, step=epoch)
        tf.summary.scalar('epoch/alpha_xy_ddot', loss.alpha_xy_ddot, step=epoch)
        tf.summary.scalar('epoch/alpha_obstacle', loss.alpha_obstacle, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        xy_cps, t_cps = model(d)
        model_loss, xy_dot_loss, xy_ddot_loss, obstacle_loss, \
        xy, xy_dot, xy_ddot, t, t_cumsum, t_loss, dt, unscaled_model_loss = loss(xy_cps, t_cps, d)

        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=val_step)
            tf.summary.scalar('metrics/obstacle_loss', tf.reduce_mean(obstacle_loss), step=val_step)
            tf.summary.scalar('metrics/xy_dot_loss', tf.reduce_mean(xy_dot_loss), step=val_step)
            tf.summary.scalar('metrics/xy_ddot_loss', tf.reduce_mean(xy_ddot_loss), step=val_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)

    w = 25
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_unscaled_epoch_loss > unscaled_epoch_loss:
        best_unscaled_epoch_loss = unscaled_epoch_loss
        experiment_handler.save_best()
    else:
        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            experiment_handler.save_best()
