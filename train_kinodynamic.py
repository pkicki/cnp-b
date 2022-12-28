import os
import tensorflow as tf
import numpy as np

from losses.kinodynamic import KinodynamicLoss
from utils.dataset import _ds
from utils.execution import ExperimentHandler
from losses.constraint_functions import two_tables_object_collision
from models.iiwa_planner_boundaries import IiwaPlannerBoundariesKinodynamic
from utils.constants import Limits, UrdfModels


class args:
    batch_size = 128
    working_dir = './trainings'
    out_name = 'name_of_the_model'
    log_interval = 100
    learning_rate = 5e-5
    dataset_path = "./data/paper/kinodynamic/train/data.tsv"


train_data = np.loadtxt(args.dataset_path, delimiter='\t').astype(np.float32)
train_size = train_data.shape[0]
train_ds = tf.data.Dataset.from_tensor_slices(train_data)

val_data = np.loadtxt(args.dataset_path.replace("train", "val"), delimiter='\t').astype(np.float32)
val_size = val_data.shape[0]
val_ds = tf.data.Dataset.from_tensor_slices(val_data)

urdf_path = os.path.join(os.path.dirname(__file__), UrdfModels.iiwa_cup)

N = 15
opt = tf.keras.optimizers.Adam(args.learning_rate)
loss = KinodynamicLoss(N, urdf_path, two_tables_object_collision, None, Limits.q_dot7, Limits.q_ddot7, Limits.q_dddot7, Limits.tau7)
model = IiwaPlannerBoundariesKinodynamic(N, 3, 2, loss.bsp, loss.bsp_t)

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
    q_dot_losses = []
    q_ddot_losses = []
    q_dddot_losses = []
    constraint_losses = []
    torque_losses = []
    vertical_losses = []
    for i, d in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            q_cps, t_cps = model(d)
            model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, vertical_loss, \
            q, q_dot, q_ddot, q_dddot, torque, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss, \
            int_torque_loss, constraint_losses_ = loss(q_cps, t_cps, d)

        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        q_dot_losses.append(q_dot_loss)
        q_ddot_losses.append(q_ddot_loss)
        q_dddot_losses.append(q_dddot_loss)
        constraint_losses.append(constraint_loss)
        torque_losses.append(torque_loss)
        vertical_losses.append(vertical_loss)
        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=train_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss), step=train_step)
            tf.summary.scalar('metrics/torque_loss', tf.reduce_mean(torque_loss), step=train_step)
            tf.summary.scalar('metrics/vertical_loss', tf.reduce_mean(vertical_loss), step=train_step)
            tf.summary.scalar('metrics/int_torque_loss', tf.reduce_mean(int_torque_loss), step=train_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss), step=train_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss), step=train_step)
            tf.summary.scalar('metrics/q_dddot_loss', tf.reduce_mean(q_dddot_loss), step=train_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=train_step)
            tf.summary.scalar('metrics/jerk_loss', tf.reduce_mean(jerk_loss), step=train_step)
        train_step += 1

    q_dot_losses = tf.reduce_mean(tf.concat(q_dot_losses, 0))
    q_ddot_losses = tf.reduce_mean(tf.concat(q_ddot_losses, 0))
    q_dddot_losses = tf.reduce_mean(tf.concat(q_dddot_losses, 0))
    constraint_losses = tf.reduce_mean(tf.concat(constraint_losses, 0))
    torque_losses = tf.reduce_mean(tf.concat(torque_losses, 0))
    vertical_losses = tf.reduce_mean(tf.concat(vertical_losses, 0))
    loss.alpha_update(q_dot_losses, q_ddot_losses, q_dddot_losses, constraint_losses, torque_losses, vertical_losses)
    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    unscaled_epoch_loss = tf.reduce_mean(tf.concat(unscaled_epoch_loss, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/unscaled_loss', unscaled_epoch_loss, step=epoch)
        tf.summary.scalar('epoch/alpha_q_dot', loss.alpha_q_dot, step=epoch)
        tf.summary.scalar('epoch/alpha_q_ddot', loss.alpha_q_ddot, step=epoch)
        tf.summary.scalar('epoch/alpha_constraint', loss.alpha_constraint, step=epoch)
        tf.summary.scalar('epoch/alpha_torque', loss.alpha_torque, step=epoch)
        tf.summary.scalar('epoch/alpha_vertical', loss.alpha_vertical, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    unscaled_epoch_loss = []
    experiment_handler.log_validation()
    for i, d in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        q_cps, t_cps = model(d)
        model_loss, constraint_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, vertical_loss, \
        q, q_dot, q_ddot, q_dddot, torque, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss, \
        int_torque_loss, _ = loss(q_cps, t_cps, d)

        epoch_loss.append(model_loss)
        unscaled_epoch_loss.append(unscaled_model_loss)
        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/unscaled_model_loss', tf.reduce_mean(unscaled_model_loss), step=val_step)
            tf.summary.scalar('metrics/constraint_loss', tf.reduce_mean(constraint_loss), step=val_step)
            tf.summary.scalar('metrics/torque_loss', tf.reduce_mean(torque_loss), step=val_step)
            tf.summary.scalar('metrics/vertical_loss', tf.reduce_mean(vertical_loss), step=val_step)
            tf.summary.scalar('metrics/int_torque_loss', tf.reduce_mean(int_torque_loss), step=val_step)
            tf.summary.scalar('metrics/q_dot_loss', tf.reduce_mean(q_dot_loss), step=val_step)
            tf.summary.scalar('metrics/q_ddot_loss', tf.reduce_mean(q_ddot_loss), step=val_step)
            tf.summary.scalar('metrics/q_dddot_loss', tf.reduce_mean(q_dddot_loss), step=val_step)
            tf.summary.scalar('metrics/t', tf.reduce_mean(t), step=val_step)
            tf.summary.scalar('metrics/jerk_loss', tf.reduce_mean(jerk_loss), step=val_step)
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
