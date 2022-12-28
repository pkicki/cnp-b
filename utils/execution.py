import argparse
import os
from glob import glob

import tensorflow as tf

class ExperimentHandler:
    def __init__(self, working_path, out_name, log_interval, model, optimizer) -> None:
        super().__init__()
        train_log_path = os.path.join(working_path, out_name, 'logs', 'train')
        val_log_path = os.path.join(working_path, out_name, 'logs', 'val')
        self.checkpoints_last_n_path = os.path.join(working_path, out_name, 'checkpoints', 'last_n')
        self.checkpoints_best_path = os.path.join(working_path, out_name, 'checkpoints', 'best')
        self.checkpoints_best_dir = os.path.dirname(self.checkpoints_best_path)

        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        os.makedirs(self.checkpoints_last_n_path, exist_ok=True)
        os.makedirs(self.checkpoints_best_path, exist_ok=True)

        self.train_writer = tf.summary.create_file_writer(train_log_path)
        self.val_writer = tf.summary.create_file_writer(val_log_path)

        self.ckpt = tf.train.Checkpoint(optimizer=optimizer,
                                        model=model)
        self.log_interval = log_interval

    def log_training(self):
        self.train_writer.set_as_default()

    def log_validation(self):
        self.val_writer.set_as_default()

    def flush(self):
        self.train_writer.flush()
        self.val_writer.flush()

    def save_best(self):
        self.ckpt.save(self.checkpoints_best_path)
        list_of_files = [f for f in os.listdir(self.checkpoints_best_dir) if f.startswith("best-") and "." in f]
        full_path = [os.path.join(self.checkpoints_best_dir, x) for x in list_of_files]
        if len(full_path) > 20:
            oldest_file = min(full_path, key=os.path.getctime)
            oldest_files = glob(oldest_file[:oldest_file.rfind(".")+1] + "*")
            for f in oldest_files:
                os.remove(f)

    def save_last(self):
        self.ckpt.save(self.checkpoints_last_n_path)

    def restore(self, path):
        self.ckpt.restore(path)


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, 'unknownargs', list())
        with values as f:
            _, unknown_args = parser.parse_known_args(f.read().split(), namespace)
            if unknown_args is not None and len(unknown_args) > 0:
                namespace.unknownargs.extend(unknown_args)
