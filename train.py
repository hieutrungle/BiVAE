'''
Implement a generic training loop
'''

from tqdm.autonotebook import tqdm
import timeit
import numpy as np
import os
import sys
from utils import utils
from generate import generate
import json
import tensorflow as tf
import pickle
import gc
from collections import namedtuple


def compute_kl_loss(model):
    total_log_q, total_log_p, kl_all, kl_diag = model.cal_kl_components()
    kl_loss = tf.reduce_sum(kl_all)
    return kl_loss

def compute_recon_loss(x_pred, x_orig):
    recon_loss = tf.reduce_sum(tf.square(tf.subtract(x_pred,x_orig)))
    return recon_loss

def compute_loss(model, x_orig, batch_size, kl_weight=1.0, training=False):
    x_pred, kl_loss = model(x_orig, training=training)
    recon_loss = compute_recon_loss(x_pred, x_orig)

    kl_loss = kl_loss / batch_size
    recon_loss = recon_loss / batch_size
    # kl_loss = compute_kl_loss(model)
    total_loss = recon_loss + tf.multiply(kl_weight, kl_loss)
    # print(f"\nrecon_loss: {recon_loss.numpy():0.6f} \tkl_loss: {kl_loss.numpy():0.6f}" +
    #         f"\t total_loss: {total_loss.numpy():0.6f}")
    return (total_loss, recon_loss, kl_loss)


@tf.function
def train_step(model, data_iter, losses, optimizer, 
                batch_size, steps_per_execution, kl_weight, strategy):
    def train_step_fn(model, x_orig, losses, 
                    optimizer, batch_size, kl_weight=1.0):
        with tf.GradientTape() as tape:
            (total_loss, recon_loss, kl_loss) = compute_loss(model, x_orig, 
                                                        batch_size, kl_weight, 
                                                        training=True)
            total_loss += sum(model.losses)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        #update metrics
        losses[0].update_state(total_loss)
        losses[1].update_state(recon_loss)
        losses[2].update_state(kl_loss)
    
    for _ in tf.range(steps_per_execution):
        strategy.run(
            train_step_fn,
            args=(model, next(data_iter), losses, optimizer, batch_size, kl_weight)
            )

def train(model, iterator, epochs, optimizer, train_portion, 
        model_dir, batch_size, steps_per_execution, kl_anneal_portion,
        epochs_til_ckpt, steps_til_summary, resume_checkpoint, strategy):
    
    summaries_dir, checkpoints_dir = utils.mkdir_storage(model_dir, resume_checkpoint)

    # Save training parameters if we need to resume training in the future
    start_epoch = 1
    if 'resume_epoch' in resume_checkpoint:
        start_epoch = resume_checkpoint['resume_epoch'] + 1

    train_loss_results, val_loss_results = [], []
    total_training_time = 0
    training_results = (train_loss_results, val_loss_results, total_training_time)

    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()
    kl_train_loss = tf.keras.metrics.Mean()
    recon_train_loss = tf.keras.metrics.Mean()
    kl_val_loss = tf.keras.metrics.Mean()
    recon_val_loss = tf.keras.metrics.Mean()
    train_losses = [train_loss, recon_train_loss, kl_train_loss]
    val_losses = [val_loss, recon_val_loss, kl_val_loss]

    min_kl_weight = 1e-3
    kl_weight = min(max((start_epoch-1)/(kl_anneal_portion*epochs), min_kl_weight), 1)

    print(f"\nStart Training...")
    print(f"Training steps per epoch:{steps_per_execution}")
    print(f"Batch size:{batch_size}\n")
    # Start training
    for epoch in range(start_epoch, epochs+1):
        
        # train_data, val_data = None, None
        # train_data, val_data = utils.split_train_val_tf(data, data.cardinality().numpy(), 
        #                                 train_size=train_portion, shuffle=True, 
        #                                 shuffle_size=data.cardinality().numpy(), seed=None)
        
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr.numpy()

        for loss in train_losses + val_losses:
            loss.reset_state()

        start_time = timeit.default_timer()
        
        # training
        tqdm.write(f"Epoch {epoch} ")

        train_step(model, iterator, tuple(train_losses), optimizer,
                    tf.constant(batch_size, dtype=tf.float32),
                    tf.constant(steps_per_execution, dtype=tf.int32),
                    tf.constant(kl_weight, dtype=tf.float32), strategy)
        
        train_loss_results.append(train_loss.result().numpy())
        val_loss_results.append(val_loss.result().numpy())

        training_time = timeit.default_timer()-start_time
        total_training_time += training_time
        
        # after each epoch
        tqdm.write(f"training time: {training_time:0.5f}, " + 
                    f"LR: {current_lr:0.5f}, kl_weight: {kl_weight:0.5f}, \n\t" +
                    f"kl_train_loss: {kl_train_loss.result():0.5f}, " +
                    f"recon_train_loss: {recon_train_loss.result():0.5f}, " +
                    f"train_loss: {train_loss.result():0.5f}, \n\t"
                    # f"kl_val_loss: {kl_val_loss.result():0.5f}, " +
                    # f"recon_val_loss: {recon_val_loss.result():0.5f}, " + 
                    # f"val_loss: {val_loss.result():0.5f}"
                    )
                    
        kl_weight = min(max((epoch)/(kl_anneal_portion*epochs), min_kl_weight), 1)

        training_results = (train_loss_results, val_loss_results, total_training_time)

        # save model when epochs_til_ckpt requirement is met
        if (not epoch % epochs_til_ckpt) and epoch:
            save_training_parameters(checkpoints_dir, epoch, model, training_results)

        gc.collect()
        
    print(f"total training time: {total_training_time:0.5f}")
    print(f"\nFinished Training!!!")
    
    # save model at end of training
    save_training_parameters(checkpoints_dir, epochs, model, training_results)

    # generate(model, next(iterator), model_dir, "gen_img_from_training.png")
    

def save_training_parameters(checkpoints_dir, epochs, model, training_results):
    (train_loss_results, val_loss_results, total_training_time) = training_results
    model.save_weights(os.path.join(checkpoints_dir, f'model_{epochs:06d}'))
    np.savetxt(os.path.join(checkpoints_dir, f'train_losses_{epochs:06d}.txt'),
                np.array(train_loss_results))
    np.savetxt(os.path.join(checkpoints_dir, f'val_losses_{epochs:06d}.txt'),
                np.array(val_loss_results))
    np.savetxt(os.path.join(checkpoints_dir, f'training_time_{epochs:06d}.txt'),
                np.array([total_training_time]))