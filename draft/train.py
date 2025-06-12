import tensorflow as tf
import numpy as np
import time
import os

from config import BATCH_SIZE, MAX_SEQ_LEN, LEARNING_RATE, MAX_EPOCHS, WARMUP_STEPS, LOAD_BALANCE_LOSS_WEIGHT, CHECKPOINT_PATH, SPECIALIZATION_LOSS_WEIGHT
from model import MoETransformer

# Custom learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=WARMUP_STEPS):
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Loss function
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# Accuracy function
def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

# Training function
def train_step(model, inp, tar, optimizer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    with tf.GradientTape() as tape:
        predictions, load_balance_loss, specialization_loss = model([inp, tar_inp], training=True)
        loss = loss_function(tar_real, predictions)
        total_loss = loss + LOAD_BALANCE_LOSS_WEIGHT * load_balance_loss + SPECIALIZATION_LOSS_WEIGHT * specialization_loss
    
    variables = model.trainable_variables
    gradients = tape.gradient(total_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss, load_balance_loss, specialization_loss, accuracy_function(tar_real, predictions)

# Main training loop
def train_model(dataset, validation_dataset=None):
    model = MoETransformer()
    learning_rate = CustomSchedule(512)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    checkpoint_dir = CHECKPOINT_PATH
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # Create metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    train_lb_loss = tf.keras.metrics.Mean(name='train_lb_loss')
    train_spec_loss = tf.keras.metrics.Mean(name='train_spec_loss')
    
    for epoch in range(MAX_EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        train_lb_loss.reset_states()
        train_spec_loss.reset_states()
        
        for (batch, (inp, tar)) in enumerate(dataset):
            loss, lb_loss, spec_loss, acc = train_step(model, inp, tar, optimizer)
            
            train_loss(loss)
            train_accuracy(acc)
            train_lb_loss(lb_loss)
            train_spec_loss(spec_loss)
            
            if batch % 50 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {train_loss.result():.4f} LB Loss {train_lb_loss.result():.4f} Spec Loss {train_spec_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        
        print(f'Epoch {epoch+1} Loss {train_loss.result():.4f} LB Loss {train_lb_loss.result():.4f} Spec Loss {train_spec_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        print(f'Time taken for 1 epoch: {time.time()-start:.2f} secs\n')

if __name__ == '__main__':
    # Placeholder for dataset loading
    # You would need to implement your own dataset loading logic here
    print('Dataset loading logic needs to be implemented')
    # Example:
    # dataset = load_dataset_function()
    # train_model(dataset) 