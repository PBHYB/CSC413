import os

# os.environ["KERAS_BACKEND"] = "jax" # you can also use tensorflow or torch

import keras_cv
import keras
from keras import ops
import tensorflow as tf

import cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import joblib
import math

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras import ops
from data_set import *
from loss import *


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    preset = "efficientnetv2_s_imagenet"  # Name of pretrained classifier
    image_size = [224, 224]  # Input image size
    epochs = 6  # Training epochs
    batch_size = 96  # Batch size
    lr_mode = "step"  # LR scheduler mode from one of "cos", "step", "exp"
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6  # Number of classes in the dataset
    num_folds = 5  # Number of folds to split the dataset
    fold = 0  # Which fold to set as validation data
    class_names = [
        "X4_mean",
        "X11_mean",
        "X18_mean",
        "X26_mean",
        "X50_mean",
        "X3112_mean",
    ]
    aux_class_names = list(map(lambda x: x.replace("mean", "sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)


def get_lr_callback(batch_size=8, mode="cos", epochs=10, plot=False):
    lr_start, lr_max, lr_min = 5e-5, 8e-6 * batch_size, 1e-5
    lr_ramp_ep, lr_sus_ep, lr_decay = 3, 0, 0.75

    def lrfn(epoch):  # Learning rate update function
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        elif mode == "exp":
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        elif mode == "step":
            lr = lr_max * lr_decay ** ((epoch - lr_ramp_ep - lr_sus_ep) // 2)
        elif mode == "cos":
            decay_total_epochs, decay_epoch_index = (
                epochs - lr_ramp_ep - lr_sus_ep + 3,
                epoch - lr_ramp_ep - lr_sus_ep,
            )
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (lr_max - lr_min) * 0.5 * (1 + math.cos(phase)) + lr_min
        return lr

    if plot:  # Plot lr curve if plot is True
        plt.figure(figsize=(10, 5))
        plt.plot(
            np.arange(epochs), [lrfn(epoch) for epoch in np.arange(epochs)], marker="o"
        )
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title("LR Scheduler")
        plt.show()

    return keras.callbacks.LearningRateScheduler(lrfn, verbose=False)  # Create


def create_model(CFG):
    # Define input layers
    img_input = keras.Input(shape=(*CFG.image_size, 3), name="images")
    feat_input = keras.Input(shape=(len(FEATURE_COLS),), name="features")

    # Branch for image input
    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(CFG.preset)
    x1 = backbone(img_input)
    x1 = keras.layers.GlobalAveragePooling2D()(x1)
    x1 = keras.layers.Dropout(0.2)(x1)

    # Branch for tabular/feature input
    x2 = keras.layers.Dense(326, activation="selu")(feat_input)
    x2 = keras.layers.Dense(64, activation="selu")(x2)
    x2 = keras.layers.Dropout(0.1)(x2)

    # Concatenate both branches
    concat = keras.layers.Concatenate()([x1, x2])

    # Output layer
    out1 = keras.layers.Dense(CFG.num_classes, activation=None, name="head")(concat)
    out2 = keras.layers.Dense(CFG.aux_num_classes, activation="relu", name="aux_head")(
        concat
    )
    out = {"head": out1, "aux_head": out2}

    # Build model
    model = keras.models.Model([img_input, feat_input], out)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "head": R2Loss(use_mask=False),
            "aux_head": R2Loss(
                use_mask=True
            ),  # use_mask to ignore `NaN` auxiliary labels
        },
        loss_weights={"head": 1.0, "aux_head": 0.3},  # more weight to main task
        metrics={"head": R2Metric()},  # evaluation metric only on main task
    )
    return model


if "__name__" == "__name__":
    keras.utils.set_random_seed(CFG.seed)
    BASE_PATH = "kaggle/input/planttraits2024"

    df = pd.read_csv(f"{BASE_PATH}/train.csv")
    df["image_path"] = f"{BASE_PATH}/train_images/" + df["id"].astype(str) + ".jpeg"
    df.loc[:, CFG.aux_class_names] = df.loc[:, CFG.aux_class_names].fillna(-1)

    # Test
    test_df = pd.read_csv(f"{BASE_PATH}/test.csv")
    test_df["image_path"] = (
        f"{BASE_PATH}/test_images/" + test_df["id"].astype(str) + ".jpeg"
    )
    FEATURE_COLS = test_df.columns[1:-1].tolist()

    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)

    # Create separate bin for each traits
    for i, trait in enumerate(CFG.class_names):

        # Determine the bin edges dynamically based on the distribution of traits
        bin_edges = np.percentile(df[trait], np.linspace(0, 100, CFG.num_folds + 1))
        df[f"bin_{i}"] = np.digitize(df[trait], bin_edges)

    # Concatenate the bins into a final bin
    df["final_bin"] = (
        df[[f"bin_{i}" for i in range(len(CFG.class_names))]]
        .astype(str)
        .agg("".join, axis=1)
    )

    # Perform the stratified split using final bin
    df = df.reset_index(drop=True)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["final_bin"])):
        df.loc[valid_idx, "fold"] = fold
    # Sample from full data
    sample_df = df.copy()
    train_df = sample_df[sample_df.fold != CFG.fold]
    train_df = train_df.head(10000)
    valid_df = sample_df[sample_df.fold == CFG.fold]
    valid_df = valid_df.head(1000)
    print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[FEATURE_COLS].values)
    valid_features = scaler.transform(valid_df[FEATURE_COLS].values)

    # Train
    train_paths = train_df.image_path.values
    train_labels = train_df[CFG.class_names].values
    train_aux_labels = train_df[CFG.aux_class_names].values
    train_ds = build_dataset(
        train_paths,
        train_features,
        CFG.seed,
        CFG.image_size,
        CFG.num_classes,
        CFG.aux_num_classes,
        train_labels,
        train_aux_labels,
        batch_size=CFG.batch_size,
        repeat=True,
        shuffle=True,
        augment=True,
        cache=False,
    )

    # Valid
    valid_paths = valid_df.image_path.values
    valid_labels = valid_df[CFG.class_names].values
    valid_aux_labels = valid_df[CFG.aux_class_names].values
    valid_ds = build_dataset(
        valid_paths,
        valid_features,
        CFG.seed,
        CFG.image_size,
        CFG.num_classes,
        CFG.aux_num_classes,
        valid_labels,
        valid_aux_labels,
        batch_size=CFG.batch_size,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
    # print(train_df[FEATURE_COLS].values)
    print(train_features.shape)
    inps, tars = next(iter(train_ds))
    imgs = inps["images"]
    feats = inps["features"]
    # images are scaled, devided by 255.0
    print(imgs.shape)
    print(feats.shape)
    # class + aux_class
    print(len(tars), tars[0].shape, tars[1].shape)

    model = create_model(CFG)

    # Model Summary
    model.summary()

    ckpt_cb = keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_head_r2",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
    )

    lr_cb = get_lr_callback(CFG.batch_size, mode=CFG.lr_mode, plot=True)

    # start training
    history = model.fit(
        train_ds,
        epochs=CFG.epochs,
        callbacks=[lr_cb, ckpt_cb],
        steps_per_epoch=len(train_df) // CFG.batch_size,
        validation_data=valid_ds,
        verbose=CFG.verbose,
    )
