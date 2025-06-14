"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from datetime import datetime
import random
from typing import Optional, Sequence
import ast
import configargparse
import os
from internal.datasets import read_h5

import numpy as np
import h5py
import json

def bundle_rays(pathToH5s, outputPath, trainJsonPath):
    with open(trainJsonPath, "r") as fp:
        meta = json.load(fp)
        train_fnames = []
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = frame["file_path"][:-2]+"h5"
            train_fnames.append(fname)

    frames = read_h5(os.path.join(pathToH5s, train_fnames[0]))
    w = frames.shape[0]
    h = frames.shape[1]
    bins = frames.shape[2]
    
    x = np.linspace(0, h-1, h)
    y = np.linspace(0, w-1, w)   
    X, Y = np.meshgrid(x, y)

    if len(frames.shape) == 4:
        channels = 3
    else:
        channels = 1
    num_train_files = len(train_fnames)
    
    data_array = np.zeros((w*h*num_train_files, bins, channels), dtype=np.float32)
    x_array = np.zeros(w*h*num_train_files)
    y_array = np.zeros(w*h*num_train_files)
    file_prefix_array = np.zeros(w*h*num_train_files)

    for ind, file in enumerate(train_fnames):
        print("Opening: " + file)
        frames = read_h5(os.path.join(pathToH5s, file))
        frames = frames.reshape(-1, frames.shape[2], frames.shape[3])
        
        data_array[ind*w*h:(ind+1)*w*h] = frames[..., :3]
        x_array[ind*w*h:(ind+1)*w*h] = X.flatten()
        y_array[ind*w*h:(ind+1)*w*h] = Y.flatten()
        file_prefix_array[ind*w*h:(ind+1)*w*h] = ind
    
    p = np.random.permutation(data_array.shape[0])
    data_array = data_array[p]
    x_array = x_array[p]
    y_array = y_array[p]
    file_prefix_array = file_prefix_array[p]


    print("Outputting to files")
    file = h5py.File(os.path.join(outputPath, "samples.h5"), 'w')
    dataset = file.create_dataset(
    "dataset", data_array.shape, dtype='f', data=data_array
    )
    file.close()

    file = h5py.File(os.path.join(outputPath, "x.h5"), 'w')
    dataset = file.create_dataset(
        "dataset", x_array.shape, dtype='f', data=x_array
    )
    file.close()

    file = h5py.File(os.path.join(outputPath, "y.h5"), 'w')
    dataset = file.create_dataset(
        "dataset", y_array.shape, dtype='f', data=y_array
    )
    file.close()
    file = h5py.File(os.path.join(outputPath, "file_indices.h5"), 'w')
    dataset = file.create_dataset(
        "dataset", file_prefix_array.shape, dtype='f', data=file_prefix_array
    )
    file.close()

def main():
    parser = configargparse.ArgParser(description="Run bundle_rays with specified arguments")
    parser.add_argument('--pathToH5s', type=str, required=True, help='Path to the directory containing H5 files')
    parser.add_argument('--outputPath', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--trainJsonPath', type=str, required=True, help='Path to the training JSON file')

    args = parser.parse_args()

    bundle_rays(args.pathToH5s, args.outputPath, args.trainJsonPath)

if __name__ == "__main__":
    main()