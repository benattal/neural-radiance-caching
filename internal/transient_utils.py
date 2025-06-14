import pdb 
# import subprocess

# # Source ~/.bashrc
# subprocess.run(["source", "~/.bashrc"], shell=True, executable="/bin/bash")
# # Activate conda environment
# subprocess.run(["conda", "activate", "yobo"], shell=True, executable="/bin/bash")
# # Export PATH variable
# subprocess.run(['export', 'PATH="/usr/local/cuda/bin:$PATH"'], shell=True, executable="/bin/bash")
# # Export LD_LIBRARY_PATH variable
# subprocess.run(['export', 'LD_LIBRARY_PATH="/usr/local/cuda/lib64/:/usr/local/cuda/targets/x86_64-linux/lib:/scratch/ondemand28/anagh/miniconda3/envs/yobo/lib64:$LD_LIBRARY_PATH"'], shell=True, executable="/bin/bash")


# pdb.set_trace()
import numpy as np 
import h5py 
import matplotlib.pyplot as plt 
import pdb 
import os 
import imageio 
# from align_data import depth2dist, get_rays
# from align_data import read_json_poses
import torch 
import numpy as np 
import os 
import json 
# from read_depth import read_array
import tqdm
import matplotlib.pyplot as plt 
import re
import plotly.graph_objs as go
import pdb
from glob import glob
import scipy 
# import jax
# import jax.numpy as jnp
import cv2

def get_rays(K, c2w, camera_rays=False):
    x, y = torch.meshgrid(
        torch.arange(512),
        torch.arange(512),
        indexing="xy",
    )

    x = x.flatten()
    y = y.flatten()

    dirs = torch.nn.functional.pad(
        torch.stack(
            [
                (x - K[0, 2] + 0.5) / K[0, 0],
                (y - K[1, 2] + 0.5) / K[1, 1] * -1.0,
            ], dim=-1), (0, 1), value=-1.0)

    if camera_rays:
        return dirs.reshape(512, 512, 3)

    directions = (dirs[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    origins = torch.broadcast_to(c2w[None, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )

    origins = torch.reshape(origins, (512, 512, 3))
    viewdirs = torch.reshape(viewdirs, (512, 512, 3))

    return origins, viewdirs


def save_video_from_h5(h5_path, output_filename='output_video.mp4', fps=30):
    # Open the HDF5 file
    data = read_h5(h5_path)        

    # Extract dimensions
    H, W, T, _ = data.shape
    
    # Determine output path
    output_path = os.path.join(os.path.dirname(h5_path), output_filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    pdb.set_trace()
    # Write each frame to the video
    mx_val = data.max()
    for t in tqdm.tqdm(range(T)):
        frame = (data[:, :, t, :3]/mx_val)**(1/2.2)
        out.write((frame*255).astype('uint8'))  # Ensure data type is uint8 for video

    # Release the video writer
    out.release()
    print(f'Video saved at {output_path}')

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def save_imgs():
    tran_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/peppers/test"
    for file in os.listdir(tran_path):
        # if "2_12" in file:
        tran = read_h5(os.path.join(tran_path, file))
        img = tran[..., :3].sum(-2)
        img = (img/img.max())**(1/2.2)
        imageio.imwrite(os.path.join("/scratch/ondemand28/anagh/active-yobo/data/yobo/peppers/imgs", file[:-3]+".png"), (img*255).astype(np.uint8))


def read_json(fname):
    with open(
            os.path.join(fname), "r"
    ) as fp:
        meta = json.load(fp)
    # camtoworlds = []

    # for i in range(len(meta["frames"])):
    #     frame = meta["frames"][i]
    #     camtoworlds.append(frame["transform_matrix"])

    # camtoworlds = np.stack(camtoworlds, axis=0)

    return meta["frames"]


def plot_pcs(pcs):

    fig = go.Figure()
    for pc, color in zip(pcs, ['blue', 'red']):

        trace1 = go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', marker=dict(size=1, color=pc[:, 2], colorscale='Viridis'))
        fig.add_trace(trace1)

    fig.update_layout(
        scene_camera=dict(up=dict(x=-1., y=0, z=0),
                          eye=dict(x=-1, y=1, z=-1)),
    )
    fig.layout.scene.camera.projection.type = "orthographic"

    fig.write_html("./statue_depth.html")
    fig.show()
 

def get_total_distances():
    path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/training_files"
    h5_files = sorted(glob(os.path.join(path,'*.h5')))
    outpath = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/total_distances"

    pulse = np.load("/scratch/ondemand28/anagh/active-yobo/data/yobo/pulse.npy")

    for file in tqdm.tqdm(h5_files):
        transient = read_h5(file).squeeze()
        lm = scipy.signal.correlate2d(transient.reshape(-1, 4096)[:, :-1500], pulse[None], 'same')
        distance = lm.argmax(-1).reshape(512, 512)*0.010595340387968393

        np.save(os.path.join(outpath, file.split("/")[-1][:-2]+"npy"), distance)

        plt.imshow(distance)
        plt.savefig(os.path.join(outpath, file.split("/")[-1][:-2]+"png"))
        plt.colorbar()
        plt.clf()
        


    
    # lm = np.correlate(normalized_transient, normalized_pulse, 'same')

def plot_from_depth():
    # K =  torch.Tensor(np.array([[491.8552373734604, 0, 258.14699999999999], [0, 486.26909966170638, 280.18729999999999], [0, 0, 1.0000]]))

    # K =  torch.Tensor(np.array([[496.74984534157818, 0, 258.14699999999999], [0, 497.70737507328056, 280.18729999999999], [0, 0, 1.0000]]))
    # K =  torch.Tensor(np.array([[423.3975, 0, 266.3641], [0, 424.2648, 171.5908], [0, 0, 1.0000]]))
    # pdb.set_trace()
    K =  torch.Tensor(np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]]))


    json_file = '/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/training_files/transforms_train.json'
    depths_path = "/scratch/ondemand28/anagh/multiview_lif/results_captured_1M/statue_14_02_relativistic/render_warped_08-22_02:48:07"
    frames= read_json(json_file)
    
    # camera id
    colmap_pcs = []

    for camid in tqdm.tqdm(range(len(frames))):
        frame = frames[camid]
        camtoworld = frame["transform_matrix"]
        camtoworld = torch.from_numpy(np.array(camtoworld))

        # camtoworld[:2] *= -1
        # camtoworld[:2, -1] *= -1

        origins, viewdirs = get_rays(K, camtoworld)

        filename = frame["filepath"].split("/")[-1][:-3]+"_depth.npy"
        
        # delimiters = "_", "."
        # regex_pattern = '|'.join(map(re.escape, delimiters))
        # num = int(re.split(regex_pattern, filename)[-2]) + (int(re.split(regex_pattern, filename)[-3])-1)*30 -1
        # depth = torch.from_numpy(read_array(f"{depths_path}/img_{str(num).zfill(4)+'.png'}.geometric.bin"))[..., None]

        # for colmap
        # depth = torch.from_numpy(read_array(f"{depths_path}/{filename}.geometric.bin"))[..., None]
        # depth = depth2dist(depth, K)
        # pdb.set_trace()
        depth = torch.from_numpy(np.load(os.path.join(depths_path, filename)))


        colmap_pc = (depth * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 60).flatten()*(depth >0).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        # colmap_pc = torch.concat([colmap_pc, origins.reshape(-1, 3)])
        
        # viewdirs[..., 1]*= -1
        # viewdirs[..., 2]*= -1

        
        # colmap_pc = (viewdirs[256, 256]*np.linspace(0, 3)[:, None]) + origins[0, 0]
        # colmap_pc = colmap_pc.reshape(-1, 3)
        colmap_pcs.append(colmap_pc)
        
    colmap_pc = torch.concat(colmap_pcs, dim=0)
    plot_pcs([colmap_pc[::1000], ])


def plot_solved_depths():
    K =  torch.Tensor(np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]]))


    json_file = '/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/training_files/transforms_train.json'
    total_distances_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/total_distances"
    frames= read_json(json_file)
    
    x = torch.Tensor([11.0766,  0.9688, -3.6915,  0.1594])
    shift = x[0]
    light_source_pos = x[1:]

    # camera id
    colmap_pcs = []

    for camid in tqdm.tqdm(range(len(frames))):
        frame = frames[camid]
        camtoworld = frame["transform_matrix"]
        camtoworld = torch.from_numpy(np.array(camtoworld))

        # camtoworld[:2] *= -1
        # camtoworld[:2, -1] *= -1

        origins, viewdirs = get_rays(K, camtoworld)

        filename = frame["filepath"].split("/")[-1][:-3]+".npy"
        
        # delimiters = "_", "."
        # regex_pattern = '|'.join(map(re.escape, delimiters))
        # num = int(re.split(regex_pattern, filename)[-2]) + (int(re.split(regex_pattern, filename)[-3])-1)*30 -1
        # depth = torch.from_numpy(read_array(f"{depths_path}/img_{str(num).zfill(4)+'.png'}.geometric.bin"))[..., None]

        # for colmap
        # depth = torch.from_numpy(read_array(f"{depths_path}/{filename}.geometric.bin"))[..., None]
        # depth = depth2dist(depth, K)
        # pdb.set_trace()
        total_dist = torch.from_numpy(np.load(os.path.join(total_distances_path, filename)))

        light_source_distance = 0
        depth = total_dist - shift


        colmap_pc = (depth * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 60).flatten()*(depth >0).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        # colmap_pc = torch.concat([colmap_pc, origins.reshape(-1, 3)])
        
        # viewdirs[..., 1]*= -1
        # viewdirs[..., 2]*= -1

        
        # colmap_pc = (viewdirs[256, 256]*np.linspace(0, 3)[:, None]) + origins[0, 0]
        # colmap_pc = colmap_pc.reshape(-1, 3)
        colmap_pcs.append(colmap_pc)
        
    colmap_pc = torch.concat(colmap_pcs, dim=0)
    plot_pcs([colmap_pc[::1000], ])


def optimize_shift_light_pos(scene_points, depth, total_distance, lam=0.005, lam2 = 5):

    def objective(x):
        shift = x[0]
        light_pos = x[1:]
        total_error = total_distance - (depth + shift + ((scene_points-light_pos[None])**2).sum(-1)**(1/2))

        total_error = torch.sum(total_error**2)/2
        print(x)
        # total_error = torch.abs(total_error).sum()
        return total_error
    
    x = torch.Tensor([9.9064,  0.9608, -4.8817,  0.3775]).float()
    x.requires_grad = True
    optim = torch.optim.LBFGS([x], lr=0.01)

    def closure():
        optim.zero_grad(set_to_none=True)
        err = objective(x)
        err.backward()
        return err

    for i in tqdm.tqdm(range(50000)):
        optim.zero_grad()
        optim.step(closure)
        # print(x)

    print(f"Solution is: {x}")
    return x.detach()

def get_patch(origins, viewdirs,total_distance,  depth, patch):
    o = origins[patch[0]:patch[1], patch[2]:patch[3]]
    d = depth[patch[0]:patch[1], patch[2]:patch[3]]
    vdir = viewdirs[patch[0]:patch[1], patch[2]:patch[3]]
    tod_dist = total_distance[patch[0]:patch[1], patch[2]:patch[3]]

    sce_pon = (d[..., None] * vdir + o)
    return sce_pon, d, tod_dist

def solve_shift_light_source_one_view(scene="statue"):
    """
    patches are indexed by (i1, i2, i3, i4), i1 is (i1, i2) is width range, (i3, i4) is height range
    """
    K =  torch.Tensor(np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]]))

    if scene=="statue":
        scene_path = "data/yobo/statue"
        views = ["1_47", "2_56", "3_60"]
        all_view_patches = {"1_47":[(50, 150, 300, 400), (360, 410, 290, 400), (200, 300, 0, 10)], 
                "2_56":[(60, 160, 360, 460), (390, 450, 110, 240)], 
                "3_60": [ (360, 440, 100, 250), (200, 220, 145, 165), (65, 160, 400, 470)]}

    json_file = f'{scene_path}/training_files/transforms_train.json'
    frames = read_json(json_file)

    all_scene_points = []
    depths = []
    total_distances = []
    for view in views:
        total_distance = torch.Tensor(np.load(f"{scene_path}/total_distances/{view}.npy"))
        depth = torch.Tensor(np.load(f"{scene_path}/rendered_depth/{view}_depth.npy"))

        depth = torch.Tensor(scipy.signal.medfilt2d(depth.squeeze(), kernel_size=3))
        total_distance = torch.Tensor(scipy.signal.medfilt2d(total_distance, kernel_size=3))
        frame = [x for x in frames if view in x["filepath"]][0]

        camtoworld = torch.from_numpy(np.array(frame["transform_matrix"]))
        origins, viewdirs = get_rays(K, camtoworld)

        patches = all_view_patches[view]
        for patch in patches:
            sp, d, td = get_patch(origins, viewdirs,total_distance,  depth, patch)
            all_scene_points.append(sp.reshape(-1, 3))
            depths.append(d.reshape(-1))
            total_distances.append(td.reshape(-1))
    
    all_scene_points = torch.concat(all_scene_points,0)
    depths = torch.concat(depths,0)
    total_distances = torch.concat(total_distances,0)

    torch.manual_seed(0)

    x = optimize_shift_light_pos(all_scene_points, depths, total_distances)

    print(x)

def shift_map_coordinates(transient_indirect, shift_val, exposure_time, n_bins):
  bins_move = bins_move/exposure_time
  x_dim = transient_indirect.shape[0]
  x = (jnp.arange(x_dim))
  y = jnp.arange(n_bins)
  z = jnp.arange(3)
  X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
  Y = Y + shift_val[:, None, None]
  indices = jnp.stack([X, Y, Z])
  indirect = jax.scipy.ndimage.map_coordinates(transient_indirect, indices, 1, mode="constant")
  return indirect 


def downsample_shift_transients():
    device = "cuda:5"
    training_files_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/training_files"
    shift_value = 11.0766
    exposure_time = 0.010595340387968393
    downsample_factor = 4
    downsample_times = np.log2(downsample_factor).astype(np.uint8)
    save_path = os.path.join("/scratch/ondemand28/anagh/active-yobo/data/yobo/statue", f"train")

    for file in tqdm.tqdm(os.listdir(training_files_path)):
        if file[0] in ["1", "2", "3"]:
            filepath = os.path.join(training_files_path, file)
            transient = read_h5(filepath).squeeze()[..., :3000]
            for i in range(downsample_times):
                transient = transient[::2, ::2]
            
            transient = torch.tensor(transient).to(device).reshape(transient.shape[0]**2, 3000, 1)
            transient = shift_transient_grid_sample_3d(transient, shift_val=shift_value, exposure_time=exposure_time, n_bins=3000).reshape(128, 128, 3000).cpu()
            np.save(os.path.join(save_path, file[:-3]+".npy"), transient)


def shift_transient_grid_sample_3d(transient, shift_val, exposure_time, n_bins):
    x_dim = transient.shape[0]
    bins_move = shift_val/exposure_time
    if x_dim%2 == 0:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2+0.5)/(x_dim//2-0.5)
    else:
        x = (torch.arange(x_dim, device=transient.device)-x_dim//2)/(x_dim//2)

    if x_dim == 1:
        x = torch.zeros_like(x)
        
    z = torch.arange(n_bins, device=transient.device).float()
    X, Z = torch.meshgrid(x, z, indexing="ij")
    Z = Z + bins_move
    Z[Z<0] = n_bins+1
    Z = (Z-n_bins//2+0.5)/(n_bins//2-0.5)
    grid = torch.stack((Z, X), dim=-1)[None, ...]
    shifted_transient = torch.nn.functional.grid_sample(transient.permute(2, 0, 1)[None], grid, align_corners=True).squeeze(0).permute(1, 2, 0)
    return shifted_transient


if __name__=="__main__":
    # save_imgs()
    save_video_from_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/peppers/test/test_0_00.h5")
    exit()
    transient_img1 = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/cornell_flash_dense/train/0_14.h5")
    # transient_img2 = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/cornell_flash_dense/train/2_28.h5")
    pdb.set_trace()
    plt.subplot(1, 2, 1)
    t1 = transient_img1[..., :3].sum(-2).astype(np.float32)
    plt.imshow((t1/t1.max())**(1/2.2))    
    plt.plot(100, 60, 'ro')  # Note: (x, y) in plot corresponds to (column, row)

    
    plt.subplot(1, 2, 2)
    # t2 = transient_img2[..., :3].sum(-2).astype(np.float32)
    # plt.imshow((t2/t2.max())**(1/2.2))
    plt.plot(transient_img1[60, 100, :, 0])
    
    plt.savefig("disp")