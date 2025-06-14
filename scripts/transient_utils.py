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

from matplotlib import cm

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
from PIL import Image
import mat73
import numpy as np
import matplotlib.pyplot as plt

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


def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

def save_imgs():
    dest = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/kitchen"
    tran_path = f"{dest}/train"
    for file in os.listdir(tran_path):
        print(file)
        if int(file[:-3]) >99:
            if "h5" in file:
                tran = read_h5(os.path.join(tran_path, file))
                img = np.clip(tran[..., :3]/20, 0, 1).sum(-2)
                img = np.clip((img/6)**(1/2.2), 0, 1)
                imageio.imwrite(os.path.join(f"{dest}/imgs", file[:-3]+".png"), (img*255).astype(np.uint8))


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
        scene=dict(
            xaxis=dict(range=[-2.5, 2.5]),  # Adjust x-axis range
            yaxis=dict(range=[-2.5, 2.5]),  # Adjust y-axis range
            zaxis=dict(range=[-2.5, 2.5])   # Adjust z-axis range
        ),

    )
    fig.layout.scene.camera.projection.type = "orthographic"

    fig.write_html("./smth_spheres.html")
    fig.show()
 
def save_video_from_h5(h5_path, output_filename='output_video.mp4', fps=30, bkgd=True):
    # Open the HDF5 file
    import cv2
    # data = read_h5(h5_path)        
    data = h5_path
    # Extract dimensions
    H, W, T, _ = data.shape
    
    # Determine output path
    # output_path = os.path.join(os.path.dirname(h5_path), h5_path.split("/")[-1][:-2]+"mp4")
    output_path = output_filename
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    # Write each frame to the video
    mx_val = data[..., :3].max()
    bkg = data[..., :3].sum(-2)
    bkg = (bkg/12)**(1/2.2)
    # data = data/mx_val
    bkg = (imageio.imread("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_material_light_from_scratch_resample_eval/save/color_cache0/0007.png")/255).astype(np.float32)[..., None]
    for t in tqdm.tqdm(range(T)):
        frame = (data[:, :, t, :3]*2)**(1/2)
        if bkgd:
            frame = 0.9*frame + 0.1*bkg
        frame = np.clip(frame, 0, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write((frame*255).astype('uint8'))  # Ensure data type is uint8 for video

    # Release the video writer
    out.release()
    print(f'Video saved at {output_path}')


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


def plot_from_depth_not_multiview():
    
    x = [17.26664936, -0.73867326,  -0.05389526, 0.16914815]
    FocalLength = [450.0188, 450.0733]
    PrincipalPoint = [261.6281, 243.4146]
    K = np.array([[FocalLength[0], 0, PrincipalPoint[0]], 
                  [0, FocalLength[1], PrincipalPoint[1]],
                  [0, 0, 1]])
    
    light_source_pos_wrt_cam = torch.Tensor(x[1:]).float()
    # K =  torch.Tensor(np.array([[491.8552373734604, 0, 258.14699999999999], [0, 486.26909966170638, 280.18729999999999], [0, 0, 1.0000]]))

    # K =  torch.Tensor(np.array([[496.74984534157818, 0, 258.14699999999999], [0, 497.70737507328056, 280.18729999999999], [0, 0, 1.0000]]))
    # K =  torch.Tensor(np.array([[423.3975, 0, 266.3641], [0, 424.2648, 171.5908], [0, 0, 1.0000]]))
    # pdb.set_trace()
    # K =  torch.Tensor(np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]]))


    json_file = '/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/globe_10_30/transforms_train.json'
    # depths_path = "/scratch/ondemand28/anagh/multiview_lif/results_captured_1M/statue_14_02_relativistic/render_warped_08-22_02:48:07"
    frames= read_json(json_file)
    device = "cpu"
    # # camera id
    colmap_pcs = []
    scale = 0.010376310322275158
    light_source_pos_wrt_cam = torch.concat([light_source_pos_wrt_cam, torch.tensor([1])]).double()

    for camid in tqdm.tqdm(range(len(frames)-53)):
        frame = frames[camid]
        camtoworld = frame["transform_matrix"]
        camtoworld = torch.from_numpy(np.array(camtoworld))
        
        total_dist = read_h5(f"data/yobo/final_captured/globe_10_30/{frame['file_path']}")
        img = total_dist.sum(-1)
        total_dist = np.argmax(total_dist, -1)*scale - x[0]
        total_dist = np.clip(total_dist, 0, None)
        total_dist = torch.from_numpy(total_dist)[..., None]
        # camtoworld[:2] *= -1
        # camtoworld[:2, -1] *= -
        # camtoworld = torch.eye(4)
        # pdb.set_trace()
        light_source_pos = (camtoworld @ light_source_pos_wrt_cam)[:3]

        origins, viewdirs = get_rays(K, camtoworld)
        # total_dist = np.clip(total_dist, 0, 1)
        depth = get_depth_from_distance_eq(total_dist.to(device).squeeze(), light_source_pos.to(device), viewdirs.to(device), origins.to(device))[..., None]

        # filename = frame["filepath"].split("/")[-1][:-3]+"_depth.npy"
        
        # delimiters = "_", "."
        # regex_pattern = '|'.join(map(re.escape, delimiters))
        # num = int(re.split(regex_pattern, filename)[-2]) + (int(re.split(regex_pattern, filename)[-3])-1)*30 -1
        # depth = torch.from_numpy(read_array(f"{depths_path}/img_{str(num).zfill(4)+'.png'}.geometric.bin"))[..., None]

        # for colmap
        # depth = torch.from_numpy(read_array(f"{depths_path}/{filename}.geometric.bin"))[..., None]
        # depth = depth2dist(depth, K)
        # pdb.set_trace()
        # depth = torch.from_numpy(np.load(os.path.join(depths_path, filename)))

        # pdb.set_trace()

        colmap_pc = (depth * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 20).flatten()*(depth >0).flatten()*(img>30).flatten()
        colmap_pc = colmap_pc[colmap_mask, :]
        # colmap_pc = torch.concat([colmap_pc, origins.reshape(-1, 3)])
        
        # viewdirs[..., 1]*= -1
        # viewdirs[..., 2]*= -1

        
        # colmap_pc = (viewdirs[256, 256]*np.linspace(0, 3)[:, None]) + origins[0, 0]
        # colmap_pc = colmap_pc.reshape(-1, 3)
        colmap_pcs.append(colmap_pc)
            
    colmap_pc = torch.concat(colmap_pcs, dim=0)
    pdb.set_trace()
    plot_pcs([colmap_pc[::100], ])
    
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

def get_depth_from_distance_optim(total_dist, light_pos, viewdirs, origins):

    def objective(x):
        x = torch.abs(x)
        total_error = total_dist - (x + ((viewdirs*x[..., None] + origins -light_pos[None, None])**2).sum(-1)**(1/2))

        total_error = torch.sum(total_error**2)/2
        # print(total_error)
        return total_error
    
    x = torch.zeros_like(total_dist).float()+1
    x = x.to(total_dist.device)
    x.requires_grad = True
    optim = torch.optim.LBFGS([x], lr=0.1)

    def closure():
        optim.zero_grad(set_to_none=True)
        err = objective(x)
        err.backward()
        return err

    for i in tqdm.tqdm(range(100)):
        optim.zero_grad()
        optim.step(closure)
        # print(x)

    print(f"Solution is: {x}")
    return torch.abs(x.detach())


def get_depth_from_distance_eq(total_dist, light_pos, viewdirs, origins):
    b = light_pos[None, None] - origins
    b_dotb = (b**2).sum(-1)
    v_dotb = (viewdirs*b).sum(-1)
    d = (b_dotb - total_dist**2)/(2*v_dotb - 2*total_dist)
    d = torch.clip(d, 0, None)
    return d

def plot_solved_depths(near=8, far=1):
    K =  torch.Tensor(np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]]))
    device = "cuda:1"
    json_file = '/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/training_files/transforms_train.json'
    total_distances_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/total_distances"
    images_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/rendered_depth"
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
        
        img = Image.open(f"{images_path}/{filename[:-4]}_ours.png")
        img = np.array(img)
        # delimiters = "_", "."
        # regex_pattern = '|'.join(map(re.escape, delimiters))
        # num = int(re.split(regex_pattern, filename)[-2]) + (int(re.split(regex_pattern, filename)[-3])-1)*30 -1
        # depth = torch.from_numpy(read_array(f"{depths_path}/img_{str(num).zfill(4)+'.png'}.geometric.bin"))[..., None]

        # for colmap
        # depth = torch.from_numpy(read_array(f"{depths_path}/{filename}.geometric.bin"))[..., None]
        # depth = depth2dist(depth, K)
        # pdb.set_trace()
        total_dist = torch.from_numpy(np.load(os.path.join(total_distances_path, filename)))
        total_dist = total_dist - shift

        # depth = get_depth_from_distance(total_dist.to(device), light_source_pos.to(device), viewdirs.to(device), origins.to(device))[..., None]
        depth = get_depth_from_distance_eq(total_dist.to(device), light_source_pos.to(device), viewdirs.to(device), origins.to(device))[..., None]

        depth = depth.to("cpu")

        colmap_pc = (depth * viewdirs + origins).reshape(-1, 3)
        colmap_mask = (depth < 8).flatten()*(depth >1).flatten()*(img>5).flatten()
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

def cast_uint16():
    orig_paths = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/cornell/train"
    new_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/cornell_uint16/train"
    scale = 2000
    uint_scale = 2**16 -1
    for file in os.listdir(orig_paths):
        if "h5" in file:
            transient = read_h5(os.path.join(orig_paths, file))
            print(transient.max())
            transient = ((transient/scale)*uint_scale).astype(np.uint16)
            save_h5(os.path.join(new_path, file), transient)

def save_h5(path, data_array):
    file = h5py.File(path, 'w')
    dataset = file.create_dataset(
    "data", data_array.shape, dtype='f', data=data_array
    )
    file.close()

def bundle_rays_cap(pathToH5s, outputPath, trainJsonPath, offset=1000):

    with open(trainJsonPath, "r") as fp:
        meta = json.load(fp)
        train_fnames = []
        for i in range(len(meta["frames"])):
            frame = meta["frames"][i]
            fname = frame["file_path"][:-2].split("/")[-1]+"mat"
            train_fnames.append(fname)

    # frames = read_h5_dataset(os.path.join(pathToH5s, train_fnames[0]))
    frames = mat73.loadmat(os.path.join(pathToH5s, train_fnames[0]))["transient"].transpose(1, 2, 0)
    w = frames.shape[0]
    h = frames.shape[1]
    # bins = frames.shape[2]
    bins = 3000
    
    x = np.linspace(0, h-1, h)
    y = np.linspace(0, w-1, w)   
    X, Y = np.meshgrid(x, y)

    # if len(frames.shape) == 4:
    #     channels = 3
    # else:
    channels = 1
    num_train_files = len(train_fnames)
    
    data_array = np.zeros((w*h*num_train_files, bins, channels), dtype=np.float32)
    x_array = np.zeros(w*h*num_train_files)
    y_array = np.zeros(w*h*num_train_files)
    file_prefix_array = np.zeros(w*h*num_train_files)

    for ind, file in enumerate(train_fnames):
        print("Opening: " + file)
        # frames = read_h5_dataset(os.path.join(pathToH5s, file))
        frames = mat73.loadmat(os.path.join(pathToH5s, file))["transient"].transpose(1, 2, 0)
        frames = frames.reshape(-1, frames.shape[2])
        
        data_array[ind*w*h:(ind+1)*w*h] = frames[...,offset:offset+bins, None]
        # del frames
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


def weighted_percentile(x, w, ps, assume_sorted=False, xnp=np):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = xnp.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = xnp.cumsum(w)
    return xnp.interp(xnp.array(ps) * (acc_w[-1] / 100), acc_w, x)

def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
    xnp=np,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    lo_auto, hi_auto = weighted_percentile(value, weight, [50 - percentile / 2, 50 + percentile / 2])

    # If `lo` or `hi` are None, use the automatically-computed bounds above.
    eps = xnp.finfo(xnp.float32).eps
    # lo = lo or (lo_auto - eps)
    # hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = xnp.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = xnp.clip((value - xnp.minimum(lo, hi)) / xnp.abs(hi - lo), 0, 1)
    value = xnp.nan_to_num(value)

    if colormap:
        colorized = colormap(value)[Ellipsis, :3]
    else:
        if value.shape[-1] != 3:
            raise ValueError(f"value must have 3 channels but has {value.shape[-1]}")
        colorized = value

    return matte(colorized, weight) if matte_background else colorized

def matte(vis, acc, dark=0.8, light=1.0, width=8, xnp=np):
    return vis + (1.0 - acc[..., None])

    # """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    # bg_mask = xnp.logical_xor(
    #     (xnp.arange(acc.shape[-2]) % (2 * width) // width)[Ellipsis, :, None],
    #     (xnp.arange(acc.shape[-1]) % (2 * width) // width)[Ellipsis, None, :],
    # )
    # bg = xnp.where(bg_mask, light, dark)
    # return vis * acc[Ellipsis, None] + (bg * (1 - acc))[Ellipsis, None]

def lm_depth():
    from scipy import signal 
    pulse = np.load("data/yobo/pulse.npy")
    transient = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/spheres_10_27/train_efficient/1_60.h5")
    # transient = transient.reshape(512*512, 4096)
    lm = np.zeros((512, 512, 3000))
    for i in tqdm.tqdm(range(512)):
        lm[i] = signal.correlate2d(transient[i, :, :3000], pulse[None], 'same') 
    
    lm = np.argmax(lm, -1)
    plt.subplot(1, 2, 1); plt.imshow(np.clip(transient.sum(-1)/1000, 0, 1)**(1/2.2)); plt.subplot(1, 2, 2); plt.imshow(np.argmax(lm, -1), vmin=1800, vmax=3000); plt.colorbar(); plt.savefig("disp")
    print("hello")
    #np.save("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/spheres_10_27/train_efficient/1_60_depth.npy", lm)

    
    
def vis_depth_fwp():
    imgs = ["1_02", "1_05", "2_50", "3_40", "3_56"]
    for img in imgs:
        depth = np.load(f"/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/rendered_depth/{img}_depth.npy").squeeze()[:450, :450]
        depth = np.clip(depth, 2, None)
        # depth = depth/depth.max()
        p = 99.0
        lo, hi = weighted_percentile(depth, np.ones_like(depth), [50 - p / 2, 50 + p / 2])
        depth_curve_fn = lambda x: -np.log(x + np.finfo(np.float32).eps)

        def vis_fn(x):
            return visualize_cmap(np.array(x), np.ones_like(depth)-np.finfo(np.float32).eps, cm.get_cmap("turbo"), lo=lo, hi=hi, curve_fn=depth_curve_fn, xnp=np)

        depth_new = vis_fn(depth)
        imageio.imwrite(f"/scratch/ondemand28/anagh/active-yobo/data/yobo/statue/rendered_depth/{img}_depth_img.png", (depth_new*255).astype(np.uint8))


import numpy as np

def direct_tof_to_cw_tof(direct_tof_data, frequency, exposure_time, phase_shifts):
    num_phase_shifts = len(phase_shifts)
    H, W, T, _ = direct_tof_data.shape
    cw_tof_data = np.zeros((H, W, num_phase_shifts, 3), dtype=np.float32)
    c = 299792458
    # Loop over ps
    for p_idx, p in enumerate(phase_shifts):
        
        phase_shift = p
        
        for t in range(T):
            # exposure time is per bin so time_intervals[t]*exposure_time gives total lenght -- should this be divided by 2? 
            time_to_travel = t*exposure_time/c
            phase = 2 * np.pi * frequency * time_to_travel + phase_shift
            cw_tof_data[:, :, p_idx, :] += direct_tof_data[:, :, t, :] * np.cos(phase)

    return cw_tof_data


def make_ctof_dataset():
    scenes = ["peppers", "kitchen"]
    settings = {"kitchen": {"exposure_time":0.02, "freq":30*10**6}, 
    "cornell": {"exposure_time":0.01, "freq":75*10**6}, 
    "peppers": {"exposure_time":0.02, "freq":30*10**6}, 
    "pots": {"exposure_time":0.01, "freq":75*10**6}}

    final_dataset_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated"

    for scene in scenes:
        print(scene)
        for subset in ["train", "test"]:
            h5_paths = os.path.join(final_dataset_path, scene, subset)
            for file in tqdm.tqdm(os.listdir(h5_paths)):
                if "h5" in file:
                    dtof = read_h5(os.path.join(h5_paths, file))[..., :3]
                    ctof = direct_tof_to_cw_tof(dtof, settings[scene]["freq"], settings[scene]["exposure_time"], [0, np.pi/2, np.pi, 3*np.pi/2])
                    target_path = os.path.join(final_dataset_path, scene+"_ctof", subset)
                    os.makedirs(target_path, exist_ok = True)
                    np.save(os.path.join(target_path, file[:-2]+"npy"), ctof)


def find_transient_max(folder):
    maxs = []
    for file in os.listdir(folder):
        if "h5" in file:
            t = read_h5(os.path.join(folder, file))[..., :3]
            print(f"File max: {t.max()} \n")
            maxs.append(t.max())
            
    print(f"total max: {np.array(maxs).max()}")
    
                    
def find_transient_max(folder):
    maxs = []
    for file in os.listdir(folder):
        if "h5" in file:
            t = read_h5(os.path.join(folder, file))[..., :3]
            print(f"File max: {t.max()} \n")
            maxs.append(t.max())
            
    print(f"total max: {np.array(maxs).max()}")

def find_img_max(folder, clip):
    maxs = []
    for file in os.listdir(folder):
        if "h5" in file:
            t = np.clip(read_h5(os.path.join(folder, file))[..., :3], 0, clip).sum(-2)
            print(f"File max: {t.max()} \n")
            maxs.append(t.max())
            
    print(f"total max: {np.array(maxs).max()}")

def rename_runs():
    import subprocess
    checkpoints_dir = "/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic"
    scenes = [
        # "peppers", 
        # "pots", 
        # "kitchen", 
        # "cornell"
        # "kettle", 
        "statue", 
        "spheres", 
        "globe", 

        # "house"
        ]
    types = [
        "tnerf_cache_eval", 
        # "fwp_cache", 
        "cache_eval",
        "material_light_from_scratch_resample_eval"
        ]
    new_add_on = "27_11"
    for scene in scenes:
        for type in types:
            command = f"mv {checkpoints_dir}/{scene}_{type} {checkpoints_dir}/{scene}_{type}_{new_add_on}"
            # print(command)
            subprocess.run(command.split())



def load_mitsuba_params(scene):
    import mitsuba as mi

    test_path = os.path.join(scene, "test")
    xmls_path = os.path.join(scene, "xmls_depth")


    # Initialize Mitsuba with the scalar_rgb variant
    mi.set_variant('scalar_rgb')

    for file in os.listdir(xmls_path):
        if "xml" in file:
            # 1. Load the scene from XML file
            scene = mi.load_file(os.path.join(xmls_path, file))
            
            integrator = mi.load_dict({'type': 'aov',
            'aovs': 'position:position,albedo:albedo,normal:sh_normal'})


            # Render the scene (all AOVs at once)
            rendered_output = mi.render(scene, integrator= integrator, spp=10)
            

            channel_names = []
            channel_names.extend(mi.Integrator.aov_names(integrator))
            bmp_new = mi.Bitmap(rendered_output, channel_names = channel_names)
            new_info = dict(bmp_new.split())

            # Split the AOVs
            positions = np.array(new_info["position"])
            albedo = np.array(new_info["albedo"])
            normal = np.array(new_info["normal"])
            
            # get depth 
            sensor = scene.sensors()[0]
            sensor_transform = sensor.world_transform()
            transform_matrix_np = np.array(sensor_transform.matrix)
            position_np = transform_matrix_np[0:3, 3]  # Extract x, y, z components
            depth = np.linalg.norm(positions - position_np[None, None], axis=-1)
            depth[albedo.sum(-1)==0] = 0 
            
            # get normals 
            ax_flip = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            f = positions - position_np[None, None]
            f = f/np.linalg.norm(f, axis=-1)[..., None]
            mask = (f*normal).sum(-1)<0
            new_normal = (ax_flip[:3, :3]@ normal.reshape(-1, 3).T).reshape(3, 512, 512).transpose(1, 2, 0)
            new_normal[mask] = new_normal[mask]*-1
            
            np.save(os.path.join(test_path, file[:-4]+"_albedo.npy"), albedo)
            np.save(os.path.join(test_path, file[:-4]+"_normals.npy"), new_normal)
            np.save(os.path.join(test_path, file[:-4]+"_depth.npy"), depth)
            
            plt.imshow(albedo)
            plt.savefig(os.path.join(test_path, file[:-4]+"_albedo_img.png")); plt.clf()
            
            plt.imshow(depth)
            plt.savefig(os.path.join(test_path, file[:-4]+"_depth_img.png")); plt.clf()

            plt.imshow(new_normal)
            plt.savefig(os.path.join(test_path, file[:-4]+"_normals_img.png")); plt.clf()



def create_train_test_split():
    scenes_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured"
    scenes = ["house_11_01"]
    FocalLength = [450.0188, 450.0733]
    PrincipalPoint = [261.6281, 243.4146]
    K = np.array([[FocalLength[0], 0, PrincipalPoint[0]], 
                  [0, FocalLength[1], PrincipalPoint[1]],
                  [0, 0, 1]])
    

    if "kettle_10_25" in scenes:
        scene = "kettle_10_25"
        # we captured [22, 47]
        all_views = [f"{i}_{j:02}" for i in range(1, 4) for j in range(22, 48)]
        all_views.sort()
        test_views = all_views[2::6]
        train_views = [ x for x in all_views if x not in test_views]
    
    if "spheres_10_27" in scenes:
        scene = "spheres_10_27"
        # we captured [22, 47]
        
        all_views = [f"{i}_{j:02}" for i in range(1, 4) for j in range(1, 16)] + [f"{i}_{j:02}" for i in range(1, 4) for j in range(54, 61)]
        all_views.sort()
        test_views = all_views[3::6]
        train_views = [ x for x in all_views if x not in test_views]

    if "globe_10_30" in scenes:
        scene = "globe_10_30"
        # we captured [22, 47]
        
        all_views = [f"{i}_{j:02}" for i in range(1, 4) for j in range(1, 16)] + [f"{i}_{j:02}" for i in range(1, 4) for j in range(54, 61)]
        all_views.sort()
        test_views = all_views[3::6]
        train_views = [ x for x in all_views if x not in test_views]

    if "house_11_01" in scenes:
        scene = "house_11_01"
        # we captured [22, 47]
        x_rots = list(range(1, 21)) + list(range(40, 61, 2))
        x_rots2 = list(range(1, 21, 2)) + list(range(40, 61))
        x_rots3 = list(range(1, 21)) + list(range(40, 61, 2))
        all_views = [f"1_{j:02}" for j in x_rots] + [f"2_{j:02}" for j in x_rots2] + [f"3_{j:02}" for j in x_rots3]
        # all_views = [f"{i}_{j:02}" for i in range(1, 4) for j in range(1, 16)] + [f"{i}_{j:02}" for i in range(1, 4) for j in range(54, 61)]
        all_views.sort()
        test_views = all_views[2::7]
        train_views = [ x for x in all_views if x not in test_views]

    print("hello")
    meta = read_json(os.path.join(scenes_path, scene, "transforms_from_colmap.json" ))
    
    train_json = {}
    train_json["camera"] = K.tolist()
    train_json["frames"] = []
    
    test_json = train_json.copy()
    test_json["frames"] = []
    
    for ind, pos in enumerate(meta):
        if pos["file_path"].split("/")[-1][:-3] in train_views:
            train_json["frames"].append(pos)
        elif pos["file_path"].split("/")[-1][:-3] in test_views:
            test_json["frames"].append(pos)

        
        
    train_json = json.dumps(train_json, indent=4)
    with open(os.path.join(scenes_path, scene, "transforms_train.json"), 'w') as f:
        f.write(train_json)
           
    
    test_json = json.dumps(test_json, indent=4)
    with open(os.path.join(scenes_path, scene, "transforms_test.json"), 'w') as f:
        f.write(test_json)

import tqdm
def mat_to_npy(mat_folder, np_folder):
    for file in tqdm.tqdm(os.listdir(mat_folder)):
        if "mat" in file:
            transient = mat73.loadmat(os.path.join(mat_folder, file))["transient"].transpose(1, 2, 0)
            # print(transient.shape)
            # np.save(os.path.join(np_folder, file[:-3]+"npy"), transient)
        
            file = h5py.File(os.path.join(np_folder, file[:-3]+"h5"), 'w')
            dataset = file.create_dataset(
            "data", transient.shape, dtype='f', data=transient
            )
            file.close()

def copy_mat_runs():
    import subprocess
    target = "/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/27_10_results"
    source = "/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic"
    for folder in os.listdir(source):
        if folder[-5:] == "27_10":
            command = f"cp -r {os.path.join(source, folder)} {os.path.join(target, folder)}"
            print(command)
            subprocess.run(command.split())


def edit_cvpr_logo():
    logo = imageio.imread("/scratch/ondemand28/anagh/active-yobo/data/yobo/cvpr_logo.png")           
    logo_new = np.zeros((900, 510, 4)).astype(np.uint8)
    offset = 100
    logo_new[-279-offset:-offset, :] = logo
    imageio.imwrite("data/yobo/cvpr_logo_botom.png", logo_new)

def mask_out_pre_depth():

    x = [17.26664936, -0.73867326,  -0.05389526, 0.16914815]
    FocalLength = [450.0188, 450.0733]
    PrincipalPoint = [261.6281, 243.4146]
    K = np.array([[FocalLength[0], 0, PrincipalPoint[0]], 
                  [0, FocalLength[1], PrincipalPoint[1]],
                  [0, 0, 1]])
    
    light_source_pos_wrt_cam = torch.Tensor(x[1:]).float()
    scale = 0.010376310322275158
    light_source_pos_wrt_cam = torch.concat([light_source_pos_wrt_cam, torch.tensor([1])]).double()


    frame  =         {
            "file_path": "./train_efficient/1_60.h5",
            "rotation": 0.012566370614359171,
            "transform_matrix": [
                [
                    0.0011904125333471007,
                    -0.9962134768331823,
                    0.08693268369228381,
                    1.1788242360573646
                ],
                [
                    0.9999990299861646,
                    0.0012487709265329443,
                    0.0006169261730023651,
                    0.11060695996045394
                ],
                [
                    -0.0007231491757164841,
                    0.08693186496972938,
                    0.996213997044889,
                    3.629742247378472
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        }
    
    camtoworld = frame["transform_matrix"]
    camtoworld = torch.from_numpy(np.array(camtoworld))

    total_dist = read_h5(f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/train_efficient/1_60.h5").squeeze()
    total_dist = np.argmax(total_dist, -1)*scale - x[0]
    total_dist = np.clip(total_dist, 0, None)
    total_dist = torch.from_numpy(total_dist)[..., None]
    # camtoworld[:2] *= -1
    # camtoworld[:2, -1] *= -
    # camtoworld = torch.eye(4)
    # pdb.set_trace()
    light_source_pos = (camtoworld @ light_source_pos_wrt_cam)[:3]
    device="cpu"
    origins, viewdirs = get_rays(K, camtoworld)
    # total_dist = np.clip(total_dist, 0, 1)
    depth = get_depth_from_distance_eq(total_dist.to(device).squeeze(), light_source_pos.to(device), viewdirs.to(device), origins.to(device))[..., None]
    virt_cam_pos = [0.8370092300209077,  0.09830451927694116, 3.6159140641004703]
    cvpr_roundtrip = (np.linalg.norm((origins + depth*viewdirs - np.array(virt_cam_pos)[None, None, :]), axis=-1) + depth.squeeze().numpy())
    bin_space_cvpr = (cvpr_roundtrip + x[0])/scale-1664
    np.save("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/train_efficient/1_60_cvpr_bin_space.npy", cvpr_roundtrip)
    pdb.set_trace()
    
    
def edit_cvpr_logo_colours():
    length = 900
    offset = 100
    logo = imageio.imread("/scratch/ondemand28/anagh/active-yobo/data/yobo/cvpr_logo.png")
    splits = [127, 382]           
    logo_c = np.zeros((length, 510, 4)).astype(np.uint8)
    logo_vp = np.zeros((length, 510, 4)).astype(np.uint8)
    logo_r = np.zeros((length, 510, 4)).astype(np.uint8)
    
    # pdb.set_trace()
    logo_c[-279-offset:-offset, :splits[0]] = logo[:, :splits[0]]
    logo_vp[-279-offset:-offset, splits[0]:splits[1]] = logo[:, splits[0]:splits[1]]
    logo_r[-279-offset:-offset, splits[1]:] = logo[:, splits[1]:]


    imageio.imwrite(f"data/yobo/cvpr_logo_c.png", logo_c)
    imageio.imwrite(f"data/yobo/cvpr_logo_vp.png", logo_vp)
    imageio.imwrite(f"data/yobo/cvpr_logo_r.png", logo_r)


    
    pdb.set_trace()
if __name__=="__main__":
    
    # t1 = imageio.imread("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_red_material_light_resample_finetune_eval_fixed_camera/save/material_indirect_color/0000.png")
    # t2= imageio.imread("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_green_material_light_resample_finetune_eval_fixed_camera/save/material_indirect_color/0000.png").squeeze()
    # t3 = imageio.imread("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_blue_material_light_resample_finetune_eval_fixed_camera/save/material_indirect_color/0000.png").squeeze()
    # transient = np.stack([t1, t2, t3], -1)
    # pdb.set_trace()
    # mask_out_pre_depth()
    # exit()
    # edit_cvpr_logo_colours()
    # shift_bin = np.load("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/train_efficient/1_60_cvpr_bin_space.npy")
    # t1 = read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_red_material_light_resample_finetune_eval_fixed_camera/save/material_transients/0000.h5").squeeze()
    # rename_runs()
    # transient = read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_cache_eval/save/transients/0000.h5")
    # save_video_from_h5(transient, "cvpr_no_shift.mp4", bkgd=True)
    # pdb.set_trace()
    exit()

    # t2= read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_green_material_light_resample_finetune_eval_fixed_camera/save/material_transients/0000.h5").squeeze()
    # t3 = read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_blue_material_light_resample_finetune_eval_fixed_camera/save/material_transients/0000.h5").squeeze()
 
    # transient = np.stack([t1, t2, t3], -1)
  
    # H, W, T, C = transient.shape

    # Loop through each element in `zero_bins` to set the appropriate bins to zero
    # for h in range(H):
    #     for w in range(W):
    #         offset = 10
    #         # if (h, w) == (256, 256):
    #         #     pdb.set_trace()
    #         bins_to_zero = np.clip(shift_bin[h, w] - offset, 0, t1.shape[-1]-1).astype(np.uint32) # Get the number of bins to zero
    #         transient[h, w, :bins_to_zero, :] = 0  # Set the first `bins_to_zero` bins to zero

    # t = read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/house_material_light_from_scratch_resample_eval_fixed_camera_t2/save/material_transients/0004.h5")
    # pdb.set_trace()
    # transient[..., 2]*= 0.5
    # transient2 = read_h5("/scratch/ondemand28/anagh/active-yobo/checkpoints/yobo_results/synthetic/cornell_material_light_from_scratch_resample_eval_fixed_camera/save/material_transients/0005.h5")
    # pdb.set_trace()
    save_video_from_h5(transient, "relit_house_rgb.mp4", bkgd=True)
    # pdb.set_trace()
    # plot_from_depth_not_multiview()
    exit()
    # create_train_test_split()
    
    # mat_to_npy("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/transients", "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/train_efficient")
    # exit()
    # plot_from_depth_not_multiview()
    
    # load_mitsuba_params("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers")
    # save_imgs()
    # find_transient_max("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers/train")
    exit()
        # save_video_from_h5(f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers/test/test_0_01.h5")
    # transient = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/cornell/train/0_00.h5")
    # make_ctof_dataset()
    # pdb.set_trace()
    # exit()
    # save_imgs()

    # exit()
    # transient_img2 = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/pots/train/0_19.h5")[..., :3]
    # cw_tof_data =  direct_tof_to_cw_tof(transient_img2, 1000*10**6, 0.01, [0, np.pi/2, np.pi, 3*np.pi/2])
    # save_imgs()
    # for file in ["3_22", "1_19", "1_10"]:
    # plot_solved_depths()
    # save_video_from_h5(f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers/test/test_0_01.h5")
    # downsample_shift_transients()

    # transient_img1 = read_h5("/scratch/ondemand28/kevin/scenes/kitchen/kitchen_sed2.h5")

    # transient_img2 = read_h5("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/pots/train/0_19.h5")

    # exit()
    # path_to_h5s = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers"
    # save_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers/train_efficient"
    # json_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/peppers/transforms_train.json"

    # bundle_rays(pathToH5s=path_to_h5s, outputPath=save_path, trainJsonPath =json_path)

    # path_to_h5s = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/transients"
    # save_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/train_efficient"
    # json_path = "/scratch/ondemand28/anagh/active-yobo/data/yobo/final_captured/house_11_01/transforms_train.json"

    # bundle_rays_cap(pathToH5s=path_to_h5s, outputPath=save_path, trainJsonPath =json_path)

    
    # exit()

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