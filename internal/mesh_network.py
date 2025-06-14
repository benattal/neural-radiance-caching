import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_args, squareplus

# import modules
import torch
import numpy as np
from tqdm import tqdm
# from pykdtree.kdtree import KDTree
import mrcfile
from radiance_fields.ngp import NGPRadianceField
import mcubes
device = torch.device('cuda')
import trimesh

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def export_model(args, model_name, N=1024, model_type='bacon', hidden_layers=8,
                 hidden_size=256, output_layers=[1, 2, 4, 8], w0=30, pe=8,
                 filter_mesh=False, scaling=None, return_sdf=False):



    step = args.max_steps

    ckpt_path = os.path.dirname(args.my_config)
    ckpt_path = os.path.join(ckpt_path, 'radiance_field_%04d.pth' % (step))

    ckpt = torch.load(ckpt_path, map_location=device)
    
    model = NGPRadianceField(aabb=args.aabb, radiance_activation=squareplus, use_viewdirs=True, args = args).to(device)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # write output
    x = torch.linspace(args.aabb[0], args.aabb[-1], N)
    # if return_sdf:
    #     x = 3*torch.arange(-N//2, N//2) / N
    #     x = x.float()
    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).to(device)
    sdf_values = [np.zeros((N**3, 1)) for i in range(1)]

    # render in a batched fashion to save memory
    bsize = int(4096**2)
    for i in tqdm(range(int(N**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]
        out = model.query_density(coords)d

        if not isinstance(out, list):
            out = [out,]

        for idx, sdf in enumerate(out):
            sdf = torch.nan_to_num(sdf)
            sdf_values[idx][i*bsize:(i+1)*bsize] = (sdf.detach().cpu().numpy())

    print(np.max(sdf_values))
    print("---")
    print(sdf_values)
    return [sdf.reshape(N, N, N) for sdf in sdf_values]


def normalize(coords, scaling=0.9):
    coords = np.array(coords).copy()
    cmean = np.mean(coords, axis=0, keepdims=True)
    coords -= cmean
    coord_max = np.amax(coords)
    coord_min = np.amin(coords)
    coords = (coords - coord_min) / (coord_max - coord_min)
    coords -= 0.5
    coords *= scaling

    scale = scaling / (coord_max - coord_min)
    offset = -scaling * (cmean + coord_min) / (coord_max - coord_min) - 0.5*scaling
    return coords, scale, offset


def get_ref_spectrum(xyz_file, N):
    pointcloud = np.genfromtxt(xyz_file)
    v = pointcloud[:, :3]
    n = pointcloud[:, 3:]

    n = n / (np.linalg.norm(n, axis=-1)[:, None])
    v, _, _ = normalize(v)
    print('loaded pc')

    # put pointcloud points into KDTree
    kd_tree = KDTree(v)
    print('made kd tree')

    # get sdf on grid and show
    x = (np.arange(-N//2, N//2) / N).astype(np.float32)
    coords = np.stack([arr.flatten() for arr in np.meshgrid(x, x, x)], axis=-1)

    sdf, idx = kd_tree.query(coords, k=3)

    # get average normal of hit point
    avg_normal = np.mean(n[idx], axis=1)
    sdf = np.sum((coords - v[idx][:, 0]) * avg_normal, axis=-1)
    sdf = sdf.reshape(N, N, N)
    return [sdf, ]


def extract_spectrum(args):

    scenes = ['armadillo']
    Ns = [420, 384, 384, 512, 512]
    methods = ['bacon']

    for method in methods:
        for scene, N in zip(scenes, Ns):
            
            config_path = os.path.dirname(args.my_config)
            print(config_path)

            # ckpt = '/home/anagh/PycharmProjects/multiview_lif/results/boxes_sp_01-16_12:59:10/radiance_field_10000.pth'

            sdfs = export_model(args, scene, model_type=method, output_layers=[2, 4, 6, 8],
                                return_sdf=True, N=N, pe=8, w0=30)

            sdfs_ft = sdfs
            thold = 0.1

            print( np.mean(sdfs_ft[0] > thold))
            vertices, triangles = mcubes.marching_cubes(sdfs_ft[0], thold)
            
            mcubes.export_obj(vertices, triangles,os.path.join(config_path, "mesh.obj"))      
            
            # mesh = trimesh.Trimesh(vertices / N - .5, triangles)
            # mesh.show()
            exit()

            # for idx, sdf_ft in enumerate(sdfs_ft):
            #     with mrcfile.new_mmap(f'./two_planes.mrc', overwrite=True, shape=(N, N, N), mrc_mode=2) as mrc:
            #         mrc.data[:] = sdf_ft



            # os.system('chimerax /tmp/two_planes.mrc')


if __name__ == '__main__':
    args = load_args()
    extract_spectrum(args)