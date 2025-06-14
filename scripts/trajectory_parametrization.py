import numpy as np 
import torch 
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import json 


class Camera:
    def __init__(self, K, target_point, up, origin, time):
        self.origin = origin 
        self.up = up
        self.target_point = target_point
        self.K = K
        self.time = time
    
    def get_intrinsics(self):
        return self.K
        
    def get_extrinsics(self):
        forward = self.target_point - self.origin
        forward /= np.linalg.norm(forward)        
        right = np.cross(forward, self.up)
        right/= np.linalg.norm(right)
        up = np.cross(right, forward)
        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = self.origin
        return view_matrix
    
    def __add__(self, other_camera):
        new_origin = self.origin + other_camera.origin
        new_up = self.up + other_camera.up
        new_target_point = self.target_point  + other_camera.target_point
        new_K = self.K + other_camera.K
        new_time = self.time + other_camera.time
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)

    def __mul__(self, scalar):
        new_origin = self.origin*scalar
        new_up = self.up*scalar
        new_target_point = self.target_point*scalar
        new_K = self.K*scalar
        new_time = self.time*scalar
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)
    
    def __rmul__(self, scalar):
        new_origin = self.origin*scalar
        new_up = self.up*scalar
        new_target_point = self.target_point*scalar
        new_K = self.K*scalar
        new_time = self.time*scalar
        return Camera(new_K, new_target_point, new_up, new_origin, new_time)

    def __str__(self):
        ret = f"origin: {np.array2string(self.origin)}, \n tp: {np.array2string(self.target_point)}"
        return ret


class Trajectory:
    def __init__(self, cameras, interpolations, center=np.array([0, 0, 0])):
        self.cameras = cameras
        self.interpolations = interpolations
        self.trajectory = []
        self.trajectory_computed = False
        self.center = center

    
    def compute_trajectory(self):
        self.trajectory_computed = True
        
        for ind, camera in enumerate(self.cameras):
            # add current camera
            self.trajectory.append(camera)
            
            #check if last camera 
            if ind < len(self.cameras)-1:
                
                #get next camera
                next_camera = self.cameras[ind+1]
                (interpolation_type, interpolation_number) = self.interpolations[ind]
                
                if interpolation_type == "sphere":
                    segment = self.sphere_interpolation(camera, next_camera, interpolation_number, self.center)
                    self.trajectory += segment 
                
                if interpolation_type == "linear":
                    segment = self.linear_interpolation(camera, next_camera, interpolation_number)
                    self.trajectory += segment 
                    

    def linear_interpolation(self, c1, c2, num_cams):
        segment = []
        for i in range(num_cams):
            c = c1*((num_cams-i)/num_cams) + (i/num_cams)*c2
            segment.append(c)
        return segment
            
    def sphere_interpolation(self, c1, c2, num_cams, center):
        segment = []
        o1, o2 = c1.origin, c2.origin
        d1, d2 = np.linalg.norm(c1.origin-center), np.linalg.norm(c2.origin-center)
        starting_vect = o1 - center
        ending_vect = o2 - center

        angle_between = np.arccos(starting_vect.T @ ending_vect / (np.linalg.norm(starting_vect) * np.linalg.norm(ending_vect))) 
        angle_increments = angle_between/num_cams
        cross_vect = np.cross(ending_vect, starting_vect)
        cross_vect = cross_vect/np.linalg.norm(cross_vect)

        for i in range(num_cams):
            c = ((num_cams-i)/num_cams)*c1 + (i/num_cams)*c2
            camera_position = center + self.rodrigues_rotation(o1 - center, cross_vect, -angle_increments*i)
            distance = ((num_cams-i)/num_cams)*d1 + (i/num_cams)*d2
            forward = center - camera_position
            forward /= np.linalg.norm(forward)
            # print(camera_position)
            camera_position = center - distance * forward 
            c.origin = camera_position
            segment.append(c)
        return segment


    def add_camera(self, camera):
        self.cameras.append(camera)
        self.trajectory_computed = False

    
    def delete_camera(self, index):
        self.cameras.pop(index)
    
    def get_trajectory(self):
        if self.trajectory_computed:
            return self.trajectory
        else:
            self.compute_trajectory()
            return self.trajectory 
    
    def save_transforms(self, path, frames = None):
        if frames == None:
            frames = self.trajectory
        frames = [fr.get_extrinsics() for fr in frames]
        new_json = {}
        new_json["camera_angle_x"] = 0.69097585
        views = []
        for ind, f in enumerate(frames):
            frame = {}
            frame["file_path"] = f"{ind:04d}.h5"
            frame["transform_matrix"] = f.tolist()
            views.append(frame)

        new_json["frames"] = views


        
        json_object = json.dumps(new_json, indent=4)
        with open(path, 'w') as f:
            f.write(json_object)

        
    
    def rodrigues_rotation(self, n, u, theta):
        u_dot_n = u @ n
        u_cross_n = np.cross(u, n)
        return n * np.cos(theta) + u_cross_n * np.sin(theta) + u * u_dot_n * (1 - np.cos(theta))

    # def smoothen_trajectory(self):
    #     positions = np.array([f.origin for f in self.trajectory])
    #     tck, u = splprep([positions[:, 0], positions[:, 1], positions[:, 2]], s=0)
    #     new_u = np.linspace(0, 1, 1000)
    #     new_points = splev(new_u, tck)
    #     for ind, cam in enumerate(self.trajectory):
    #         cam.origin = new_points[ind]
    #         self.trajectory[ind] = cam
        
    
    def smoothen_trajectory(self):
        points = np.array([f.origin for f in self.trajectory])
        times = np.array([f.time for f in self.trajectory])
        times = ((times-times.min())/(times.max()-times.min()))
        
        # Define the helix equation
        # def helix_equation(theta, a, b, c, d, e, f, pitch):
        #     x = a * np.cos(theta) + d
        #     y = c * np.sin(theta+f) + e
        #     z = b * theta / (2 * np.pi * pitch)
        #     return np.column_stack((x, y, z))
        
        
        
        def bezier_curve(params, t):
            t = t[:, None]
            P0, P1, P2, P3, P4 = params[:3][None], params[3:6][None], params[6:9][None], params[9:12][None], params[12:15][None]
            # P0 = np.array([0.896, -0.666, 3.555])[None]
            return ((1 - t)**3) * P0 + 3 * t * ((1 - t)**2) * P1 + 3 * (t**2) * (1 - t) * P2 + (t**3) * P0 +  (t**2) * (1 - t)**3 * P3 +  (t)**3 * ((1 - t)**2 )* P4
            

        # Define the cost function
        def cost_function(params, points):
            # a, b, c, d, e, f, pitch = params
            theta = times
            # helix_points = helix_equation(theta, a, b, c, d, e, f, pitch)
            helix_points = bezier_curve(params, theta)
            return np.sum((points - helix_points)**2)

        # Initial guess for parameters
        # initial_guess = [1, 1, 1, 1, 1, 1, 1]
        initial_guess = np.ones((15))

        # Minimize the cost function
        result = minimize(cost_function, initial_guess, args=(points,))
        best_params = result.x

        # Generate points on the helix using best parameters
        # theta = np.linspace(0, 2 * np.pi, len(points))
        theta = times
        # helix_points = helix_equation(theta, *best_params)
        helix_points = bezier_curve(best_params, theta)
        for ind, cam in enumerate(self.trajectory):
            cam.origin = helix_points[ind]
            self.trajectory[ind] = cam



def jfk_traj():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.4, -3.7, -0.44]),
        up=np.array([-0.955, 0.101, 0.276]), 
        target_point=np.array([1.1, -0.2, -0.4]),
        time=1540,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.4, -3.3, -0.44]),
        up=np.array([-0.955, 0.101, 0.276]), 
        target_point=np.array([1.1, -0.2, -0.4]),
        time=1600,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.980, -3.01, 2.118]),
        up=np.array([-0.955, 0.101, 0.276]), 
        target_point=np.array([1.1, -0.2, -0.4]),
        time=1800,
        K = K,        
    )
    
    # c4 = Camera(
    #     origin=np.array([0.097, -3.39, 1.187]),
    #     up=np.array([-0.950, 0.273, 0.155]), 
    #     target_point=np.array([1.1, 0, -0.4]),
    #     time=1950,
    #     K = K,        
    # )
    
    c4 = Camera(
        origin=np.array([0.097, -3.39, 1.187]),
        up=np.array([-0.950, 0.273, 0.155]), 
        # target_point=np.array([0.4, 1.3, 0.7]),
        target_point=np.array([1.1, 0.1, -0.4]),
        time=2200,
        K = K,        
    )
        
    # c5 = Camera(
    #     origin=np.array([0.097, -3.39, 1.187]),
    #     up=np.array([-0.950, 0.273, 0.155]), 
    #     target_point=np.array([0.4, 1.3, 0.7]),
    #     time=2400,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4]
    interpolations = [("linear", 60), ("sphere", 200), ("sphere", 400)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()
    t = traj.get_trajectory()
    
    return t

    # starting_point = np.array([0.3, -3.73,  -0.814])
    # # ending_point =  np.array([0.55, -0.65, 3.52 ])
    # ending_point =  np.array([0.3, -0.65, 4 ])

    # target_point = np.array([1.1, -0, -0.4])

    #     up = np.array([-0.4, 0, 0.11])

def jfk_smooth_traj():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.3, -3.73,  -0.814]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1500,
        K = K)
    
    c2 = Camera(
        origin=np.array([1, -3.01, 2.118]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1660,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.3, -0.65, 4 ]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=1820,
        K = K,        
    )
    
    c4 = Camera(
        origin=np.array([-1, -2.869,  2.162]),
        up=np.array([-0.4, 0, 0.11]), 
        # target_point=np.array([0.4, 1.3, 0.7]),
        target_point=np.array([1.1, 0.1, -0.4]),
        time=1980,
        K = K,        
    )
    
    c5 = Camera(
        origin=np.array([0.3, -3.73,  -0.814]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, -0, -0.4]),
        time=2140,
        K = K,        
    )
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("linear", 160), ("sphere", 160), ("sphere", 160), ("sphere", 160)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    traj.save_transforms("/scratch/ondemand28/anagh/multiview_lif/final_dataset_captured/jfk_18_02/training_files/transforms_bezier.json")
    return t


def mirror_smooth_traj():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([1.23, -0.3, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1650,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.4, -2.62, 2.446]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1900,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.4, 1.59, 3.2 ]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2000,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([1.23, -0.3, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2200,
        K = K)
    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4]
    interpolations = [("sphere", 250), ("sphere", 100), ("sphere", 200)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t


def grating_smooth_trajectory():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([1.230, -0.673, 3.562]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1600,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.6, -2.62, 2.446]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=1750,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([0.5, 1.695, 3.273 ]),
        up=np.array([-0.4, 0, 0.11]), 
        target_point=np.array([1.1, 0.2, -0.4]),
        time=2050,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([1.23, -0.67, 3.561]),
        up=np.array([-0.5, 0, 0.11]), 
        target_point=np.array([1, 0.2, -0.25]),
        time=2200,
        K = K)
    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4]
    interpolations = [("sphere", 150), ("sphere", 300), ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t



def coke_smooth_trajectory_old():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1650,
        K = K)
    
    c2 = Camera(
        origin=np.array([0, 2.781, 2.0]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1800,
        K = K,        
    )
    
    c3 = Camera(
        origin=np.array([-0.6, -0.84, -3.3]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1850,
        K = K,        
    )
    
    c4= Camera(
        origin=np.array([0.2, -4.2, 1]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1950,
        K = K)
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2100,
        K = K)
    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 150), ("sphere", 100), ("sphere", 100),  ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    # traj.smoothen_trajectory()

    t = traj.get_trajectory()
    return t


def coke_smooth_trajectory():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1550,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.1, -3.8, 1]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1700,
        K = K)
    
    c3 = Camera(
        origin=np.array([-0.4, -0.84, -3.3]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1850,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0, 2.781, 2.0]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2000,
        K = K,        
    )
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=2150,
        K = K)
    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 150), ("sphere", 100), ("sphere", 100),  ("sphere", 150)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    # traj.smoothen_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    return t


def coke_one_wide():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1500,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1600,
        K = K)
    
    c3 = Camera(
        origin=np.array([1.07, 1.696,  3.27]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1880,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2200,
        K = K,        
    )
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2300,
        K = K)

    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 100), ("sphere", 280), ("sphere", 320),  ("sphere", 100)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    # traj.smoothen_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    traj.save_transforms("/scratch/ondemand28/anagh/multiview_lif/final_dataset_captured/coke_08_02/training_files/transforms_bezier.json", t)    
    return t


def coke_unwarped():
    
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=np.array([0.98, -0.11, 0.334]),
        time=1500-280,
        K = K)
    
    c2 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1600-280,
        K = K)
    
    c3 = Camera(
        origin=np.array([1.07, 1.696,  3.27]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=1880-280,
        K = K,  
    )
    
    c4 = Camera(
        origin=np.array([0.098, -3.399, 1.187]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2200-280,
        K = K,        
    )
    
    c5= Camera(
        origin=np.array([0.896, -0.666, 3.555]),
        up=np.array([-0.5, 0, 0.09]), 
        target_point=   np.array([0.98, -0.11, 0.334]),
        time=2300-280,
        K = K)

    
    # c5 = Camera(
    #     origin=np.array([0.3, -3.73,  -0.814]),
    #     up=np.array([-0.4, 0, 0.11]), 
    #     target_point=np.array([1.1, -0, -0.4]),
    #     time=2140,
    #     K = K,        
    # )
    

    
    cameras = [c1, c2, c3, c4, c5]
    interpolations = [("sphere", 100), ("sphere", 280), ("sphere", 320),  ("sphere", 100)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([1.1, -0.2, -0.4]))
    
    traj.compute_trajectory()
    # traj.smoothen_trajectory()

    t = traj.get_trajectory()
    origins = [f.origin for f in t]
    for i in range(100):
        origins = [((1/2)*x[0] + (1/2)*x[1]) for x in zip(origins[1:]+[origins[0]], [origins[-1]]+origins[:-1])]

    for ind, cam in enumerate(t):
        cam.origin = origins[ind]
        t[ind] = cam
    
    # traj.save_transforms("/scratch/ondemand28/anagh/multiview_lif/final_dataset_captured/coke_08_02/training_files/transforms_bezier.json", t)    
    return t

def kitchen_training():
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([-0.90, -4.5, 0.4]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 2]),
        time=0,
        K = K)

    c2 = Camera(
        origin=np.array([2.75, 1.1, 0.4]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0, 0, 1.2]),
        time=0,
        K = K)

    c3 = Camera(
        origin=np.array([2.8, 1.2, 1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0, 0, 1]),
        time=0,
        K = K)
    
    c4 = Camera(
        origin=np.array([-1.0, -4.8, 1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 1.5]),
        time=0,
        K = K)
    
    c5 = Camera(
        origin=np.array([-1.0, -4.8, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 1]),
        time=0,
        K = K)
    
    c6 = Camera(
        origin=np.array([2.8, 1.2, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-2, 1, 1]),
        time=0,
        K = K)

    cameras = [c1, c2, c3, c4, c5, c6]
    interpolations = [("sphere", 30), ("linear", 1), ("sphere", 30), ("linear", 3), ("linear", 30)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([0, 0, 0.4]))
    
    traj.compute_trajectory()
    t = traj.get_trajectory()



    traj.save_transforms("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/kitchen/transforms_train.json", t) 


def kitchen_test():
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([-0.95, -4.4, 0.3]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 2.1]),
        time=0,
        K = K)

    c2 = Camera(
        origin=np.array([2.7, 1, 0.4]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0.1, 0.1, 1.2]),
        time=0,
        K = K)

    c3 = Camera(
        origin=np.array([2.7, 1.1, 1.1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0.1, 0.1, 1]),
        time=0,
        K = K)
    
    c4 = Camera(
        origin=np.array([-1.1, -4.7, 1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 1.9, 1.4]),
        time=0,
        K = K)
    
    c5 = Camera(
        origin=np.array([-1.0, -4.81, 2.51]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 1.9, 1.1]),
        time=0,
        K = K)
    
    c6 = Camera(
        origin=np.array([2.8, 1.24, 2.51]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-2, 1.1, 1.1]),
        time=0,
        K = K)

    cameras = [c1, c2, c3, c4, c5, c6]
    interpolations = [("sphere", 9), ("linear", 1), ("sphere", 9), ("linear", 2), ("linear", 9)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([0, 0, 0.4]))
    
    traj.compute_trajectory()
    t = traj.get_trajectory()



    traj.save_transforms("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/kitchen/transforms_evaluation.json", t) 



def kitchen_training_extra():
    K =  np.array([[443.5582, 0, 266.3042], [0, 444.2890, 247.5339], [0, 0, 1.0000]])

    c1= Camera(
        origin=np.array([-0.90, -4.5, 0.4]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 2]),
        time=0,
        K = K)

    c2 = Camera(
        origin=np.array([2.75, 1.1, 0.4]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0, 0, 1.2]),
        time=0,
        K = K)

    c3 = Camera(
        origin=np.array([2.8, 1.2, 1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0, 0, 1]),
        time=0,
        K = K)
    
    c4 = Camera(
        origin=np.array([-1.0, -4.8, 1]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 1.5]),
        time=0,
        K = K)
    
    c5 = Camera(
        origin=np.array([-1.0, -4.8, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-1, 2, 1]),
        time=0,
        K = K)
    
    c6 = Camera(
        origin=np.array([2.8, 1.2, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([-2, 1, 1]),
        time=0,
        K = K)

    c7 = Camera(
        origin=np.array([1.8, 1.2, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([0.5, -2.9, 1]),
        time=0,
        K = K)
    
    c8 = Camera(
        origin=np.array([0.8, 1.2, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([3.3, -1,  1.5]),
        time=0,
        K = K)
    
    c9 = Camera(
        origin=np.array([-1.8, 1.2, 2.5]),
        up=np.array([0, 0, 1]),
        target_point=np.array([3.3, 2,  1.5]),
        time=0,
        K = K)
    
    cameras = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    interpolations = [("sphere", 30), ("linear", 1), ("sphere", 30), ("linear", 3), ("linear", 30), ("linear", 5), ("linear", 5), ("linear", 5)]
    
    traj = Trajectory(cameras=cameras, interpolations=interpolations, center = np.array([0, 0, 0.4]))
    
    traj.compute_trajectory()
    t = traj.get_trajectory()



    traj.save_transforms("/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/kitchen/transforms_train_27_10.json", t) 
    # traj.save_transforms("./transforms_train_27_10.json", t) 


if __name__=="__main__":
    kitchen_training_extra()
    print("helo")