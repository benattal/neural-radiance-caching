import json
import imageio 
import numpy as np 
import os
# from batch_mitsuba import batch, read_h5 
# from transform_jsons import generate_random_camera_transforms
# from transform_xml import gen_xml

def transform_base_json_random():
    # positions = "/home/anagh/PycharmProjects/multiview_lif/data/regular-cornell/cornell-90.json"
    outpath = "/home/anagh/PycharmProjects/multiview_lif/data/peppers/diverse_cams_scaled_diffused_glass/transforms.json"
    # with open(positions) as fp:
    #     positions = json.load(fp)

    # positions = generate_random_camera_transforms(90, (-20, 100), (-20, 100), (-10, 100), (1, 2))
    # positions = generate_random_camera_transforms(90, (-1, 5), (-2, -1.5), (0, 2), (3.5, 5), target_point=np.array([0, 0, -0.5]))
    positions = generate_random_camera_transforms(90, (1.4, 2), (-2, -1.5), (0, 1.5), (3.5, 5), target_point=np.array([0, 0, -0.5]))


    new_json = {}
    new_json["camera_angle_x"] = 0.69097585
    new_json["frames"] = []

    for ind, pos in enumerate(positions):
        frame = {}
        frame["filepath"] = f"{ind}-nonconfocal.pt"
        frame["transform_matrix"] = pos.tolist()
        new_json["frames"].append(frame)
    
    # json_object = json.dumps(new_json, indent=4)
    with open(outpath, 'w') as f:
        json.dump(new_json, f)

import math


def transforms_spiral(starting_point, ending_point, target_point, distance, N, outpath):
    
    starting_vect = starting_point - target_point
    ending_vect = ending_point - target_point

    angle_between = np.arccos(starting_vect.T @ ending_vect) / (np.linalg.norm(starting_vect) * np.linalg.norm(ending_vect))
    angle_increments = angle_between/N
    cross_vect = np.cross(ending_vect, starting_vect)
    cross_vect = cross_vect/np.linalg.norm(cross_vect)

    camera_position = starting_point
    test = True
    
    frames = []

    for n in range(N):
        if n> 0:
            # tp = camera_position.copy()
            # tp[:2] = 0
            camera_position = target_point + rodrigues_rotation(camera_position - target_point, cross_vect, -angle_increments)

        # # Step 3: Generate a random distance within the specified range.
        # distance = np.random.uniform(distance_range[0], distance_range[1]) if distance_range[1] is not None else np.random.uniform(0.1, 5)
        up = np.array([0, 0, 1])

        # Step 4: Compute the view matrix.

        # Calculate the forward vector from the camera position to the target point.
        forward = target_point - camera_position
        forward /= np.linalg.norm(forward)

        camera_position = target_point - distance * forward  # Adjust camera position based on distance.

        # Calculate the right vector using the cross product of forward and up vectors.
        
        right = np.cross(forward, up)
        right/= np.linalg.norm(right)
        
        # Recalculate the up vector to ensure it's orthogonal to the forward and right vectors.
        up = np.cross(right, forward)

        view_matrix = np.eye(4)
        view_matrix[:3, 0] = right
        view_matrix[:3, 1] = up
        view_matrix[:3, 2] = -forward
        view_matrix[:3, 3] = camera_position

        # transforms.append(view_matrix)
        
        frame = {}
        if test:
            frame["filepath"] = f"spiral_{n}.h5"
        else:
            frame["filepath"] = f"spiral_{n}.h5"
        frame["transform_matrix"] = view_matrix.tolist()
        frames.append(frame)    
    

    new_json = {}
    new_json["camera_angle_x"] = 0.69097585
    new_json["frames"] = frames

    
    json_object = json.dumps(new_json, indent=4)
    with open(outpath, 'w') as f:
        f.write(json_object)
        

def transform_base_json():
    positions = "/home/anagh/PycharmProjects/multiview_lif/data/peppers/arch_cams/camera.json"
    outpath = "/home/anagh/PycharmProjects/multiview_lif/data/peppers/arch_cams/transforms.json"
    with open(positions) as fp:
        positions = json.load(fp)



    new_json = {}
    new_json["camera_angle_x"] = 0.69097585
    new_json["frames"] = []

    for ind, pos in enumerate(positions):
        frame = {}
        frame["filepath"] = f"{ind}-nonconfocal.pt"
        frame["transform_matrix"] = pos
        new_json["frames"].append(frame)
    
    # json_object = json.dumps(new_json, indent=4)
    with open(outpath, 'w') as f:
        json.dump(new_json, f)


def save_train_test_json(train_views, test_views, positions, target_path):
    train_views = [positions["frames"][i] for i in train_views]
    test_views =  [positions["frames"][i] for i in test_views]
    u = positions.copy()
    u["frames"] = train_views

    json_object = json.dumps(u, indent=4)
    with open(os.path.join(target_path, "transforms_train.json"), "w") as outfile:
        outfile.write(json_object)
    
    u = positions.copy()
    u["frames"] = test_views

    json_object = json.dumps(u, indent=4)
    with open(os.path.join(target_path, "transforms_test.json"), "w") as outfile:
        outfile.write(json_object)


def rodrigues_rotation(n, u, theta):
    u_dot_n = u @ n
    u_cross_n = np.cross(u, n)
    return n * np.cos(theta) + u_cross_n * np.sin(theta) + u * u_dot_n * (1 - np.cos(theta))


def transforms_rot_stage(outpath, x_rots, y_rots, x_rot_deg, y_rot_deg, init_camera_position, camera_position, distance, target_point, test=False):
    # outpath = "/home/anagh/PycharmProjects/multiview_lif/data/peppers/final_cams/transforms.json"
    
    # x_rots = 16
    # y_rots = 3
    # x_rot_deg = 3
    # y_rot_deg = 10
    # init_camera_position = np.array([0, -1.6, 0.5])
    # camera_position = init_camera_position
    
    # distance = 1.6
    # target_point = np.array([0, 0, -0.5])
    frames = []

    for j in range(y_rots):
        if j>0:
            camera_position = target_point + rodrigues_rotation(camera_position - target_point, -y_rot_axis, j*np.deg2rad(y_rot_deg) )
            
          
        for i in range(x_rots):
            if i> 0:
                # tp = camera_position.copy()
                # tp[:2] = 0
                camera_position = target_point + rodrigues_rotation(camera_position - target_point, np.array([0, 0, 1]), -np.deg2rad(x_rot_deg) )


            # # Step 3: Generate a random distance within the specified range.
            # distance = np.random.uniform(distance_range[0], distance_range[1]) if distance_range[1] is not None else np.random.uniform(0.1, 5)
            up = np.array([0, 0, 1])

            # Step 4: Compute the view matrix.
            # camera_position = np.array([x, y, z])
            # target_point = 


            # Calculate the forward vector from the camera position to the target point.
            forward = target_point - camera_position
            forward /= np.linalg.norm(forward)

            camera_position = target_point - distance * forward  # Adjust camera position based on distance.

            # Calculate the right vector using the cross product of forward and up vectors.
            
            right = np.cross(forward, up)
            right/= np.linalg.norm(right)
            
            if i==0:
                y_rot_axis = right.copy()

            # Recalculate the up vector to ensure it's orthogonal to the forward and right vectors.
            up = np.cross(right, forward)

            view_matrix = np.eye(4)
            view_matrix[:3, 0] = right
            view_matrix[:3, 1] = up
            view_matrix[:3, 2] = -forward
            view_matrix[:3, 3] = camera_position

            # transforms.append(view_matrix)
            
            frame = {}
            if test:
                frame["file_path"] = f"./test/test_{j}_{i:02d}.h5"
            else:
                frame["file_path"] = f"./train/{j}_{i:02d}.h5"
            frame["transform_matrix"] = view_matrix.tolist()
            frames.append(frame)
        camera_position = init_camera_position


    new_json = {}
    new_json["camera_angle_x"] = 0.69097585
    new_json["frames"] = frames

    
    json_object = json.dumps(new_json, indent=4)
    with open(outpath, 'w') as f:
        f.write(json_object)

def make_images(training_path, images_path):
    for file in os.listdir(training_path):
        if file.endswith("h5"):
            try:
                filepath = os.path.join(training_path, file)
                tran = read_h5(filepath)
                image = tran.sum(-2)[..., :3]
                image = (image/image.max())**(1/2.2)
                imageio.imwrite(
                            os.path.join(images_path, f"{file.split('.')[0]}.png"),
                            (image * 255).astype(np.uint8),
                        )
            except:
                pass

def transforms_rot_stage_make_xmls_batch_mitsuba_make_images():
    scene = "cornell"
    test = False

    if scene=="cornell":
        folder = f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/{scene}"
        xmls_path = os.path.join(folder, "xmls")
        # training_path = os.path.join(folder, "training_files")
        images_path = os.path.join(folder, "images")
        

        # os.makedirs(xmls_path, exist_ok=True)
        # os.makedirs(training_path, exist_ok=True)
        # os.makedirs(images_path, exist_ok=True)


        x_rots = 32
        y_rots = 3
        x_rot_deg = 3
        y_rot_deg = 25
        init_camera_position =  np.array([0, 1.5, 0.25])
        camera_position = init_camera_position
        
        distance = 1.65
        target_point = np.array([0, 0, 0.35])
    
        if test:
            outpath = os.path.join(folder, "transforms_test.json")
        else:
            outpath = os.path.join(folder, "transforms_train.json")

    
    transforms_rot_stage(outpath, x_rots, y_rots, x_rot_deg, y_rot_deg, init_camera_position, camera_position, distance, target_point, test=test)

    # starting_point = np.array([-0.5, 1.6, 0.3])
    # ending_point =  np.array([1.5, -0.1, 1])
    # target_point = np.array([0, 0, 0.4])
    # distance = 1.6
    # N = 100
    # transforms_spiral(starting_point, ending_point, target_point, distance, N, outpath)
    
    # check this path out 
    # template_path = os.path.join(folder, "0-nonconfocal.xml")
    
    # generate xml files
    # gen_xml(target_path=xmls_path, json_path=ouetpath, template_path=template_path, res=res)
    
    # # run mitsuba on it
    # batch(path=xmls_path, out_path=training_path, test=test, spiral=False)
    
    # # generate images 
    # make_images(training_path, images_path)
    
    
    
    
    
    
        
if __name__=="__main__":
    
    transforms_rot_stage_make_xmls_batch_mitsuba_make_images()
    
    exit()
    # transform_base_json()
    # positions = "/home/anagh/PycharmProjects/multiview_lif/data/regular-cornell/transforms.json"
    # with open(positions) as fp:
    #     positions = json.load(fp)
    
    # test_views = np.arange(0, 100, 10)
    # train_views = [int(x) for x in np.arange(0, 90) if x not in test_views]
    # save_train_test_json(train_views, test_views, positions, "/home/anagh/PycharmProjects/multiview_lif/data/regular-cornell/downsampled")

    transforms_rot_stage()