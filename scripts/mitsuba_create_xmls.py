import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json

import numpy as np
from xml.etree import ElementTree
from glob import glob
from imageio import imread, imsave
import h5py
import pdb 

def read_xml_lines(xml_path):
    inFile = open(xml_path, 'r', encoding='utf-8')
    xml_file = inFile.readlines()
    return xml_file


def save_xml(xml_file, path):
    outFile = open(os.path.join(path), 'w', encoding='utf-8')
    outFile.writelines(xml_file)
    outFile.close()
    return


def read_json(json_path):
    f = open(json_path)
    positions = json.load(f)
    f.close()
    return positions

def read_h5(path):
    with h5py.File(path, 'r') as f:
        frames = np.array(f['data'])
    return frames

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


def gen_xml(target_path='./train', json_path='./transforms_test1.json', template_path = None, res=64, spp=8192):

    spp = spp
    resx = res
    resy = res

    xml_path = template_path
    positions = read_json(json_path)
    # train_views = np.arange(0, 200, 2)
    # test_views = []
    # train_views = list(range(48))
    # test_views = [np.arange(0, 90, 10)]
    # train_views = [x for x in np.arange(0, 90) if x not in test_views]

    # save_train_test_json(train_views, test_views, positions, os.path.join(target_path, "../training_files"))




    # os.makedirs(target_path, exist_ok=True)

    xml_file = ElementTree.parse(xml_path)
    root = xml_file.getroot()
    root[0].attrib['value'] = str(spp)
    root[1].attrib['value'] = str(resx)
    root[2].attrib['value'] = str(resy)

    # ALL SET MANUALLY
    ax_flip = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    for ind, pos in enumerate(positions["frames"]):
        # if ind in train_views or ind in test_views:
        # tm = np.fromstring(pos["transform_matrix"][1:-1], sep=",").reshape(4, 4).T
        tm = np.array(pos["transform_matrix"])
        trans = ax_flip@tm
        trans[:, 2] *= -1
        trans[:, 0] *= -1

        # xml_file[25] = '<matrix value="' + str(trans.bflatten())[1:-1] + ' " />\n'
        # xml_file[132] = '<matrix value="' + str(trans.flatten())[1:-1] + ' " />\n'
        # save_xml(xml_file, path=os.path.join(target_path, f'{str(ind)}.xml'))
        # pdb.set_trace()
        root[4][6][0].attrib["value"] = str([x for x in trans.flatten()])[1:-1]  # str(trans.flatten())[1:-1]
        root[5][1][0].attrib["value"] = str([x for x in trans.flatten()])[1:-1]  # str(trans.flatten())[1:-1]
        # xml_file.write(os.path.join(target_path, f'{ind}-nonconfocal.xml'))

        xml_file.write(os.path.join(target_path, f"{pos['file_path'].split('/')[-1][:-3]}.xml"))

    
def batch(path, out_path, test=False):
    for file in os.listdir(path):
        if file[-3:] == "xml" and file[0]!="d":
            if ("test" in file) or ("inner" in file):
                pass
                # filepath = os.path.join(path, file)
                # # out_path = os.path.join(path, "training_files")
                # out_path_new = os.path.join(out_path, f"test/{file[:-4]}.h5")
                # os.system(f"/scratch/ondemand28/anagh/mitsuba2-transient-nlos/build/dist/mitsuba {filepath} -o {out_path_new}")
                # # print(f"/scratch/ondemand28/anagh/mitsuba2-transient-nlos/build/dist/mitsuba {filepath} -o {out_path_new}")
                # # os.system(f"conda init && conda activate ingp && mitsuba {filepath} -o {out_path}")
            elif "test" not in file and int(file[:-4])>99:
                filepath = os.path.join(path, file)
                # out_path = os.path.join(path, "training_files")
                out_path_new = os.path.join(out_path, f"train/{file[:-4]}.h5")
                # print(f"/scratch/ondemand28/anagh/mitsuba2-transient-nlos/build/dist/mitsuba {filepath} -o {out_path_new}")
                os.system(f"/scratch/ondemand28/anagh/mitsuba2-transient-nlos/build/dist/mitsuba {filepath} -o {out_path_new}")
                # os.system(f"conda init && conda activate ingp && mitsuba {filepath} -o {out_path}")

def render_transients():
    # # xmls_path
    scene = "kitchen"
    path_to_folder = f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/{scene}"
    for file in ["train"]:
        target_path = f"{path_to_folder}/xmls"
        json_path = f"{path_to_folder}/transforms_{file}.json"
        template_path = f"{path_to_folder}/0_00.xml"
        gen_xml(target_path=target_path, json_path=json_path, template_path=template_path, res=512, spp=4096)
    
    # render
    path = f"{path_to_folder}/xmls"
    out_path = f"{path_to_folder}"
    batch(path = path, out_path=out_path)


def render_depth():
    # # xmls_path
    scene = "peppers"
    path_to_folder = f"/scratch/ondemand28/anagh/active-yobo/data/yobo/final_simulated/{scene}"
    for file in ["test"]:
        target_path = f"{path_to_folder}/xmls_depth"
        json_path = f"{path_to_folder}/transforms_{file}.json"
        template_path = f"{path_to_folder}/0_00_depth.xml"
        gen_xml(target_path=target_path, json_path=json_path, template_path=template_path, res=512, spp=4096)
    
    # # render
    # path = f"{path_to_folder}/xmls"
    # out_path = f"{path_to_folder}"
    # batch(path = path, out_path=out_path)

    
if __name__ == '__main__':
    
    render_transients()
