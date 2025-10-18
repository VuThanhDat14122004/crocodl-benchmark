import argparse
import os
import random


def check_root_dir(paths):
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"{path} is not exist, please check again")

def check_destination_dir(paths):
    for path in paths:
        if not os.path.exists(path):
            if path.endswith('.txt'):
                with open(path, 'w') as f:
                    f.write("")
            else:
                os.makedirs(path, exist_ok=True)

def get_sample_data_single_cam(root_folder, des_folder, subsessions, img_list_dir, percent: float = 0.1,
                               images_txt_destination_dir: str = "", components_dict: dict = None,
                               dest_trajectory_folder: str = "", dict_trajectories: dict = None):
    for subsession in subsessions:
        imgs_subsession_list = [img_list_dir[i] for i in range(len(img_list_dir)) if subsession in img_list_dir[i]]
        total_images = len(imgs_subsession_list)
        number_of_images = int(total_images * percent)
        random_index = random.sample(range(total_images), number_of_images)
        selected_lines = [imgs_subsession_list[i] for i in random_index]
        # write image_dir consist subsession to images_txt_destination_dir
        with open(images_txt_destination_dir, 'a') as f:
            # check if first line is exist
            if os.path.getsize(images_txt_destination_dir) == 0:
                first_line="# timestamp, sensor_id, image_path\n"
                f.write(first_line)
            for idx in range(len(selected_lines)):
                img_path = root_folder + selected_lines[idx]
                des_path = des_folder + selected_lines[idx]
                f.write(components_dict[selected_lines[idx]])
                if not os.path.exists(os.path.dirname(des_path)):
                    os.makedirs(os.path.dirname(des_path), exist_ok=True)
                if not os.path.exists(des_path):
                    os.symlink(img_path, des_path)
        # write trajectory consist subsession to dest_trajectory_folder
        if dict_trajectories is None:
            continue
        with open(dest_trajectory_folder, 'a') as f:
            # check if first line is exist
            if os.path.getsize(dest_trajectory_folder) == 0:
                first_line="# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covar\n"
                f.write(first_line)
            for idx in range(len(selected_lines)):
                f.write(dict_trajectories[selected_lines[idx]])

def get_sample_data_multi_cam(root_folder, des_folder, img_list_dir, percent: float = 0.1,
                              images_txt_destination_dir: str = "", components_dict: dict = None,
                              dest_trajectory_folder: str = "", dict_trajectories: dict = None):
    set_subsession = set()
    for dir in img_list_dir:
        components = dir.split('/')
        sub_session = ""
        for i in range(len(components)-1):
            sub_session += components[i] + "/"
        set_subsession.add(sub_session)
    for subsession in set_subsession:
        #
        imgs_subsession_list = [img_list_dir[i] for i in range(len(img_list_dir)) if subsession in img_list_dir[i]]
        total_images = len(imgs_subsession_list)
        number_of_images = int(total_images * percent)
        random_index = random.sample(range(total_images), number_of_images)
        selected_lines = [imgs_subsession_list[i] for i in random_index]
        # write image_dir consist subsession to images_txt_destination_dir
        # write trajectory consist subsession to dest_trajectory_folder
        with open(images_txt_destination_dir, 'a') as fi:
            if os.path.getsize(images_txt_destination_dir) == 0:
                    first_line="# timestamp, sensor_id, image_path\n"
                    fi.write(first_line)
            if dict_trajectories is None:
                # check if first line is exist
                for idx in range(len(selected_lines)):
                    img_path = root_folder + selected_lines[idx]
                    des_path = des_folder + selected_lines[idx]
                    fi.write(components_dict[selected_lines[idx]])
                    if not os.path.exists(os.path.dirname(des_path)):
                        os.makedirs(os.path.dirname(des_path), exist_ok=True)
                    if not os.path.exists(des_path):
                        os.symlink(img_path, des_path)
            if dict_trajectories is not None:
                with open(dest_trajectory_folder, 'a') as f:
                    # check if first line is exist
                    if os.path.getsize(dest_trajectory_folder) == 0:
                        first_line="# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covar\n"
                        f.write(first_line)
                    for idx in range(len(selected_lines)):

                        try:
                            f.write(dict_trajectories[selected_lines[idx]])
                            img_path = root_folder + selected_lines[idx]
                            des_path = des_folder + selected_lines[idx]
                            fi.write(components_dict[selected_lines[idx]])
                            if not os.path.exists(os.path.dirname(des_path)):
                                os.makedirs(os.path.dirname(des_path), exist_ok=True)
                            if not os.path.exists(des_path):
                                os.symlink(img_path, des_path)
                        except:
                            continue
        

def get_cross_valid_data(root_dir: str, destination_dir: str, images_txt_root_dir: str,
                         images_txt_destination_dir: str, is_map: bool, multi_cam: bool,percent: float = 0.1):
    root_dir_raw_data = root_dir + "raw_data/"
    root_dir_proc = root_dir + "proc/"
    root_dir_images = root_dir + "images.txt"
    root_dir_sensors = root_dir + "sensors.txt"
    root_dir_rigs = root_dir
    root_dir_trajectories = root_dir
    if is_map:
        root_dir_trajectories = root_dir + "trajectories.txt"
    if multi_cam:
        root_dir_rigs = root_dir + "rigs.txt"
    check_root_dir([root_dir_raw_data, root_dir_proc, root_dir_images, root_dir_sensors, root_dir_rigs, root_dir_trajectories])
    
    dest_dir_raw_data = destination_dir + "raw_data/"
    dest_dir_proc = destination_dir + "proc/"
    dest_dir_images = destination_dir + "images.txt"
    dest_dir_sensors = destination_dir + "sensors.txt"
    dest_dir_rigs = destination_dir
    dest_dir_trajectories = destination_dir
    if is_map:
        dest_dir_trajectories = destination_dir + "trajectories.txt"
    if multi_cam:
        dest_dir_rigs = destination_dir + "rigs.txt"
    check_destination_dir([dest_dir_raw_data, dest_dir_proc, dest_dir_images, dest_dir_sensors, dest_dir_rigs, dest_dir_trajectories])
    with open(images_txt_root_dir, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    percent = percent
    subsessions = [lines[i].split(',')[1].split('/')[0].strip() for i in range(len(lines))]
    img_list_dir = []
    keys_spot = []
    if "spot" in root_dir_images:
        for i in range(len(lines)):
            keys_spot.append(lines[i].split(',')[0].strip())
    components_dict = {}
    for i in range(len(lines)):
        key = lines[i].split(',')[-1].strip()
        value = lines[i]
        components_dict.update({key: value})
        img_list_dir.append(key)

            
        
    trajectories_dict = None
    if is_map:
        if "spot" in root_dir_trajectories:
            trajectories_dict = {}
            with open(root_dir_trajectories, 'r') as f:
                lines_trajectories = f.readlines()
            lines_trajectories = lines_trajectories[1:]
            
            for i in range(len(keys_spot)):
                for line in lines_trajectories:
                    if keys_spot[i] in line:
                        trajectories_dict.update({img_list_dir[i]: line})
                        break
        else:
            trajectories_dict = {}
            with open(root_dir_trajectories, 'r') as f:
                lines_trajectories = f.readlines()
            lines_trajectories = lines_trajectories[1:]
                
            if '-' in img_list_dir[0].split('/')[-1]:
                for img_dir in img_list_dir:
                    timestamp = img_dir.split('/')[-1].split('-')[0].strip()
                    for line in lines_trajectories:
                        if timestamp in line:
                            trajectories_dict.update({img_dir: line})
                            break
            else:
                for img_dir in img_list_dir:
                    timestamp = img_dir.split('/')[-1].split('.')[0].strip()
                    for line in lines_trajectories:
                        if timestamp in line:
                            trajectories_dict.update({img_dir: line})
                            break
    if not multi_cam:
        get_sample_data_single_cam(root_dir_raw_data, dest_dir_raw_data, set(subsessions),
                                   img_list_dir, percent, images_txt_destination_dir, components_dict,
                                  dest_dir_trajectories, trajectories_dict)
    else:
        get_sample_data_multi_cam(root_dir_raw_data, dest_dir_raw_data, img_list_dir, percent, images_txt_destination_dir, components_dict,
                                  dest_dir_trajectories, trajectories_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--capture_dir', type=str, required=True)
    parser.add_argument('--capture_cross_valid_dir', type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    parser.add_argument('--session', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--images_txt', type=str, required=True)
    args = parser.parse_args().__dict__
    is_map=False
    multi_cam=False
    if args["device"] != 'ios':
        multi_cam = True
    if args['session'] == 'map':
        is_map = True
    root_dir = args["capture_dir"] + args["location"] + "/sessions/" + args["device"] + "_" + args["session"] + "/"
    destination_dir = args["capture_cross_valid_dir"] + args["location"] + "/sessions/" + args["device"] + "_" + args["session"] + "/"
    images_txt_root_dir = root_dir + args["images_txt"]
    print(f"images_txt_root_dir: {images_txt_root_dir}")
    images_txt_destination_dir = destination_dir + args["images_txt"]
    percent = 0.06
    get_cross_valid_data(root_dir, destination_dir, images_txt_root_dir, images_txt_destination_dir, is_map, multi_cam, percent)
    