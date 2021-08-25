import os
import cv2
import yaml
import math 

def make_dirs(name):
    cache_dir = f"cache/{name}" 
    logs_dir = f"logs/{name}" 
    int_output_dir = f"int_output/{name}"
    results_dir = f"results/{name}"

    dirs = [cache_dir, logs_dir, int_output_dir, results_dir]

    for dir in dirs:
        if os.path.exists(dir) == True:
            os.system("rm -r " + dir)

        os.mkdir(dir)

    return cache_dir, logs_dir, int_output_dir, results_dir


def update_cfg(cfg,
               img_dir_path,
               name,
               output_bounds=(-5000, -5000, 5000, 5000),
               output_size=(4000, 4000),
               output_mode="overlay",
               hessian_threshold=100,
               optim_n_iter=2000,
               output_iter=[0, 1000, 1999],
               min_inliers=15, 
               max_dist = False
               ):
       
    img = cv2.imread([f"{img_dir_path}/{f}" for f in os.listdir(img_dir_path) if ".jpg" in f or ".JPG" in f][0])
    img_height, img_width = img.shape[0], img.shape[1]
    
    img_dir_name = img_dir_path.split("/")[-1]
    cache_dir_path, logs_dir_path, int_output_dir_path, results_dir_path = make_dirs(name)

    # Configure Paths
    cfg["in_dir_images"] = img_dir_path
    cfg["logs_dir"] = logs_dir_path
    cfg["show_file"] = int_output_dir_path
    
    cfg["error_map_path"] = logs_dir_path + "/error_map.jpg"
    cfg["gif_file"] = logs_dir_path + "/" + img_dir_name + ".gif"
    cfg["gif_folder"] = results_dir_path
    cfg["out_dir_cache"] = cache_dir_path
    cfg["out_dir_meta"] = results_dir_path
    cfg["in_dir_csv"] = f"{img_dir_path}/init.csv"

    # Configure output size
    cfg["output_bounds"] = output_bounds
    cfg["output_size"] = output_size
    cfg["mode"] = "overlay"

    # Configure stitiching algorithim parameters
    cfg["hessian_threshold"] = hessian_threshold
    cfg['optim_n_iter'] = optim_n_iter
    cfg["min_inliers"] = min_inliers
    cfg['output_iter'] = output_iter
    
    #Configure Image Height and Width
    cfg["img_height"] = img_height 
    cfg["img_width"] = img_width 
    
    #Configure Max Distance
    if max_dist != None:
        cfg["max_dist"] = math.sqrt(img_height ** 2 + img_width ** 2) * max_dist
    else:
        cfg["max_dist"] = None
    return cfg