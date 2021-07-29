import os
import yaml


def make_dirs(name):
    cache_dir = "cache/cache_" + name
    logs_dir = "logs/logs_" + name
    int_output_dir = "int_output/int_output_" + name
    results_dir = "results/results_" + name

    dirs = [cache_dir, logs_dir, int_output_dir, results_dir]

    for dir in dirs:
        if os.path.exists(dir) == True:
            os.system("rm -r " + dir)

        os.mkdir(dir)

    return cache_dir, logs_dir, int_output_dir, results_dir


def update_cfg(cfg,
               img_dir_path,
               output_bounds=(-5000, -5000, 5000, 5000),
               output_size=(4000, 4000),
               output_mode="overlay",
               hessian_threshold=100,
               optim_n_iter=10000,
               output_iter=[0, 2000, 4000, 6000, 8000, 9999],
               min_inliers=15
               ):
    img_dir_name = img_dir_path.split("/")[-1]
    cache_dir_path, logs_dir_path, int_output_dir_path, results_dir_path = make_dirs(img_dir_name)

    # Configure Paths
    cfg["in_dir_images"] = img_dir_path
    cfg["logs_dir"] = logs_dir_path
    cfg["show_file"] = int_output_dir_path
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

    return cfg