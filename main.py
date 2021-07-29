import yaml
import time
import imageio
import glob
import os
import numpy as np

from src import vrt, stitch

np.set_printoptions(precision=4)


def run(cfg):
    """Runs the main script.

    Args:
        cfg (dict): all configurations

    Returns:
        src.vrt.VirtualRaster: the virtual raster built
    """
    tic = time.time()

    # step 1: create virtual raster
    print("Step 1")
    v = vrt.VirtualRaster.from_csv(
        file=cfg['in_dir_csv'],
        img_dir=cfg['in_dir_images'],
        wld_dir=cfg['out_dir_meta'],
        img_suffix=cfg['img_suffix'],
        wld_suffix=cfg['wld_suffix'],
        crs=cfg['crs'],
        index_cols=['index'])

    # step 2: build stitcher
    print("Step 2")
    s = stitch.Stitcher(
        scales=cfg['scales'],
        crop=cfg['stitch_crop'],
        cache_dir=cfg['out_dir_cache'],
        hessian_threshold=cfg['hessian_threshold'],
        min_inliers=cfg['min_inliers'],
        ransac_reproj_threshold=cfg['ransac_reproj_threshold'])

    # step 3: build links
    print("Step 3")
    v.build_graph_links(
        f=s.stitch_pair,
        method='all',
        show_file=cfg["show_file"],
        verbose=True)

    print(v.links)

    # step 4: globally optimize
    print("Step 4")
    v.global_optimize(
        n_iter=cfg['optim_n_iter'],
        lr_theta=cfg['optim_lr_theta'],
        lr_scale=cfg['optim_lr_scale'],
        lr_xy=cfg['optim_lr_xy'],
        lr_scheduler_milestones=cfg['optim_lr_scheduler_milestones'],
        logging=True,
        logdir=cfg["logs_dir"],
        output_iter=cfg['output_iter'])

    # step 5: visualize static images
    print("Step 5")
    for i, (n_iter, _, affine) in enumerate(zip(
            v.optim_iters, v.optim_losses, v.optim_affines)):
        v.df['relative_trans'] = affine
        v.georef()
        output, _ = v.show(
            verbose=True, crop=cfg['viz_crop'],
            mode=cfg['mode'], max_pixel=1e10,
            output_bounds=cfg["output_bounds"], output_size=cfg["output_size"])

        output = np.einsum('ijk->jki', output)
        imageio.imwrite(
            os.path.join(cfg['gif_folder'], '{}.jpg'.format(str(i))),
            output)

    # step 6: make gif
    print("Step 6")
    image_files = sorted(glob.glob(os.path.join(cfg['gif_folder'], '*.jpg')))
    images = [imageio.imread(f) for f in image_files]
    imageio.mimsave(cfg['gif_file'], images, duration=1)

    toc = time.time()
    print(f'Main function runtime: {((toc - tic) / 3600):.3f} hour(s)')
    return v


if __name__ == '__main__':
    # parse config file
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # run through all the steps
    v = run(cfg)