#Package Imports
import os
import utm
import cv2
import numpy as np
from math import radians, sin, cos, tan
from scipy.spatial.transform import Rotation as R
from utils.metadata import get_metadata, get_camera_specs, get_img_list


class PoseInit:
    def __init__(self, dir_path, resize_factor=1, cnt_utmy=None, cnt_utmx=None, verbose=True):
        self.md_dict = get_metadata(dir_path)
        sensor_dim, focal_len = get_camera_specs(self.md_dict["model"][0])

        self.sensor_dim = sensor_dim
        self.focal_len = focal_len

        self.cnt_utmy = cnt_utmy
        self.cnt_utmx = cnt_utmx

        self.img_list = get_img_list(dir_path, self.md_dict["name"])
        self.img_shape = self.img_list[0].shape

        utmx_og_list, utmy_og_list, utmx_list, utmy_list, utm_zone_number, utm_zone_letter = self.get_utm_cood()
        self.md_dict["utmx og"] = utmx_og_list
        self.md_dict["utmy og"] = utmy_og_list
        self.md_dict["utmx"] = utmx_list
        self.md_dict["utmy"] = utmy_list
        self.md_dict["utm_zone_number"] = utm_zone_number
        self.md_dict["utm_zone_letter"] = utm_zone_letter

        gsd_w, gsd_h = self.get_gsd()
        self.md_dict["gsd_w"] = gsd_w
        self.md_dict["gsd_h"] = gsd_h

        x_list, y_list = self.get_init_img_cood()
        self.md_dict["x"] = x_list
        self.md_dict["y"] = y_list

        self.abs_trans_list = self.get_abs_trans_list()
        self.rot_trans_list = self.get_rot_trans_list()

        self.verbose = verbose
        if verbose:
            table_dict = {}
            for key in self.md_dict.keys():
                val = self.md_dict[key]
                if isinstance(val, list):
                    name_val_tup = list(zip(self.md_dict["name"], val))
                    table_dict[key] = val


    def get_utm_cood(self):
        utm_cood_list = [utm.from_latlon(lat, lng) for lat, lng in
                         zip(self.md_dict["latitude"], self.md_dict["longitude"])]

        utm_zone_number = utm_cood_list[0][2]
        utm_zone_letter = utm_cood_list[0][3]

        utmx_og_list, utmy_og_list, utmx_list, utmy_list = [], [], [], []

        for flight_yaw, flight_pitch, flight_roll, cam_yaw, cam_pitch, cam_roll, alt, utm_cood in zip(
                self.md_dict['flight yaw'],
                self.md_dict["flight pitch"],
                self.md_dict["flight roll"],
                self.md_dict['gimbal yaw'],
                self.md_dict["gimbal pitch"],
                self.md_dict["gimbal roll"],
                self.md_dict['relative altitude'],
                utm_cood_list):

            flight_rot = R.from_euler("ZYX", [flight_yaw, flight_pitch, flight_roll], degrees=True)
            cam_rot = R.from_euler("ZYX", [cam_yaw, cam_pitch + 90, cam_roll], degrees=True)
            refl_mat = R.from_matrix(np.array([[1, 0, 0],
                                               [0, -1, 0],
                                               [0, 0, 1]]))

            total_rot = refl_mat * flight_rot * cam_rot

            vec = total_rot.apply([0, 0, alt])
            res = LinePlaneCollision(np.array([0, 0, 1]),
                                     np.array([0, 0, 0]),
                                     np.array(vec), np.array([0, 0, alt]))

            utmx = utm_cood[0] + res[0]
            utmy = utm_cood[1] + res[1]

            utmx_og_list.append(utm_cood[0])
            utmy_og_list.append(utm_cood[1])

            utmx_list.append(utmx)
            utmy_list.append(utmy)

        return utmx_og_list, utmy_og_list, utmx_list, utmy_list, utm_zone_number, utm_zone_letter

    def get_gsd(self):
        alt_cm = np.array(self.md_dict["relative altitude"]) * 100
        gsd_w = (alt_cm * self.sensor_dim[0]) / (self.focal_len * self.img_shape[1])
        gsd_h = (alt_cm * self.sensor_dim[1]) / (self.focal_len * self.img_shape[0])

        return gsd_w.tolist(), gsd_h.tolist()

    def get_cnt_cood(self):
        return (self.cnt_lat, self.cnt_lng)

    def get_init_img_cood(self):

        if self.cnt_utmx == None and self.cnt_utmy == None:
            min_utmx, max_utmx, min_utmy, max_utmy = min(self.md_dict["utmx"]), max(self.md_dict["utmx"]), min(
                self.md_dict["utmy"]), max(self.md_dict["utmy"])
            min_utmx_met, max_utmx_met, min_utmy_met, max_utmy_met = float(str(min_utmx)[3:]), float(
                str(max_utmx)[3:]), float(str(min_utmy)[3:]), float(str(max_utmy)[3:])
            self.cnt_utmx, self.cnt_utmy = (max_utmx_met + min_utmx_met) / 2, (max_utmy_met + min_utmy_met) / 2

        x_list = []
        y_list = []
        for utmx, utmy, gsd_w, gsd_h in zip(self.md_dict["utmx"], self.md_dict["utmy"], self.md_dict["gsd_w"],
                                            self.md_dict["gsd_h"]):
            utmx_met = float(str(utmx)[3:])
            utmy_met = float(str(utmy)[3:])
            utmx_dist = (utmx_met - self.cnt_utmx) * 100
            utmy_dist = (utmy_met - self.cnt_utmy) * 100
            utmx_dist = -1 * utmx_dist if utmx < self.cnt_utmx else utmx_dist
            utmy_dist = -1 * utmy_dist if utmy < self.cnt_utmy else utmy_dist
            utmy_dist_pix = -utmy_dist / gsd_h
            utmx_dist_pix = utmx_dist / gsd_w

            x_list.append(utmx_dist_pix)
            y_list.append(utmy_dist_pix)

        return x_list, y_list

    def get_abs_trans_list(self):
        abs_transform_list = []

        for x, y, yaw, pitch, roll, cam_yaw, cam_pitch, cam_roll in zip(self.md_dict["x"], self.md_dict["y"],
                                                                        self.md_dict["flight yaw"],
                                                                        self.md_dict["flight pitch"],
                                                                        self.md_dict["flight roll"],
                                                                        self.md_dict["gimbal yaw"],
                                                                        self.md_dict["gimbal pitch"],
                                                                        self.md_dict["gimbal roll"]):
            yaw, pitch, roll, cam_yaw, cam_pitch, cam_roll = radians(yaw), radians(pitch), radians(roll), radians(
                cam_yaw), radians(cam_pitch + 90), radians(cam_roll)

            center_translation = np.array([
                [1, 0, 0, -self.img_shape[1] / 2],
                [0, 1, 0, -self.img_shape[0] / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            refl_mat = R.from_matrix(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

            flight_rot = R.from_euler("ZYX", [yaw, pitch, roll], degrees=False)
            cam_rot = R.from_euler("ZYX", [cam_yaw, cam_pitch, cam_roll], degrees=False)

            total_rot = (flight_rot * cam_rot).as_matrix()

            # Extrinsic Matrix
            ext_mat = np.array([
                [total_rot[0, 0], total_rot[0, 1], total_rot[0, 2], x],
                [total_rot[1, 0], total_rot[1, 1], total_rot[1, 2], y],
                [total_rot[2, 0], total_rot[2, 1], total_rot[2, 2], 0],
                [0, 0, 0, 1]
            ])

            sw, sh = self.sensor_dim
            mult = sw / sh

            # Intrisic Matrix
            int_mat = np.array([
                [1, 0, 0, 3000],
                [0, 1, 0, 3000],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            abs_transform = int_mat.dot(ext_mat.dot(center_translation)).astype("float32")

            abs_transform_2D = np.array([
                [abs_transform[0, 0], abs_transform[0, 1], abs_transform[0, 3]],
                [abs_transform[1, 0], abs_transform[1, 1], abs_transform[1, 3]],
                [0, 0, 1],
            ])

            abs_transform_list.append(abs_transform_2D)

        return abs_transform_list

    def get_rot_trans_list(self):
        rot_trans_list = []

        for flight_yaw, flight_pitch, flight_roll, cam_yaw, cam_pitch, cam_roll in zip(self.md_dict['flight yaw'],
                                                                                       self.md_dict["flight pitch"],
                                                                                       self.md_dict["flight roll"],
                                                                                       self.md_dict['gimbal yaw'],
                                                                                       self.md_dict["gimbal pitch"],
                                                                                       self.md_dict["gimbal roll"]):
            center_translation = np.array([
                [1, 0, 0, -self.img_shape[1] / 2],
                [0, 1, 0, -self.img_shape[0] / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            rev_center_translation = np.array([
                [1, 0, 0, self.img_shape[1] / 2],
                [0, 1, 0, self.img_shape[0] / 2],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            flight_rot = R.from_euler("ZYX", [flight_yaw, flight_pitch, flight_roll], degrees=True)
            cam_rot = R.from_euler("ZYX", [cam_yaw, cam_pitch + 90, cam_roll], degrees=True)
            refl_mat = R.from_matrix(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]))

            total_rot = (flight_rot * cam_rot).as_matrix()

            ext_mat = np.array([
                [total_rot[0, 0], total_rot[0, 1], total_rot[0, 2], 0],
                [total_rot[1, 0], total_rot[1, 1], total_rot[1, 2], 0],
                [total_rot[2, 0], total_rot[2, 1], total_rot[2, 2], 0],
                [0, 0, 0, 1]
            ])

            rot_transform = rev_center_translation.dot(ext_mat.dot(center_translation)).astype("float32")

            rot_trans_list.append(rot_transform)

        return rot_trans_list

    def get_md_dict(self):
        return self.md_dict

    def get_img_list(self):
        return self.img_list

    def get_img_shape(self):
        return self.img_shape


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
 
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")
 
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi
