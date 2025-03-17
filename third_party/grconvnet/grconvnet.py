from third_party.grconvnet.network.hardware.device import get_device
#from network.inference.post_process import post_process_output
from third_party.grconvnet.network.utils.data.camera_data import CameraData
from third_party.grconvnet.network.utils.visualisation.plot import plot_results
from third_party.grconvnet.network.utils.dataset_processing.grasp import detect_grasps
from skimage.filters import gaussian
import matplotlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from datetime import datetime

DEPTH_RADIUS = 1
IMG_SIZE = 224

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class GraspGenerator:
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius, fig, IMG_WIDTH=224, network='GR_ConvNet', device='cpu'):

        cwd = os.getcwd()
        os.chdir(f'./third_party/grconvnet/checkpoints/{network}')
        self.net = torch.load(net_path,  map_location=device)
        self.device = get_device(force_cpu=True if device == 'cpu' else False)
        os.chdir(cwd)
        assert os.getcwd() == cwd
        
        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius
        
        self.fig = fig
        self.network = network

        self.PIX_CONVERSION = 277 * IMG_WIDTH/224

        self.IMG_WIDTH = IMG_WIDTH
        print (IMG_WIDTH, self.IMG_WIDTH)

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(-img_center/self.PIX_CONVERSION,
                                                    img_center/self.PIX_CONVERSION,
                                                    0,
                                                    self.IMG_ROTATION)
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION)

    def get_transform_matrix(self, x, y, z, rot):
        return np.array([
                        [np.cos(rot),   -np.sin(rot),   0,  x],
                        [np.sin(rot),   np.cos(rot),    0,  y],
                        [0,             0,              1,  z],
                        [0,             0,              0,  1]
                        ])

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p-self.depth_r, 0, self.IMG_WIDTH).astype(np.int32)
        x_max = np.clip(x_p+self.depth_r, 0, self.IMG_WIDTH).astype(np.int32)
        y_min = np.clip(y_p-self.depth_r, 0, self.IMG_WIDTH).astype(np.int32)
        y_max = np.clip(y_p+self.depth_r, 0, self.IMG_WIDTH).astype(np.int32)
        
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (grasp.length / int(self.MAX_GRASP *
                          self.PIX_CONVERSION)) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        robot_pose = robot_frame_ref[0], robot_frame_ref[1], robot_frame_ref[2], roll, opening_length, obj_height
        grasp_2d = [grasp.center[0], grasp.center[1], roll, grasp.length, grasp.width]
        return robot_pose, grasp_2d

    def post_process_output(self, q_img, cos_img, sin_img, width_img, pixels_max_grasp):
        """
        Post-process the raw output of the network, convert to numpy arrays, apply filtering.
        :param q_img: Q output of network (as torch Tensors)
        :param cos_img: cos output of network
        :param sin_img: sin output of network
        :param width_img: Width output of network
        :return: Filtered Q output, Filtered Angle output, Filtered Width output
        """
        q_img = q_img.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin_img, cos_img) / 2.0).cpu().numpy().squeeze()
        width_img = width_img.cpu().numpy().squeeze() * pixels_max_grasp

        q_img = gaussian(q_img, 1.0, preserve_range=True)
        ang_img = gaussian(ang_img, 1.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)

        return q_img, ang_img, width_img

    def predict(self, rgb, depth, n_grasps=1, show_output=False):

        max_val = np.max(depth)
        depth = depth * (255 / max_val)
        depth = np.clip((depth - depth.mean())/175, -1, 1)
        
        if (self.network == 'GR_ConvNet'):
            ##### GR-ConvNet #####
            depth = np.expand_dims(np.array(depth), axis=2)
            img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
            x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
        else:
            print("The selected network has not been implemented yet -- please choose another network!")
            exit() 

        with torch.no_grad():
            xc = x.to(self.device)

            if (self.network == 'GR_ConvNet'):
                ##### GR-ConvNet #####
                pred = self.net.predict(xc)
                # print (pred)
                pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
                q_img, ang_img, width_img = self.post_process_output(pred['pos'],
                                                                pred['cos'],
                                                                pred['sin'],
                                                                pred['width'],
                                                                pixels_max_grasp)
            else: 
                print ("you need to add your function here!")        
        
        save_name = None
        if show_output:
            #fig = plt.figure(figsize=(10, 10))
            #im_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            plot = plot_results(self.fig,
                                rgb_img=rgb,
                                grasp_q_img=q_img,
                                grasp_angle_img=ang_img,
                                depth_img=depth,
                                no_grasps=3,
                                grasp_width_img=width_img)

            if not os.path.exists('network_output'):
                os.mkdir('network_output')
            time = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
            save_name = 'network_output/{}'.format(time)
            plot.savefig(save_name + '.png')
            plot.clf()

        grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=n_grasps)
        return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False):
        predictions, save_name = self.predict(rgb, depth, n_grasps=n_grasps, show_output=show_output)
        grasps, grasp_rects = [], []
        for grasp in predictions:
            #cx, cy = grasp.center; angle, w, h = grasp.angle, grasp.width, grasp.length
            #x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(grasp, depth)
            grasp_pose, grasp_rect = self.grasp_to_robot_frame(grasp, depth)
            grasps.append(grasp_pose)
            grasp_rects.append(grasp_rect)
        return grasps, grasp_rects, save_name

    def predict_grasp_from_mask(self, rgb, depth, mask, n_grasps=1, show_output=False):
        mask_3 = np.expand_dims(mask, 2).repeat(3, 2)
        mask_rgb = np.where(mask_3, rgb.copy(), np.ones_like(rgb) * 0xff)
        mask_d = np.where(mask, depth.copy(), np.ones_like(depth) * depth.max())
        grasps3d, grasp_rects, _ = self.predict_grasp(mask_rgb, mask_d, n_grasps, show_output)
        return grasps3d, grasp_rects


def load_grasp_generator(camera, img_size=224, checkpoint="cornell-randsplit-rgbd-grconvnet3-drop1-ch32/epoch_19_iou_0.98"):
    fig = plt.figure(figsize=(10,10)); 
    return GraspGenerator(checkpoint, camera, DEPTH_RADIUS, fig, IMG_WIDTH=img_size, network='GR_ConvNet', device=device)