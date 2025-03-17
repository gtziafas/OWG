import pybullet as p
import datetime
import numpy as np
import math

class Camera2:
    def __init__(self, 
                 position, 
                 orientation, 
                 image_size = (720, 720),
                 intrinsics = (360., 0, 360., 0, 360., 360., 0, 0, 1),
                 zrange = (0.01, 10.),
                 noise=True,
                 shadow=1,
    ):
        # Camera parameters.
        self.x, self.y, self.z = position
        self.position = position
        self.euler = orientation
        self.quaternion = p.getQuaternionFromEuler(self.euler)
        self.rotmat = p.getMatrixFromQuaternion(self.quaternion)
        self.zrange = zrange
        self.znear, self.zfar = self.zrange
        self.noise = noise
        self.shadow = shadow

        self.intrinsics  = intrinsics
        self.focal_len = intrinsics[0]
        self.image_size = image_size
       
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotm = np.float32(self.rotmat).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = position + lookdir
        
        self.view_matrix = p.computeViewMatrix(position, lookat, updir)
        self.fovh = (image_size[0] / 2) / self.focal_len
        self.fovh = 180 * np.arctan(self.fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        self.aspect_ratio = image_size[1] / image_size[0]
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fovh, self.aspect_ratio, self.znear, self.zfar)

        self.rec_id = None

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Render with OpenGL camera settings.
        _, _, color, depth, segm = p.getCameraImage(
            width=self.image_size[1],
            height=self.image_size[0],
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            shadow=1,
            #flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            #renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return color[..., :3], depth, segm
        image_size = self.image_size
        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if self.noise:
          color = np.int32(color)
          color += np.int32(np.random.normal(0, 3, color.shape))
          color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (image_size[0], image_size[1])
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (self.zfar + self.znear - (2 * zbuffer - 1) * (self.zfar - self.znear))
        depth = (2 * self.znear * self.zfar) / depth
        if self.noise:
          depth += np.random.normal(0, 0.003, depth.shape)
        return color, depth, segm

class Camera:
    def __init__(self, cam_pos, cam_target, near, far, size, fov):
        self.x, self.y, self.z = cam_pos
        self.x_t, self.y_t, self.z_t = cam_target
        self.width, self.height = size[0:2]
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov, aspect, near, far)
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_target, [0, 1, 0])

        # Compute intrinsic parameters
        self.fx = self.height / (2 * math.tan(math.radians(fov) / 2))
        self.fy = self.fx * aspect
        self.cx = self.width / 2
        self.cy = self.height / 2
        self.intrinsics = np.array([self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3,3)

        self.rec_id = None

    def get_cam_img(self):
        """
        Method to get images from camera
        return:
        rgb
        depth
        segmentation mask
        """
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   # renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                   )
        return rgb[:, :, 0:3], depth, seg

    def get_pointcloud(self):
        """Get 3D pointcloud from perspective depth image.
        Args:xyz
          depth: HxW float array of perspective depth in meters.
          intrinsics: 3x3 float array of camera intrinsics matrix.
        Returns:
          points: HxWx3 float array of 3D points in camera coordinates.
        """
        _, depth, _ = self.get_cam_img()
        depth_image_size = depth.shape
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (self.far + self.near - (2 * zbuffer - 1) * (self.far - self.near))
        depth = (2 * self.near * self.far) / depth
        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - self.intrinsics[0, 2]) * (depth / self.intrinsics[0, 0])
        py = (py - self.intrinsics[1, 2]) * (depth / self.intrinsics[1, 1])
        points = np.float32([px, py, depth]).transpose(1, 2, 0)
        return points

    def start_recording(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file = f'{save_dir}/{now}.mp4'

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.rec_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, file)

    def stop_recording(self):
        p.stopStateLogging(self.rec_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)