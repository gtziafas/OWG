import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def grasp_to_mat(g):
    x, y, z, yaw, gripper_opening_length, obj_height = g
    pos = [x,y,z]
    rmat = R.from_euler('xyz', [yaw, np.pi/2, 0]).as_matrix()
    T = np.eye(4)
    T[:3,3] = pos
    T[:3,:3] = rmat
    return T


class Grasp2D:
    def __init__(self, W: int = 640, H: int = 480, normalized: bool = False):
        """
        Initialize a Grasp2D object as coordinates in an image frame.
        
        Args:
            W (int): Width of the image in pixels.
            H (int): Height of the image in pixels.
            normalized (bool): If True, coordinates will be normalized between (0,0) and (1,1). 
                               If False, coordinates will be in pixel values (0 to W, 0 to H).
        """
        self.W = W
        self.H = H
        self.normalized = normalized
        self._rectangle = None  # Will store the rectangle representation (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        self._vector = None     # Will store the vector representation (x,y,w,h,theta)

    @classmethod
    def from_vector(cls, x: float, y: float, w: float, h: float, theta: float, W: int, H: int, normalized: bool = True, line_offset=0.):
        """ Initialize the grasp from the vector representation. """
        grasp = cls(W, H, normalized)
        width = max(32, w+line_offset)
        length = max(15, h+line_offset)
        grasp._vector = (x, y, width, length, theta)
        grasp._rectangle = grasp._vector_to_rectangle(x, y, width, length, theta)
        return grasp

    @classmethod
    def from_rectangle(cls, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, x4: float, y4: float, W: int, H: int, normalized: bool = True):
        """ Initialize the grasp from the rectangle representation. """
        grasp = cls(W, H, normalized)
        grasp._rectangle = np.asarray([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
        grasp._vector = grasp._rectangle_to_vector((x1, y1), (x2, y2), (x3, y3), (x4, y4))
        return grasp

    @property
    def vector(self):
        """Return the grasp in vector form (x, y, w, h, theta)."""
        return self._vector

    @property
    def rectangle(self):
        """Return the grasp in rectangle form ((x1, y1), (x2, y2), (x3, y3), (x4, y4))."""
        return self._rectangle

    def _vector_to_rectangle(self, x: float, y: float, w: float, h: float, theta: float) -> np.ndarray:
        """ Convert vector (x, y, w, h, theta) to rectangle coordinates. """
        if self.normalized:
            x *= self.W
            y *= self.H
            w *= self.W
            h *= self.H

        R = np.array([[np.cos(theta), -np.sin(theta)], 
                      [np.sin(theta),  np.cos(theta)]])
        
        corners = np.array([[-w / 2, -h / 2], 
                            [w / 2, -h / 2], 
                            [w / 2,  h / 2], 
                            [-w / 2,  h / 2]])
        
        rotated_corners = (R @ corners.T).T + np.array([x, y])
        return np.asarray(tuple(map(tuple, rotated_corners)))

    def _rectangle_to_vector(self, p1: tuple, p2: tuple, p3: tuple, p4: tuple) -> tuple:
        """ Convert rectangle corner coordinates to vector (x, y, w, h, theta). """
        if self.normalized:
            p1 = (p1[0] * self.W, p1[1] * self.H)
            p2 = (p2[0] * self.W, p2[1] * self.H)
            p3 = (p3[0] * self.W, p3[1] * self.H)
            p4 = (p4[0] * self.W, p4[1] * self.H)

        x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
        y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4

        w = np.linalg.norm(np.array(p2) - np.array(p1))
        h = np.linalg.norm(np.array(p3) - np.array(p2))

        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        if self.normalized:
            x /= self.W
            y /= self.H
            w /= self.W
            h /= self.H

        return (x, y, w, h, theta)

    def rescale_to_crop(self, crop_box: tuple):
        """
        Rescale the grasp coordinates relative to a cropped box of the original image.
        
        Args:
            crop_box (tuple): (xc, yc, width, height) of the crop in image coordinates.
        
        Returns:
            Grasp2D: A new instance of Grasp2D with rescaled coordinates.
        """
        # Unpack the crop box
        xc, yc, crop_w, crop_h = crop_box
        
        # Rescale the rectangle coordinates relative to the crop box
        rescaled_rect = []
        for (x, y) in self._rectangle:
            new_x = x - xc
            new_y = y - yc
            rescaled_rect.append((new_x, new_y))
        
        # Convert the list back to numpy array
        rescaled_rect = np.asarray(rescaled_rect)
        
        # Recreate a new Grasp2D object with rescaled coordinates
        x1, y1 = rescaled_rect[0]
        x2, y2 = rescaled_rect[1]
        x3, y3 = rescaled_rect[2]
        x4, y4 = rescaled_rect[3]

        # Create a new Grasp2D object for the crop's size (new W, H)
        return Grasp2D.from_rectangle(x1, y1, x2, y2, x3, y3, x4, y4, crop_w, crop_h, normalized=False)

    def annotate_in_frame(self, frame: np.ndarray, as_line: bool = True, thickness: int = 2, color_side: tuple = (0,0,0xff), color_line: tuple = (0xff,0,0)) -> np.ndarray:
        """Draw grasp rectangles (or lines) in given image"""
        tmp = frame.copy()
        ptA, ptB, ptC, ptD = np.int32(self._rectangle)
        if as_line:
            ptAB = [ int(0.5 * (ptA[0] + ptB[0])), int(0.5 * (ptA[1] + ptB[1])) ]
            ptCD = [ int(0.5 * (ptC[0] + ptD[0])), int(0.5 * (ptC[1] + ptD[1])) ]
            tmp = cv2.line(tmp, ptAB, ptCD, color_line, thickness)
        else:
            tmp = cv2.line(tmp, ptA, ptD, color_line, thickness)
            tmp = cv2.line(tmp, ptB, ptC, color_line, thickness)
        tmp = cv2.line(tmp, ptA, ptB, color_side, thickness)
        tmp = cv2.line(tmp, ptD, ptC, color_side, thickness)
        return tmp

    def __repr__(self):
        """Return a detailed string representation of the grasp."""
        return f"Grasp2D(Vector: {self._vector}, Rectangle: {self._rectangle})"

    def __str__(self):
        """Return a nicely formatted string when printing the grasp."""
        vector = f"Vector (x, y, w, h, theta): {self._vector}"
        rectangle = f"Rectangle (x1, y1, x2, y2, x3, y3, x4, y4): {self._rectangle}"
        return f"Grasp2D:\n  {vector}\n  {rectangle}"
