import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch
from matplotlib import colors
import cv2
import numpy as np
import torch
from PIL import Image, ImageColor
from math import sqrt
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans
import string

import supervision as sv
from supervision.annotators.base import BaseAnnotator
from supervision.annotators.utils import ColorLookup, Trace, resolve_color
from supervision.detection.core import Detections
from supervision.detection.utils import clip_boxes, mask_to_polygons
from supervision.draw.color import Color, ColorPalette
from supervision.draw.utils import draw_polygon
from supervision.geometry.core import Position

from owg.markers.postprocessing import masks_to_marks
from owg.utils.config import load_config
from owg.utils.image import compute_mask_center_of_mass, compute_mask_bounding_box, compute_mask_contour
from owg.utils.grasp import Grasp2D


# helper function
def display_image(path_or_array, size=(10, 10)):
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, 'rb')).convert("RGB"))
  else:
    image = path_or_array
  
  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis('off')
  plt.show()


# MY_PALETTE =  [
#     "#66FF66",  # Light green
#     "#BB00FF",  # Light red
#     "#9999FF",  # New light blue
#     "#FFFF66",  # Light yellow
#     "#FF66FF",  # Light magenta
#     "#FFB266",  # Light orange,
#     "#66FFFF",  # New light cyan
#     "#BF40BF",  # Light purple
#     "#FFCC99",  # Light brown
#     "#33CCFF",  # New light navy
#     "#99FF99",  # Light lime
#     "#FF99C8",  # Light pink
#     "#C0C0C0",  # Light gray
#     "#66CCCC",  # Light teal,
#     "#CCCC66",  # Light olive
#     "#FF9999",  # Light maroon
#     "#9999FF"   # Light indigo (kept as is, since it was not too similar)
# ]


MY_PALETTE = [
    "66FF66",  # red
    "FF66FF",  # green
    "FF6666",  # blue
    "CCFFFF",  # yellow
    "E0E080",  # purple
    "E0F3D7",  # pink
    "D7FF80",  # orange
    "A5D780",  # brown
    "D0D0C0",  # gray
    "E0E0D0",  # silver
    "D7FFFF",  # gold
    "FFD7FF",  # lavender
    "FF80AA",  # turquoise
]



def assign_colors(point_centers, palette=MY_PALETTE):
    # Define a function to calculate the distance between colors
    def color_distance(c1, c2):
        # Convert hex color to RGB
        rgb1 = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
        rgb2 = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
        # Calculate the Euclidean distance between the two colors
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


    # Cluster the points to find groups of nearby points
    num_clusters = min(len(point_centers), len(palette))
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(point_centers)

    # Sort the palette by luminance to maximize the color differences
    palette.sort(key=lambda color: sum(int(color[i:i+2], 16) for i in (1, 3, 5)))

    # Assign colors to clusters ensuring that similar colors are not used for adjacent clusters
    assigned_colors = {}
    for i in range(num_clusters):
        assigned_colors[i] = palette[i % len(palette)]

    # Map the cluster labels to colors
    color_assignment = [assigned_colors[label] for label in labels]

    return color_assignment


# AVAILABLE_CROP_METHODS = [
# 	"default", 
# 	"mask-grid", 
# 	"mask-grid-hi",
# 	"maskbox-grid", 
# 	"maskbox-grid-hi",
# 	"mask", 
# 	"mask-hi",
# 	"grid", 
# 	"grid-hi",
# 	"box-grid", "box-grid-hi"
# ]

AVAILABLE_MARKER_METHODS = [
	"default", 
	"SoM",
	"RoI"
]

CROP_RES = 224
CROP_RES_HIGH = 896

def background_color(rgb):
    # Calculate the perceived luminance of the color
    # using the formula: 0.299*R + 0.587*G + 0.114*B
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    # Return 'black' for light colors and 'white' for dark colors
    return Color.black() if luminance > 128 else Color.white()


@dataclass
class MyColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls) -> ColorPalette:
        """
        Returns a default color palette.

        Returns:
            ColorPalette: A ColorPalette instance with default colors.

        Example:
            ```
            >>> ColorPalette.default()
            ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), ...])
            ```
        """
        return ColorPalette.from_hex(color_hex_list=MY_PALETTE)

class GraspVisualizer:
    """
    A class for visualizing 4-DoF grasp annotations in a cropped image

    Parameters:
    """
    def __init__(
        self,
        as_line: bool = True,
        grasp_colors: List[Color] = [Color.red(), Color.blue()],
        mark_color: Color = Color.green(),
        line_thickness: int = 2,
        text_color: Color = Color.white(),
        text_rect_color: Color = Color.black(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 2,
        text_position: str = "CENTER",
        with_label: bool = False,
        with_box: bool = False,
        with_mask: bool = False,
        with_polygon: bool = False,
        with_gray: bool = False,
        mask_opacity: float = 0.1,
        resize_dim: Optional[Tuple[int]] = None
    ):
        """
        Args:
            as_line (bool): Whether to draw grasp as line (True) or rectangle (False)
            line_thickness (int): Thickness of the grasp, box and polygon lines
            grasp_colors (List[Color])): The two colors to use for grasp annotations. 
                Defaults to red for approach and blue for fingers.
            mark_color (Color): The color to use for box, mask and polygon annotation
            text_color (Color): The color to use for the text label annotation
            text_rect_color (Color): The color to use for the text background rectangle.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (str): Position of the text relative to the detection.
                Possible values are ("CENTER", "BOTTOM_LEFT", "TOP_RIGHT")
            with_label (bool): Whether to draw grasp label ID. Defaults to False.
            with_box (bool): Whether to draw bounding box around object. Defaults to False.
            with_gray (bool): Whether to paint the background. Defaults to False.
            with_mask (bool): Whether to draw mask of object. Defaults to False.
            with_polygon (bool): Whether to draw mask contour around object. Defaults to False.
            mask_opacity (float): Opacity for drawing mask in object
            resize_dim (Optional[Tuple[int]]): If specified, will resize the marked cropped image
                before overlaying text labels for better visual clarity.
        """
        self.grasp_colors: List[Color] = grasp_colors
        self.text_rect_color: Color = text_rect_color
        self.text_color: Color = text_color
        self.mark_color: Color = mark_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_position: str = text_position
        self.as_line: bool = as_line
        self.line_thickness: int = line_thickness
        self.with_box: bool = with_box
        self.with_mask: bool = with_mask
        self.with_polygon: bool = with_polygon
        self.with_label: bool = with_label
        self.with_gray: bool = with_gray
        self.resize_dim: Optional[tuple] = resize_dim
        # other mark visualizers
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=mask_opacity)
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)

    def annotate_labels(self, image, labels):
        img = image.copy()

        for i, label in enumerate(labels):
            text = str(label)
            text_color = self.text_color[i]

            # Calculate text size
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)[0]
            text_width, text_height = text_size[0], text_size[1]

            # Calculate bottom-left corner of the text
            position  = self.text_position
            if position == 'BOTTOM_RIGHT':
                bottom_left_corner = (point[0], point[1] + text_height)
            elif position == 'CENTER':
                bottom_left_corner = (point[0] - text_width // 2, point[1] + text_height // 2)
            elif position == 'TOP_LEFT':
                bottom_left_corner = (point[0] - text_width, point[1])

            # Calculate coordinates of the rectangle
            top_left_rect = (bottom_left_corner[0], bottom_left_corner[1] - text_height)
            bottom_right_rect = (bottom_left_corner[0] + text_width, bottom_left_corner[1])

            # pad the rect
            text_padding = self.text_padding
            top_left_rect = [max(0, top_left_rect[0] - text_padding), max(0, top_left_rect[1] - text_padding)]
            bottom_right_rect = [min(img.shape[0], bottom_right_rect[0] + text_padding), min(img.shape[1], bottom_right_rect[1] + text_padding)]
            
            # Draw the rectangle
            cv2.rectangle(
                    img=img,
                    pt1=top_left_rect,
                    pt2=bottom_right_rect,
                    color = self.text_rect_color.as_rgb(),
                    thickness=cv2.FILLED,
            )

            # Draw the text
            cv2.putText(
                img, 
                text, 
                bottom_left_corner, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                self.text_scale, 
                text_color, 
                self.text_thickness, 
                lineType=cv2.LINE_AA
            ) 
            
        return img

    def visualize(
        self, 
        image: np.ndarray, 
        grasps: List[Grasp2D],
        mask: Optional[np.ndarray] = None,
        labels: Optional[List[int]] = None,
        with_box: Optional[bool] = None,
        with_mask: Optional[bool] = None,
        with_polygon: Optional[bool] = None,
        with_label: Optional[bool] = None,
        with_gray: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Visualizes 4-DoF grasps on an cropped image.

        This method takes an image and an list of owg.utils.grasp.Grasp2D, and overlays
        the grasps on the image, labeled with text and potentially other information (mask,bbox).

        Parameters:
            image (np.ndarray): The cropped image on which to overlay annotations.
            grasps (List[Grasp2D]): A list of 4-DoF grasp annotations
            mask (Optional[np.ndarray]): The mask of the object for drawing boxes, masks and polygons
            labels (Optional[List[int]]): Potentially give list of text labels to annotate
            with_box (bool): Whether to draw bounding box around object. Defaults to False.
            with_mask (bool): Whether to draw mask of object. Defaults to False.
            with_polygon (bool): Whether to draw mask contour around object. Defaults to False.
            with_label (bool): Whether to draw grasp label ID. Defaults to False.
            with_gray (bool): Whether to paint the background. Defaults to False.
            
        Returns:
            np.ndarray: The annotated image.
        """
        with_box = with_box or self.with_box
        with_mask = with_mask or self.with_mask
        with_polygon = with_polygon or self.with_polygon
        with_label = with_label or self.with_label
        with_gray = with_gray or self.with_gray

        # single marker from object mask
        if with_mask or with_box or with_polygon:
            assert mask is not None, "Must provide object mask for using with_{box,mask,polygon}=True"
            marks = masks_to_marks(mask[None, ...])

        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_gray:
            gray_image = cv2.cvtColor(
                annotated_image, cv2.COLOR_RGB2GRAY)
            gray_image = cv2.merge([gray_image, gray_image, gray_image])
            annotated_image[mask==False] = gray_image[mask==False]

        for grasp_index, grasp in enumerate(grasps):
            if labels is not None:
                text = str(labels[grasp_index])    
            else:
                text = str(grasp_index+1)
            
            text_color = self.text_color.as_rgb()

            # Calculate text size
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness)[0]
            text_width, text_height = text_size[0], text_size[1]

            # Calculate bottom-left corner of the text
            point = grasp.vector[0:2]
            position  = self.text_position
            if position == 'BOTTOM_RIGHT':
                bottom_left_corner = (point[0], point[1] + text_height)
            elif position == 'CENTER':
                bottom_left_corner = (point[0] - text_width // 2, point[1] + text_height // 2)
            elif position == 'TOP_LEFT':
                bottom_left_corner = (point[0] - text_width, point[1])

            # Calculate coordinates of the rectangle
            top_left_rect = (bottom_left_corner[0], bottom_left_corner[1] - text_height)
            bottom_right_rect = (bottom_left_corner[0] + text_width, bottom_left_corner[1])

            # pad the rect
            text_padding = self.text_padding
            top_left_rect = [max(0, top_left_rect[0] - text_padding), max(0, top_left_rect[1] - text_padding)]
            bottom_right_rect = [min(annotated_image.shape[0], bottom_right_rect[0] + text_padding), min(annotated_image.shape[1], bottom_right_rect[1] + text_padding)]
            
            # Draw the grasp
            annotated_image = grasp.annotate_in_frame(
                annotated_image, 
                as_line=self.as_line,
                thickness=self.line_thickness,
                color_side=self.grasp_colors[1].as_rgb(),
                color_line=self.grasp_colors[0].as_rgb(),
            )

            if with_label:
                #Draw the rectangle
                annotated_image = cv2.rectangle(
                        img=annotated_image,
                        pt1=list(map(int, top_left_rect)),
                        pt2=list(map(int, bottom_right_rect)),
                        color = self.text_rect_color.as_rgb(),
                        thickness=cv2.FILLED,
                )

                #Draw the text
                annotated_image = cv2.putText(
                    annotated_image, 
                    text, 
                    list(map(int, bottom_left_corner)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    self.text_scale, 
                    text_color, 
                    self.text_thickness, 
                    lineType=cv2.LINE_AA
                ) 
            
        return annotated_image

class LabelAnnotator:
    """
    A class for annotating labels on an image using provided detections.
    """

    def __init__(
        self,
        color: Color = Color.black(),
        text_color: Union[Color, ColorPalette] = ColorPalette.default(),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: Position = Position.CENTER_OF_MASS,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[Color, ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_padding (int): Padding around the text within its background box.
            text_position (Position): Position of the text relative to the detection.
                Possible values are defined in the `Position` enum.
            color_lookup (str): Strategy for mapping colors to annotations.
                Options are `INDEX`, `CLASS`, `TRACK`.
        """
        self.text_color: Union[Color, ColorPalette] = text_color
        self.color: Color = color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.text_anchor: Position = text_position
        self.color_lookup: ColorLookup = color_lookup

    @staticmethod
    def resolve_text_background_xyxy_dist(
        binary_mask: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask.astype(np.uint8) * 255, 
            cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]
        return coords_x[0], coords_y[0]


    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates: Tuple[int, int],
        text_wh: Tuple[int, int],
        position: Position,
    ) -> Tuple[int, int, int, int]:
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh

        if position == Position.TOP_LEFT:
            return center_x, center_y - text_h, center_x + text_w, center_y
        elif position == Position.TOP_RIGHT:
            return center_x - text_w, center_y - text_h, center_x, center_y
        elif position == Position.TOP_CENTER:
            return (
                center_x - text_w // 2,
                center_y - text_h,
                center_x + text_w // 2,
                center_y,
            )
        elif position == Position.CENTER or position == Position.CENTER_OF_MASS:
            return (
                center_x - text_w // 2,
                center_y - text_h // 2,
                center_x + text_w // 2,
                center_y + text_h // 2,
            )
        elif position == Position.BOTTOM_LEFT:
            return center_x, center_y, center_x + text_w, center_y + text_h
        elif position == Position.BOTTOM_RIGHT:
            return center_x - text_w, center_y, center_x, center_y + text_h
        elif position == Position.BOTTOM_CENTER:
            return (
                center_x - text_w // 2,
                center_y,
                center_x + text_w // 2,
                center_y + text_h,
            )

    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        labels: List[str] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided detections.

        Args:
            scene (np.ndarray): The image where labels will be drawn.
            detections (Detections): Object detections to annotate.
            labels (List[str]): Optional. Custom labels for each detection.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.

        Returns:
            The annotated image.

        Example:
            ```python
            >>> import supervision as sv

            >>> image = ...
            >>> detections = sv.Detections(...)

            >>> label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = label_annotator.annotate(
            ...     scene=image.copy(),
            ...     detections=detections
            ... )
            ```

        ![label-annotator-example](https://media.roboflow.com/
        supervision-annotator-examples/label-annotator-example-purple.png)
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        anchors_coordinates = detections.get_anchors_coordinates(
            anchor=self.text_anchor
        ).astype(int)
        #num_anchors = len(anchors_coordinates)
        #centers = [compute_mask_center_of_mass(det[1].squeeze()) for det in detections]
        #use_colors = assign_colors(centers)
        for detection_idx, center_coordinates in enumerate(anchors_coordinates):
            text_color = resolve_color(
                #color=self.color,
                color=self.text_color,
                #color = use_colors[detection_idx],
                detections=detections,
                detection_idx=detection_idx,
                color_lookup=self.color_lookup
                if custom_color_lookup is None
                else custom_color_lookup,
            )

            text = (
                f"{detections.class_id[detection_idx]}"
                if (labels is None or len(detections) != len(labels))
                else labels[detection_idx]
            )
            #text = str(int(text)+1)
            
            text_w, text_h = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]
            text_w_padded = text_w + 2 * self.text_padding
            text_h_padded = text_h + 2 * self.text_padding

            # text_background_xyxy = self.resolve_text_background_xyxy(
            #     center_coordinates=tuple(center_coordinates),
            #     text_wh=(text_w_padded, text_h_padded),
            #     position=self.text_anchor,
            # )
            
            _mask = detections[detection_idx].mask.squeeze()
            center_coordinates_dist = self.resolve_text_background_xyxy_dist(
                _mask)

            text_background_xyxy = self.resolve_text_background_xyxy(
                center_coordinates=center_coordinates_dist,
                text_wh=(text_w_padded, text_h_padded),
                position=self.text_anchor,
            )

            text_x = text_background_xyxy[0] + self.text_padding
            text_y = text_background_xyxy[1] + self.text_padding + text_h

            #rect_color = Color.black() if np.mean(color.as_rgb()) > 127 else Color.white() 
            #rect_color = background_color(color.as_rgb())
            # rect_color = Color.black()
            cv2.rectangle(
                img=scene,
                pt1=(text_background_xyxy[0], text_background_xyxy[1]),
                pt2=(text_background_xyxy[2], text_background_xyxy[3]),
                color = self.color.as_rgb(),
                thickness=cv2.FILLED,
            )

            #text = string.ascii_lowercase[int(text)]
            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=text_color.as_bgr(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene



class MarkVisualizer:
    """
    A class for visualizing different marks including bounding boxes, masks, polygons,
    and labels.

    Parameters:
        line_thickness (int): The thickness of the lines for boxes and polygons.
        mask_opacity (float): The opacity level for masks.
        text_scale (float): The scale of the text for labels.
    """
    def __init__(
        self,
        with_box: bool = False,
        with_mask: bool = False,
        with_polygon: bool = False,
        with_label: bool = True,
        line_thickness: int = 2,
        mask_opacity: float = 0.05,
        text_scale: float = 0.6
    ) -> None:
        self.with_box = with_box
        self.with_mask = with_mask
        self.with_label = with_label
        self.with_polygon = with_polygon
        self.box_annotator = sv.BoundingBoxAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.mask_annotator = sv.MaskAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            opacity=mask_opacity)
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=line_thickness)
        self.label_annotator = sv.LabelAnnotator(
            color=sv.Color.black(),
            text_color=sv.Color.white(),
            color_lookup=sv.ColorLookup.INDEX,
            text_position=sv.Position.CENTER_OF_MASS,
            text_scale=text_scale)

    def visualize(
        self,
        image: np.ndarray,
        marks: sv.Detections,
        with_box: Optional[bool] = None,
        with_mask: Optional[bool] = None,
        with_polygon: Optional[bool] = None,
        with_label: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Visualizes annotations on an image.

        This method takes an image and an instance of sv.Detections, and overlays
        the specified types of marks (boxes, masks, polygons, labels) on the image.

        Parameters:
            image (np.ndarray): The image on which to overlay annotations.
            marks (sv.Detections): The detection results containing the annotations.
            with_box (bool): Whether to draw bounding boxes. Defaults to False.
            with_mask (bool): Whether to overlay masks. Defaults to False.
            with_polygon (bool): Whether to draw polygons. Defaults to True.
            with_label (bool): Whether to add labels. Defaults to True.

        Returns:
            np.ndarray: The annotated image.
        """
        with_box = with_box or self.with_box
        with_mask = with_mask or self.with_mask
        with_polygon = with_box or self.with_polygon
        with_label = with_box or self.with_label
        
        annotated_image = image.copy()
        if with_box:
            annotated_image = self.box_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=marks)
        if with_label:
            #labels = list(map(str, range(len(marks))))
            annotated_image = self.label_annotator.annotate(
                scene=annotated_image, detections=marks)
        return annotated_image



def load_mark_visualizer(cfg):
    if isinstance(cfg, str):
        # load config from file
        cfg = load_config(cfg)
    vis = MarkVisualizer(
        with_label = cfg.label.text_include,
        with_mask = cfg.mask.mask_include,
        with_polygon = cfg.polygon.polygon_include,
        with_box = cfg.box.box_include
    )
    # label markers
    if cfg.label.text_include:
        vis.label_annotator = LabelAnnotator(
            text_color = MyColorPalette.default(),
            color = sv.Color.black(), # background rectangle
            color_lookup = sv.ColorLookup.INDEX,
            text_position = getattr(sv.Position, cfg.label.text_position),
            text_scale = cfg.label.text_scale,
            text_thickness = cfg.label.text_thickness,
            text_padding = cfg.label.text_padding
        )
    # box markers
    if cfg.box.box_include:
        vis.box_annotator = sv.annotators.core.BoundingBoxAnnotator(
            color = MyColorPalette.default(),
            thickness = cfg.box.thickness,
            color_lookup = sv.ColorLookup.INDEX,
        )
    # mask markers
    if cfg.mask.mask_include:
        vis.mask_annotator = sv.annotators.core.MaskAnnotator(
            color = MyColorPalette.default(),
            opacity = cfg.mask.mask_opacity,
            color_lookup = sv.ColorLookup.INDEX
        )
    # polygon markers
    if cfg.polygon.polygon_include:
        vis.polygon_annotator = sv.annotators.core.PolygonAnnotator(
            color = MyColorPalette.default(),
            thickness = cfg.polygon.polygon_thickness,
            color_lookup = sv.ColorLookup.INDEX
        )
    return vis


def load_grasp_visualizer(cfg):
    if isinstance(cfg, str):
        cfg = load_config(cfg)
    vis = GraspVisualizer(    
        as_line = cfg.as_line,
        grasp_colors = [getattr(Color, c)() for c in cfg.grasp_colors.split(',')],
        line_thickness = cfg.line_thickness,
        text_rect_color = getattr(Color, cfg.label.text_rect_color)(),
        text_color = getattr(Color, cfg.label.text_color)(),
        text_scale = cfg.label.text_scale,
        text_thickness = cfg.label.text_thickness,
        text_padding = cfg.label.text_padding,
        text_position = cfg.label.text_position,
        with_mask = cfg.mask.mask_include,
        with_polygon = cfg.polygon.polygon_include,
        with_box = cfg.box.box_include,
        with_label = cfg.label.label_include,
        with_gray = cfg.with_gray
    )
    # box marker
    if cfg.box.box_include:
        vis.box_annotator = sv.annotators.core.BoundingBoxAnnotator(
            color = getattr(Color, cfg.box.box_color)(),
            thickness = cfg.box.thickness,
            color_lookup = sv.ColorLookup.INDEX,
        )
    # mask marker
    if cfg.mask.mask_include:
        vis.mask_annotator = sv.annotators.core.MaskAnnotator(
            color = getattr(Color, cfg.mask.mask_color)(),
            opacity = cfg.mask.mask_opacity,
            color_lookup = sv.ColorLookup.INDEX,
        )
    # polygon marker
    if cfg.polygon.polygon_include:
        vis.polygon_annotator = sv.annotators.core.PolygonAnnotator(
            color = getattr(Color, cfg.polygon.polygon_color)(),
            thickness = cfg.polygon.polygon_thickness,
            color_lookup = sv.ColorLookup.INDEX,
        )
    return vis