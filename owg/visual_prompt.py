import os
import re
import pickle
import ast
import copy
import json
import numpy as np
import open3d as o3d
from PIL import Image
from typing import List, Union, Dict, Any, Optional, Tuple

from owg.gpt_utils import request_gpt
from owg.utils.config import load_config
from owg.utils.grasp import Grasp2D, grasp_to_mat
from owg.utils.image import (
    compute_mask_bounding_box,
    crop_square_box,
    create_subplot_image,
    mask2box,
)
from owg.markers.postprocessing import (
    masks_to_marks,
    refine_marks,
    extract_relevant_masks,
)
from owg.utils.pointcloud import to_o3d, create_robotiq_mesh, render_o3d_image
from owg.markers.visualizer import load_mark_visualizer, load_grasp_visualizer

#o3d.visualization.rendering.OffscreenRenderer.enable_headless(True)


class VisualPrompter:

    def __init__(
        self,
        prompt_root_dir: str,
        system_prompt_name: str,
        config: Dict[str, Any],
        prompt_template: str,
        inctx_examples_name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """
        Base class for sending visual prompts to GPT.
        Initializes the VisualPrompter with a path to the system prompt file,
        a configuration dictionary for the GPT request, and a prompt template.

        Args:
            prompt_root_dir (str): Path to the directory containing hte prompts.
            system_prompt_name (str): Name of the .txt file containing the system prompt.
            config (Dict[str, Any]): A dictionary containing the arguments for the GPT request
                                     except for 'images', 'prompt', and 'system_prompt'.
            prompt_template (str): An f-string template for constructing the user prompt.
            inctx_examples_name (Optional[str]): Path to a pickle binary file containing in-context examples.
                                        Defaults to None (zero-shot).
            debug (bool): Whether to print GPT responses.
        """
        self.prompt_root_dir = prompt_root_dir
        self.system_prompt_path = os.path.join(prompt_root_dir,
                                               system_prompt_name)
        self.request_config = config
        self.prompt_template = prompt_template
        self.system_prompt = self._load_text_prompt(self.system_prompt_path)
        self.debug = debug

        self.do_inctx = False
        if inctx_examples_name is not None:
            self.do_inctx = True
            self.inctx_examples = pickle.load(
                open(os.path.join(self.prompt_root_dir, inctx_examples_name),
                     "rb"))

    @staticmethod
    def _load_text_prompt(prompt_path) -> str:
        """
        Reads the text prompt from a specified .txt file.

        Returns:
            str: The content of the text prompt file.
        """
        try:
            with open(prompt_path, "r") as file:
                text_prompt = file.read().strip()
            return text_prompt
        except FileNotFoundError:
            raise ValueError(f"Text prompt file not found: {prompt_path}")

    def prepare_image_prompt(self, image: Union[Image.Image, np.ndarray, str],
                             data: Dict[str, Any]) -> Any:
        """
        Placeholder method for preparing the image inputs.
        This will be implemented in subclasses.

        Args:
            image (Union[Image.Image, np.ndarray, str]):
                Image (PIL, numpy or path string) to construct the visual prompt from.
            data (Dict[str, Any]): Additional data that are usefull for `prepare_image_prompt` method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def parse_response(self, response: str, data: Dict[str, Any]) -> Any:
        """
        Placeholder method for parsing the response from GPT.
        This will be implemented in subclasses.

        Args:
            response (str): The response from GPT.
            data (Dict[str, Any]): Additional data that are usefull for `prepare_image_prompt` method.

        Returns:
            Any: Parsed response data (to be defined by subclasses).
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def request(
        self,
        image: Union[Image.Image, np.ndarray, str],
        data: Dict[str, Any],
        text_query: Optional[str] = None,
    ) -> Dict[int, Any]:
        """
        Sends the constructed prompt to GPT via the OpenAI API.

        Args:
            image (Union[Image.Image, np.ndarray, str]):
                Image (PIL, numpy or path string) to construct the visual prompt from.
            text_query (Optional[str]): The text query that will be inserted into the prompt template.
            data (Dict[str, Any]): Additional data that are usefull for `prepare_image_prompt` method.

        Returns:
            Any: The parsed response from GPT (to be processed by subclasses).
        """
        # Construct the prompt using the provided template and user input
        if text_query is not None:
            text_prompt = self.prompt_template.format(user_input=text_query)
        else:
            text_prompt = self.prompt_template  # no text query

        # Prepare images based on markers
        image_prompt, image_prompt_utils = self.prepare_image_prompt(
            image, data)

        # Extract relevant settings from the config dictionary
        temperature: float = self.request_config.get("temperature", 0.0)
        max_tokens: int = self.request_config.get("n_tokens", 256)
        n: int = self.request_config.get("n", 1)
        model_name: str = self.request_config.get("model_name", "gpt-4o")

        # Call the request_gpt function to get the response
        response: str = request_gpt(
            images=image_prompt,
            prompt=text_prompt,
            system_prompt=self.system_prompt,
            temp=temperature,
            n_tokens=max_tokens,
            n=n,
            in_context_examples=self.inctx_examples if self.do_inctx else None,
            model_name=model_name,
        )
        if self.debug:
            print("\033[92mGPT response:\033[0m")
            print("\033[92m" + response.strip() + "\033[0m")
            print()

        # Parse and return the response (this will be subclassed to define behavior)
        return self.parse_response(response, image_prompt_utils)


class VisualPrompterGrounding(VisualPrompter):

    def __init__(self, config_path: str, debug: bool = False) -> None:
        """
        Initializes the VisualPrompterGrounding class with a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Load config from YAML file
        cfg = load_config(config_path)
        self.image_size = (cfg.image_size_h, cfg.image_size_w)
        self.image_crop = cfg.image_crop
        self.cfg = cfg.grounding
        self.use_subplot_prompt = self.cfg.use_subplot_prompt

        # Extract config related to VisualPrompter and initialize superclass
        config_for_prompter = self.cfg.request
        config_for_visualizer = self.cfg.visualizer

        # Initialize superclass
        super().__init__(
            prompt_root_dir=cfg.prompt_root_dir,
            system_prompt_name=self.cfg.prompt_name,
            config=config_for_prompter,
            prompt_template=self.cfg.prompt_template,
            inctx_examples_name=self.cfg.inctx_prompt_name
            if self.cfg.do_inctx else None,
            debug=debug,
        )

        # Create visualizer using the visualizer config in YAML
        self.visualizer = load_mark_visualizer(config_for_visualizer)

    def prepare_image_prompt(
        self, image: Union[Image.Image, np.ndarray],
        data: Dict[str,
                   np.ndarray]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """
        Prepares the image prompt by resizing and overlaying segmentation masks.

        Args:
            image (Union[Image.Image, np.ndarray]): The input image (as a PIL image or numpy array).
            data (Dict[str, np.ndarray]): 
                Contains `masks`, boolean array of size (N, H, W) for N instance segmentation masks.
                (Optional) Contains `labels`, list of label IDs to name the markers.
        Returns:
            List[Union[Image.Image, np.ndarray]]: The processed image or a list containing both the raw and marked images if configured.
            Dict[str, Any]: The detection markers, potentially refined
        """
        masks = data["masks"]
        labels = data['labels'] if ('labels' in data.keys()
                                    and data['labels'] is not None) else list(
                                        range(1,
                                              len(masks) + 1))

        image_size_h = self.image_size[0]
        image_size_w = self.image_size[1]
        image_crop = self.image_crop
        include_raw_image = self.cfg.include_raw_image
        use_subplot_prompt = self.use_subplot_prompt

        # Resize image and masks if sizes differ
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
            image = np.array(image_pil)

        if image_pil.size != (image_size_w, image_size_h):
            image_pil = image_pil.resize((image_size_w, image_size_h),
                                         Image.Resampling.LANCZOS)
            masks = np.array([
                np.array(
                    Image.fromarray(mask).resize((image_size_w, image_size_h),
                                                 Image.LANCZOS)).astype(bool)
                for mask in masks
            ])
            image = np.array(image_pil)

        if image_crop:
            image = image[image_crop[0]:image_crop[2],
                          image_crop[1]:image_crop[3]].copy()
            masks = np.stack([
                m[image_crop[0]:image_crop[2],
                  image_crop[1]:image_crop[3]].copy() for m in masks
            ])

        # Process markers from masks
        markers = masks_to_marks(masks, labels=labels)

        # Optionally refine markers
        if self.cfg.do_refine_marks:
            refine_kwargs = self.cfg.refine_marks
            markers = refine_marks(markers, **refine_kwargs)

        if use_subplot_prompt:
            # Use separate legend image
            assert (
                include_raw_image is True
            ), "`use_subplot_prompt` should be set to True together with `include_raw_image`"
            # Masked cropped object images
            boxes = [mask2box(mask) for mask in masks]
            crops = []
            for mask, box in zip(masks, boxes):
                masked_image = image.copy()
                masked_image[mask == False] = 127
                crop = masked_image[box[1]:box[3], box[0]:box[2]]
                crops.append(crop)
            subplot_size = self.cfg.subplot_size
            marked_image = create_subplot_image(crops,
                                                h=subplot_size,
                                                w=subplot_size)

        else:
            # Use the visualizer to overlay the markers on the image
            marked_image = self.visualizer.visualize(
                image=np.array(image).copy(), marks=markers)

        # Prepare the image prompt
        img_prompt = [marked_image]
        if include_raw_image:
            img_prompt = [image.copy(), marked_image]
        output_data = {
            "markers": markers,
            "raw_image": image.copy(),
            'labels': labels,
        }

        return img_prompt, output_data

    def parse_response(self, response: str, data: Dict[str,
                                                       Any]) -> Dict[int, Any]:
        """
        Parses the GPT response to extract relevant mask IDs and returns corresponding markers.

        Args:
            response (str): The raw response from GPT, which contains the IDs of the objects identified.
            data (Dict[int, Any]): Contains `markers`, a dictionary where keys are mask IDs and values are corresponding mask data.
                                   (Optional) Contains `labels`, list of label IDs to name the markers.

        Returns:
            Dict[int, Any]: A dictionary of selected markers based on GPT's response.
        """
        markers = data["markers"]
        labels = list(data['labels'])
        try:
            # Extract the portion of the response that contains the final answer IDs
            output_IDs_str = (response.split("final answer is:")[1].replace(
                ".", "").strip())
            output_IDs = eval(output_IDs_str)  # Convert string to list of IDs

            # Convert to `labels` indexing
            output_IDs_ret = [labels.index(x) for x in output_IDs]

            # Return the masks corresponding to the extracted IDs
            outputs = {mark: markers[mark] for mark in output_IDs_ret}
            output_mask = np.zeros_like(markers[0].mask.squeeze(0))
            for _, mark in outputs.items():
                output_mask[mark.mask.squeeze(0) == True] = True

            return outputs, output_mask, output_IDs

        except Exception as e:
            print(f"Failed parsing response: {e}")
            return {}


class VisualPrompterPlanning(VisualPrompterGrounding):

    def __init__(self, config_path: str, debug: bool = False) -> None:
        """
        Inherits from VisualPromptGrounding with a separate YAML configuration file.
        The two subclasses use same visual prompting but differ in text prompt and response format.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Initialize superclass
        cfg = load_config(config_path)
        self.image_size = (cfg.image_size_h, cfg.image_size_w)
        self.image_crop = cfg.image_crop
        self.cfg = cfg.planning
        self.use_subplot_prompt = self.cfg.use_subplot_prompt

        # Extract config related to VisualPrompter and initialize superclass
        config_for_prompter = self.cfg.request
        config_for_visualizer = self.cfg.visualizer

        # Initialize superclass
        VisualPrompter.__init__(
            self,
            prompt_root_dir=cfg.prompt_root_dir,
            system_prompt_name=self.cfg.prompt_name,
            config=config_for_prompter,
            prompt_template=self.cfg.prompt_template,
            inctx_examples_name=self.cfg.inctx_prompt_name
            if self.cfg.do_inctx else None,
            debug=debug,
        )

        # Create visualizer using the visualizer config in YAML
        self.visualizer = load_mark_visualizer(config_for_visualizer)

        # Appropriate response format parsing
        self.parse_response = (self.parse_response_json
                               if self.cfg.response_format == "json" else
                               self.parse_response_text)

    def parse_response_text(self, response: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def parse_response_json(self, response: str,
                            data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parses a response string into a list of Python dictionaries.

        Args:
        response: The response string to parse.

        Returns:
        A list of Python dictionaries containing the parsed actions and inputs.
        """
        match = re.search(r"Plan:\s*```json(.*?)```", response, re.DOTALL)
        if match:
            # Replace single quotes with double quotes for valid JSON
            json_str = match.group(1).strip().replace("'", '"')
            return json.loads(json_str)
        return None


class VisualPrompterGraspRanking(VisualPrompter):

    def __init__(self, config_path: str, debug: bool = False) -> None:
        """
        Initializes the RequestGraspRanking class with a YAML configuration file.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        # Load config from YAML file
        cfg = load_config(config_path)
        self.image_size = (cfg.image_size_h, cfg.image_size_w)
        self.cfg = cfg.grasping
        self.crop_size = self.cfg.crop_square_size
        self.use_subplot_prompt = self.cfg.use_subplot_prompt
        self.use_3d_prompt = self.cfg.use_3d_prompt

        # Extract config related to VisualPrompter and initialize superclass
        prompt_path = os.path.join(cfg.prompt_root_dir, self.cfg.prompt_name)
        config_for_prompter = self.cfg.request
        config_for_visualizer = self.cfg.visualizer

        # Initialize superclass
        super().__init__(
            prompt_root_dir=cfg.prompt_root_dir,
            system_prompt_name=self.cfg.prompt_name,
            config=config_for_prompter,
            prompt_template=self.cfg.prompt_template,
            inctx_examples_name=self.cfg.inctx_prompt_name
            if self.cfg.do_inctx else None,
            debug=debug,
        )

        # Create visualizer using the visualizer config in YAML
        if self.use_3d_prompt:
            self.prepare_image_prompt = self.prepare_image_prompt_3d
            self.gripper_mesh = create_robotiq_mesh(self.cfg.gripper_mesh_path)
        else:
            self.prepare_image_prompt = self.prepare_image_prompt_2d
            self.visualizer = load_grasp_visualizer(config_for_visualizer)

    def prepare_image_prompt_2d(
        self,
        image: Union[Image.Image, np.ndarray],
        data: Dict[str, Any],
    ) -> np.ndarray:
        grasps = data["grasps"]
        mask = data["mask"]

        image_size_h = self.image_size[0]
        image_size_w = self.image_size[1]

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image
            image = np.array(image_pil)

        # crop region of interest
        x, y, w, h = compute_mask_bounding_box(mask)
        crop_size = max(max(w, h), self.crop_size)
        image_roi, bbox = crop_square_box(image.copy(), int(x + w // 2),
                                          int(y + h // 2), crop_size)
        x1, y1, x2, y2 = bbox
        mask_roi = mask[y1:y2, x1:x2]

        # rescale grasp coordinates to cropped image frame
        grasps_res = [g.rescale_to_crop(bbox) for g in grasps]
        grasp_markers = {k: g for k, g in enumerate(grasps_res)}

        if self.use_subplot_prompt:
            per_grasp_images = [
                self.visualizer.visualize(
                    image=image_roi.copy(),
                    grasps=[g],
                    mask=mask_roi,
                    labels=[1 + j],
                ) for j, g in enumerate(grasps_res)
            ]
            subplot_size = self.cfg.subplot_size
            marked_image = create_subplot_image(per_grasp_images,
                                                h=subplot_size,
                                                w=subplot_size)
            marked_image = np.array(marked_image)

        else:
            marked_image = self.visualizer.visualize(image=image_roi.copy(),
                                                     grasps=grasps_res,
                                                     mask=mask_roi)

        output_data = {
            "grasp_markers": grasp_markers,
            "image_roi": image_roi,
            "mask_roi": mask_roi,
            "bbox": bbox,
        }

        return [marked_image], output_data

    def prepare_image_prompt_3d(
        self,
        pointcloud: o3d.geometry.PointCloud,
        data: Dict[str, Any],
    ) -> np.ndarray:
        grasps = data["grasps"]
        grasp_markers = {k: g for k, g in enumerate(grasps)}

        grasp_poses = [grasp_to_mat(g) for g in grasps]
        grasp_meshes = [
            copy.deepcopy(self.gripper_mesh).transform(p) for p in grasp_poses
        ]

        # def render_with_clean_context(*args, **kwargs):
        #     # Temporarily disable PyBullet rendering
        #     p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        #     # Do Open3D rendering
        #     image = render_o3d_image(*args, **kwargs)

        #     # Re-enable PyBullet rendering
        #     p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        #     p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        #     return image

        lookat = np.array(grasp_poses[0][:3, 3])
        grasp_images = [
            render_o3d_image([pointcloud, gm],
                             lookat=lookat,
                             front=np.array([0, 1, 1]),
                             up=np.array([0, 0, 1]),
                             zoom=0.25)
            for g, gm in zip(grasp_poses, grasp_meshes)
        ]

        subplot_size = self.cfg.subplot_size
        marked_image = create_subplot_image(grasp_images,
                                            h=subplot_size,
                                            w=subplot_size)

        output_data = {
            "grasp_markers": grasp_markers,
        }

        return [marked_image], output_data

    def parse_response(self, response: str, data: Dict[str,
                                                       Any]) -> Dict[int, Any]:
        """
        Parses the GPT response to extract relevant grasp IDs and returns corresponding markers.

        Args:
            response (str): The raw response from GPT, which contains the IDs of the objects identified.
            data (Dict[int, Any]): Contains `grasp_markers`, a list of Grasp2D candidates.

        Returns:
            Dict[int, Any]: A dictionary of selected markers based on GPT's response.
        """
        grasps = data["grasp_markers"]
        try:
            # Extract the portion of the response that contains the final answer IDs
            output_IDs_str = (response.split("final answer is:")[1].replace(
                ".", "").strip())
            output_IDs = eval(output_IDs_str)  # Convert string to list of IDs

            # Convert to 1-indexing
            output_IDs_ret = [x - 1 for x in output_IDs]

            # Return the masks corresponding to the extracted IDs
            sorted_grasps = [grasps[i] for i in output_IDs_ret]
            best_grasp = sorted_grasps[0]

            return sorted_grasps, best_grasp, output_IDs_ret

        except Exception as e:
            print(f"Failed parsing response: {e}")
            return {}
