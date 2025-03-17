from typing import List, Dict, Any, Optional
from owg.visual_prompt import VisualPrompterPlanning, VisualPrompterGrounding, VisualPrompterGraspRanking
from owg.utils.image import display_image
from owg.utils.grasp import Grasp2D
from owg.utils.pointcloud import to_o3d
import numpy as np
from PIL import Image


class OwgPolicy:
    """
    Implement OWG algorithm to predict next gripper action given user query and observation
    """

    def __init__(self,
                 config_path: str,
                 verbose: bool = False,
                 vis: bool = False,
                 use_grasp_ranker: bool = True):
        self.vis = vis
        self.grounder = VisualPrompterGrounding(config_path, debug=verbose)
        self.planner = VisualPrompterPlanning(config_path, debug=verbose)
        if use_grasp_ranker:
            self.grasp_ranker = VisualPrompterGraspRanking(config_path,
                                                           debug=verbose)
        self.use_grasp_ranker = use_grasp_ranker

    def predict(self,
                obs: Dict[str, Any],
                user_query: str,
                grasps: Optional[List[tuple]] = None):
        image, seg = obs['image'], obs['seg']
        # per object mask and Grasp2D objects
        obj_ids = np.unique(seg)[1:]
        all_masks = np.stack([seg == objID for objID in obj_ids])
        marker_data = {'masks': all_masks, 'labels': obj_ids}

        # grounding
        if self.vis:
            # Visualize grounding visual prompt
            visual_promppt, _ = self.grounder.prepare_image_prompt(
                image.copy(), marker_data)
            marked_image_grounding = visual_promppt[-1]
            Image.fromarray(marked_image_grounding).show()

        dets, target_mask, target_ids = self.grounder.request(text_query=user_query,
                                                    image=image.copy(),
                                                    data=marker_data)
        try:
            target_mask_index = list(dets.keys())[0]
            target_id = target_ids[0]  # assume single correct object
        except IndexError:
            print(f'Object not found: {user_query}')
            return {'action': 'fail'}

        # planning
        plan = self.planner.request(text_query=target_id,
                                    image=image.copy(),
                                    data=marker_data)
        action = plan[0]
        action['target_id'] = target_id
        action['grasps'] = list(range(len(grasps[action['input']])))

        try:
            obj_grasps = grasps[action['input']]
            if action['action'] == 'pick':
                obj_mask = target_mask
            else:
                obj_mask = all_masks[obj_ids.tolist().index(action['input'])]
        except IndexError:
            print(f'Object {action["input"]} not detected in image.')
            return {'action': 'fail'}

        # grasp ranking
        if self.use_grasp_ranker:

            if not self.grasp_ranker.use_3d_prompt:
                req_data = {'grasps': obj_grasps, 'mask': obj_mask}
                inp_prompt = image.copy()

            else:
                points = obs['points']
                colors = image.reshape(-1, 3) / 255.
                cloud = to_o3d(points, colors)
                req_data = {
                    'grasps': obj_grasps,
                }
                inp_prompt = cloud

            if self.vis:
                # Visualize grounding visual prompt
                visual_prompt, _ = self.grasp_ranker.prepare_image_prompt(
                    inp_prompt, req_data)
                marked_image_grasping = visual_prompt[-1]
                Image.fromarray(marked_image_grasping).show()

            _, _, grasp_indices = self.grasp_ranker.request(
                inp_prompt, req_data)

            action['grasps'] = grasp_indices

        return action
