import time
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter, HtmlFormatter
import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import *

from owg_robot.env import *
from owg_robot.camera import Camera
from owg_robot.objects import YcbObjects
from owg.policy import OwgPolicy
from owg.utils.config import load_config
from owg.utils.grasp import Grasp2D
from third_party.grconvnet import load_grasp_generator


# GUI stuff
# Function to create a text input dialog using Tkinter
def ask_for_user_input():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    user_input = simpledialog.askstring("Input", "User Input: ")
    root.destroy()
    return user_input


class RobotEnvUI:

    def __init__(self, config: Union[Dict[str, Any], str]):
        self.cfg = load_config(config) if isinstance(config, str) else config
        self.n_objects = self.cfg.n_objects
        self.seed = self.cfg.seed

        # init env
        center_x, center_y, center_z = CAM_X, CAM_Y, CAM_Z
        cam_center = (self.cfg.camera.center_x, self.cfg.camera.center_y,
                      self.cfg.camera.center_z)
        cam_target = (self.cfg.camera.target_x, self.cfg.camera.target_y,
                      self.cfg.camera.target_z)
        self.img_size = (self.cfg.camera.img_size, self.cfg.camera.img_size)
        self.camera = Camera(cam_center, cam_target, self.cfg.camera.znear,
                             self.cfg.camera.zfar, self.img_size,
                             self.cfg.camera.fov)
        self.n_grasp_attempts = self.cfg.n_grasp_attempts
        self.env = Environment(self.camera,
                               vis=True,
                               asset_root='./owg_robot/assets',
                               debug=False,
                               finger_length=self.cfg.finger_length,
                               n_grasp_attempts=self.cfg.n_grasp_attempts)

        # load objects
        self.objects = YcbObjects(
            './owg_robot/assets/ycb_objects',
            mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
            mod_stiffness=['Strawberry'],
            seed=self.seed)
        self.objects.shuffle_objects()
        self.env.dummy_simulation_steps(10)

        # init OWG policy
        self.policy = OwgPolicy(
            self.cfg.policy.config_path,
            verbose=self.cfg.policy.verbose,
            vis=self.cfg.policy.vis,
            use_grasp_ranker=self.cfg.policy.use_grasp_ranker)

        self.grasp_rank_3d = False
        if self.cfg.policy.use_grasp_ranker:
            self.grasp_rank_3d = self.policy.grasp_ranker.use_3d_prompt

        # spawn scene
        obs = self.spawn(self.n_objects)

        # GR-ConvNet grasp generator
        self.grasp_generator = load_grasp_generator(self.env.camera)
        # setup and visualize once
        self.setup_grasps(obs, visualise_grasps=True)

        self.n_action_attempts = self.cfg.n_action_attempts
        self.n_grasp_attempts = self.cfg.n_grasp_attempts

    def spawn(self, n_objects):
        self.n_objects = n_objects
        for obj_name in self.objects.obj_names[:self.n_objects]:
            path, mod_orn, mod_stiffness = self.objects.get_obj_info(obj_name)
            self.env.load_isolated_obj(path, obj_name, mod_orn, mod_stiffness)
            self.env.dummy_simulation_steps(30)
        self.init_obj_state = self.env.get_obj_states()
        obs = self.env.get_obs()
        return obs

    def reset_same(self):
        assert self.init_obj_state is not None, "Have to spawn once to initialize state"
        self.env.reset_robot()
        self.env.set_obj_state(self.init_obj_state)
        self.env.dummy_simulation_steps(10)
        obs = self.update()
        self.init_obj_state = self.env.get_obj_states()
        for _ in range(30):
            self.env.step_simulation()
        return obs

    def reset(self, new=False):
        if new:
            self.env.remove_all_obj()
            for _ in range(30):
                self.env.step_simulation()
            # self.objects = YcbObjects('./owg_robot/assets/ycb_objects',
            #         mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
            #         mod_stiffness=['Strawberry'],
            #         seed=self.seed
            # )
            self.seed += 100
            self.objects.set_seed(self.seed)
            self.objects.shuffle_objects()
            self.env.dummy_simulation_steps(10)
            return self.spawn(self.n_objects)
        return self.reset_same()

    def update(self):
        self.env.dummy_simulation_steps(10)
        self.env.update_obj_states()
        obs = self.env.get_obs()
        self.setup_grasps(obs)
        self.env.dummy_simulation_steps(10)
        return obs

    def setup_grasps(self,
                     obs: Dict[str, Any],
                     visualise_grasps: bool = False):
        """
        Run inference with GR-ConvNet grasp generator on current observation
        """
        rgb, depth, seg = obs['image'], obs['depth'], obs['seg']
        img_size = self.grasp_generator.IMG_WIDTH
        if img_size != self.env.camera.width:
            rgb = cv2.resize(rgb, (img_size, img_size))
            depth = cv2.resize(depth, (img_size, img_size))
        for obj_id in self.env.obj_ids:
            mask = seg == obj_id
            if img_size != self.env.camera.width:
                mask = np.array(
                    Image.fromarray(mask).resize((img_size, img_size),
                                                 Image.LANCZOS))
            grasps, grasp_rects = self.grasp_generator.predict_grasp_from_mask(
                rgb, depth, mask, n_grasps=self.n_grasp_attempts, show_output=False)
            if img_size != self.env.camera.width:
                # normalize to original size
                for j, gr in enumerate(grasp_rects):
                    grasp_rects[j][0] = int(gr[0] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][1] = int(gr[1] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][4] = int(gr[4] / img_size *
                                            self.env.camera.width)
                    grasp_rects[j][3] = int(gr[3] / img_size *
                                            self.env.camera.width)
            grasp_rects = [
                Grasp2D.from_vector(
                    x=g[1],
                    y=g[0],
                    w=g[4],
                    h=g[3],
                    theta=g[2],
                    W=self.env.camera.width,
                    H=self.env.camera.width,
                    normalized=False,
                    line_offset=5,
                ) for g in grasp_rects
            ]
            self.env.set_obj_grasps(obj_id, grasps, grasp_rects)

        if visualise_grasps:
            LID = []
            for obj_id in self.env.obj_ids:
                grasps = self.env.get_obj_grasps(obj_id)
                color = np.random.rand(3).tolist()
                for g in grasps:
                    LID = self.env.draw_predicted_grasp(g,
                                                        color=color,
                                                        lineIDs=LID)

            time.sleep(1)
            self.env.remove_drawing(LID)

    def step(self, action):
        '''
        Wrapper around OWG action predictions and implemented robot primitives.

        Args:
          action: Predicted action by OWG 
            - `action`: Either `remove` to place blocking object in free space, or `pick` to put target in tray.
            - `input`: The object ID of object to manipulate.
        '''
        #for attempt in range(self.cfg.n_action_attempts):
        if action['action'] == 'remove':
            success_grasp, success_target = self.env.put_obj_in_free_space(
                action['input'], grasp_indices=action['grasps'])
        elif action['action'] == 'pick':
            success_grasp, success_target = self.env.put_obj_in_tray(
                action['input'], grasp_indices=action['grasps'])
        for _ in range(30):
            self.env.step_simulation()

        if not success_grasp or not success_target:
            print(f'Action failed...')
            success, done = False, False

        elif action['input'] != action['target_id']:
            # successfull action, but not terminal
            print(f'Done {action["action"]} {action["input"]}')
            success, done = True, False

        else:
            # successfull terminal action
            print(f'Done {action["action"]} {action["input"]}')
            success, done = True, True

        return success, done

    def run(self):
        while True:
            # ask user for command
            self.env.reset_robot()
            user_input = ask_for_user_input()

            # reset same scene
            if user_input == ':reset':
                self.reset(new=False)
                self.env.dummy_simulation_steps(10)
                continue

            # spawn new scene
            elif user_input == ':new':
                self.reset(new=True)
                self.env.dummy_simulation_steps(10)
                continue

            # close demo
            elif user_input == ':exit':
                print('Exitting demo')
                self.env.close()
                break

            # actual query
            else:
                # call OWG Policy to predict next action
                attempt = 0
                while True:
                    self.env.reset_robot()
                    obs = self.update()

                    # save a dict for all predicted grasps for policy grasp ranking
                    if self.grasp_rank_3d:
                        all_grasps = {
                            k: self.env.get_obj_grasps(k)
                            for k in self.env.obj_ids
                        }
                    else:
                        all_grasps = {
                            k: self.env.get_obj_grasp_rects(k)
                            for k in self.env.obj_ids
                        }

                    action = self.policy.predict(obs, user_input, all_grasps)

                    if action['action'] == 'fail':
                        break
                    # convert marker IDs to state IDs
                    success, done = self.step(action)
                    if success and done:
                        # task finished
                        break
                    elif not success:
                        attempt += 1
                        if attempt == self.cfg.n_action_attempts:
                            print(f'Action failed. No more atempts.')
                            break
                        print(f'Action failed. {attempt} attempt. Retrying..')
                        continue
                    else:
                        # plan step finished, move to next
                        attempt = 0
                        self.env.dummy_simulation_steps(30)
                        continue
