import random 

YCB_CATEGORIES = [
 'scissors', # i want to cut some paper (affordance), silver tool ()
 'gelatin box',
 'cracker box',
 'strawberry',
 'power drill',
 'chips can',
 'foam brick',
 'potted meat can',
 'tennis ball',
 'medium clamp',
 'banana', 
 'pear',
 'master chef can',
 'tomato soup can',
 'mustard bottle',
 'hammer'
]


class YcbObjects:
    def __init__(self, load_path, mod_orn=None, mod_stiffness=None, exclude=None, seed=None):
        self.set_seed(seed)
        self.load_path = load_path
        self.mod_orn = mod_orn
        self.mod_stiffness = mod_stiffness
        with open(load_path + '/obj_list.txt') as f:
            lines = f.readlines()
            self.obj_names = [line.rstrip('\n') for line in lines]
        if exclude is not None:
            for obj_name in exclude:
                self.obj_names.remove(obj_name)

    def set_seed(self, seed):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def shuffle_objects(self):
        random.shuffle(self.obj_names)

    def get_obj_path(self, obj_name):
        return f'{self.load_path}/Ycb{obj_name}/model.urdf'

    def check_mod_orn(self, obj_name):
        if self.mod_orn is not None and obj_name in self.mod_orn:
            return True
        return False

    def check_mod_stiffness(self, obj_name):
        if self.mod_stiffness is not None and obj_name in self.mod_stiffness:
            return True
        return False

    def get_obj_info(self, obj_name):
        return self.get_obj_path(obj_name), self.check_mod_orn(obj_name), self.check_mod_stiffness(obj_name)

    def get_n_first_obj_info(self, n):
        info = []
        for obj_name in self.obj_names[:n]:
            info.append(self.get_obj_info(obj_name))
        return info