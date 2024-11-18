import numpy as np


class instruction_generater:
    def __init__(self, seed, template_path, slackness_type) -> None:
        import random

        self.random = random
        self.random.seed(seed)
        self.templates = self.load_templates(template_path, slackness_type)

    def load_templates(self, file_path, slackness_type):
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get(slackness_type, [])

    def fill_in_template(self, element):
        template = self.random.sample(self.templates, k=1)[0]
        if self.has_placeholder(template, "other_init"):
            ref = element["init"]
        else:
            ref = "it"
        for key in element.keys():
            if element[key] is not None:
                element[key] = element[key].format(ref = ref)
        instruction = template.format(**element)
        return instruction
        
    def has_placeholder(self, template: str, placeholder: str) -> bool:
        formatted_placeholder = f"{{{placeholder}}}"
        return formatted_placeholder in template

    def get_absolute_position_pair(self, layout, tgt_mug_idx):
        element_list = []
        x = layout[:, 0]
        y = layout[:, 1]
        x_min_idx = np.argmin(x)
        if x_min_idx == tgt_mug_idx:
            element_list.append(["the left battery", "the rigt battery"])
        x_max_idx = np.argmax(x)
        if x_max_idx == tgt_mug_idx:
            element_list.append(["the right battery", "the left battery"])
        y_min_idx = np.argmin(y)
        if y_min_idx == tgt_mug_idx:
            element_list.append(["the back battery", "the front battery"])
        y_max_idx = np.argmax(y)
        if y_max_idx == tgt_mug_idx:
            element_list.append(["the front battery", "the back battery"])
        if len(element_list) != 0:
            sample_idx = self.random.sample(range(len(element_list)), k=1)[0]
            return element_list[sample_idx]
        else:
            return None

    def get_absolute_position(self, layout, tgt_mug_idx):
        element_list = []
        x = layout[:, 0]
        y = layout[:, 1]
        x_min_idx = np.argmin(x)
        if x_min_idx == tgt_mug_idx:
            element_list.append("the left one")
        x_max_idx = np.argmax(x)
        if x_max_idx == tgt_mug_idx:
            element_list.append("the right one")
        y_min_idx = np.argmin(y)
        if y_min_idx == tgt_mug_idx:
            element_list.append("the back one")
        y_max_idx = np.argmax(y)
        if y_max_idx == tgt_mug_idx:
            element_list.append("the front one")
        if len(element_list) != 0:
            sample_idx = self.random.sample(range(len(element_list)), k=1)[0]
            return element_list[sample_idx]
        else:
            return None

    def get_relative_position(self, layout, reference, tgt_branch_idx):
        dist = np.linalg.norm(layout - reference, ord=2, axis=-1)
        dist_max_idx = np.argmax(dist)
        dist_min_idx = np.argmin(dist)
        if dist_max_idx == tgt_branch_idx:
            return ["the furthest slot relative to it", "the nearest slot relative to it"]
        elif dist_min_idx == tgt_branch_idx:
            return ["the nearest slot relative to it", "the furthest slot relative to it"]
        else:
            return None

    def config_to_descriptive_element(
        self, config, init_list, vis_info, blank_descriptive_element
    ):
        init_idx = config["init"]
        tgt_idx = config["tgt"]

        init_layout = config["init_layout"]
        tgt_layout = config["tgt_layout"]

        assigned_init_position = init_layout[init_idx]


        init_description_list = []
        for index in range(len(init_list)):
            init_name = init_list[index]
            if init_list.count(init_name) > 1:
                index_with_same_vis = [index for index, value in enumerate(init_list) if value == init_name]
                position_with_same_vis = init_layout[index_with_same_vis]
                absolute_position = self.get_absolute_position(position_with_same_vis, index)
                if absolute_position is not None:
                    color_info = vis_info["init"][init_name][0]
                    assigned_init_vis = f"{absolute_position} in the {color_info} batteries"
                else:
                    assigned_init_vis = None
            else:
                assigned_init_vis = vis_info["init"][init_name][0]
            init_description_list.append(assigned_init_vis)

        assiged_element = init_description_list[0]
        other_init_list = [element for element in init_description_list[1:] if element is not None]

        if len(other_init_list) != 0:
            assigned_init_vis = [assiged_element,
                                self.random.choice(other_init_list)]
        else:
            assigned_init_vis = [assiged_element,
                                None]
        
        other_list = [ index for index in list(range(12)) if index != tgt_idx[0]]
        other_slot_index = self.random.sample(other_list, k=1)[0]

        assigned_tgt_name = self.random.sample(vis_info["tgt"][str(tgt_idx[0])], k=1)[0]
        other_tgt_name = self.random.sample(vis_info["tgt"][str(other_slot_index)], k=1)[0]

        # define description
        blank_descriptive_element["init"]["color"] = assigned_init_vis
        blank_descriptive_element["init"]["position"] = self.get_absolute_position_pair(
            init_layout, init_idx[0]
        )
        blank_descriptive_element["tgt"]["position"] = [assigned_tgt_name, other_tgt_name]
        blank_descriptive_element["tgt"]["relative_position"] = (
            self.get_relative_position(
                tgt_layout, assigned_init_position, tgt_idx[0]
            )
        )
        return blank_descriptive_element

    def distill_descriptive_element(self, descriptive_element):
        distill_element = {}
        for key in descriptive_element.keys():
            descriptive = descriptive_element[key]
            element = []
            for sub_key in descriptive.keys():
                if descriptive[sub_key] is not None:
                    element.append(descriptive[sub_key])
            sel_element = self.random.sample(element, k=1)
            distill_element[key] = sel_element[0]

        distill_element["other_init"] = distill_element["init"][-1]
        distill_element["init"] = distill_element["init"][0]
        distill_element["other_tgt"] = distill_element["tgt"][-1]
        distill_element["tgt"] = distill_element["tgt"][0]
        return distill_element

    def element_to_str(self, input_element):
        result = ", ".join([f"{key}: {value}" for key, value in input_element.items()])
        return result

    def from_config_to_string(self, config, mug_list, vis_info):
        blank_descriptive_element = {
            "init": {
                "color": None,
                "position": None,
            },
            "tgt": {
                "position": None,
                "relative_position": None,
            },
        }

        descriptive_element = self.config_to_descriptive_element(
            config=config,
            init_list=mug_list,
            vis_info=vis_info,
            blank_descriptive_element=blank_descriptive_element,
        )

        descriptive_element = self.distill_descriptive_element(descriptive_element)
        return descriptive_element



if __name__ == "__main__":
    mug_list = ["nescafe1", "nescafe2", "nescafe3"]

    config = {
        "init": np.array([0]),
        "init_layout": np.array(
            [
                [-0.07552315, -0.13899174, 0.04233006],
                [0.01711246, -0.00962532, 0.04233007],
                [0.21464348, -0.06992011, 0.04233007],
            ],
            dtype=np.float32,
        ),
        "tgt": np.array([2]),
        "tgt_layout": np.array(
            [
                [0.04604891, 0.15104604, 0.36848236],
                [-0.05060494, 0.15107961, 0.3704836],
                [-0.02316604, 0.18560973, 0.26565792],
            ]
        ),
    }

    slackness_type = "branch_slackness"

    generater = instruction_generater(
        seed=4,
        keys=["mug", "branch"],
        template_path="/home/neo/lyt_dp/diffusion/sapien_env/sapien_env/teleop/instruction_generate/template.json",
        slackness_type=slackness_type,
    )
    descriptive_element = generater.from_config_to_string(config, mug_list)
    instruction = generater.fill_in_template(descriptive_element)
    print(instruction)
