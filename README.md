# OWG: Towards Open-World Grasping with Large Vision-Language Models
This is the official implementation for the paper "Towards Open-World Grasping with Large Vision-Language Models" (CoRL 2024).

<p align="center"> <img src='media/fdsfad.drawio.png' align="center" > </p>
The ability to grasp objects in-the-wild from open-ended language instructions constitutes a fundamental challenge in robotics. An open-world grasping system should be able to combine high-level contextual with low-level physical-geometric reasoning in order to be applicable in arbitrary scenarios. Recent works exploit the web-scale knowledge inherent in large language models (LLMs) to plan and reason in robotic context, but rely on external vision and action models to ground such knowledge into the environment and parameterize actuation. This setup suffers from two major bottlenecks: a) the LLM’s reasoning capacity is constrained by the quality of visual grounding, and b) LLMs do not contain low-level spatial understanding of the world, which is essential for grasping in contact-rich scenarios. In this work we demonstrate that modern vision-language models (VLMs) are capable of tackling such limitations, as they are implicitly grounded and can jointly reason about semantics and geometry. We propose OWG, an open-world grasping pipeline that combines VLMs with segmentation and grasp synthesis models to unlock grounded world understanding in three stages: open-ended referring segmentation, grounded grasp planning and grasp ranking via contact reasoning, all of which can be applied zero-shot via suitable visual prompting mechanisms. We conduct extensive evaluation in
cluttered indoor scene datasets to showcase OWG’s robustness in grounding from open-ended language, as well as open-world robotic grasping experiments in both simulation and hardware that demonstrate superior performance compared to previous supervised and zero-shot LLM-based methods. 


[[project page]](https://gtziafas.github.io/OWG_project/) | [[arxiv]](https://arxiv.org/abs/2406.18722v2) | [[bibtex]](#citation)

## Release

- >[Coming Next]  Stay tuned for: Improved referring segmentation, Multi-cam grounding, 6-DoF Grasp Synthesis Model Integration, Task-Oriented Queries
- [2025/03/17] 🔥 Release a Pybullet environment that integrates OWG for online open-world grasping demos.
- [2024/11/04] 🔥 Release the source code and prompts for implementing all OWG components, as well as visualizations / evaluations in the OCID-VLG sub-set.

## Installation
The code has been tested with `python3.9` with `torch` version 2.0 and CUDA driver 11.8. Create a virtual environment and install `torch` for your own CUDA driver from [here](https://pytorch.org/get-started/locally/). 

Install local dependencies with 
```
pip install -r requirements.txt
```

You will have to download the pretrained Gr-ConvNet model from the original repo, (e.g. [here](https://github.com/skumra/robotic-grasping/tree/master/trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch32) for RGB-D model pretrained in Cornell). Create a folder `third_party/grconvnet/checkpoints` in the repo's root directory and place it there.

Before you run OWG, remember to set the following environment variables for your OpenAI access token:

```
export OPENAI_API_KEY=your_openai_key
```


## Open-Ended Grounding in OCID
See [this example notebook](https://github.com/gtziafas/OWG/blob/main/notebooks/ocid_grounding.ipynb) for instructions on how to run inference and dataset evaluation in OCID-VLG scenes with the OWG grounder.


## Pybullet Environment 
We release a Pybullet-based environment for testing the OWG pipeline in grasping [YCB](https://www.ycbbenchmarks.com/) objects from open-ended language. Check [this notebook](https://github.com/gtziafas/OWG/blob/main/notebooks/setup_pyb_env_and_primitives.ipynb) for more context in setting up the robot environment and motion primitives, and [this notebook](https://github.com/gtziafas/OWG/blob/main/notebooks/pyb_owg_policy.ipynb) for a step-by-step example of how to use OWG to control the robot in Pybullet. We develop a version of OWG as a closed-loop grasping policy and integrate everything together in `owg.ui`, which can be configured from `config/pyb/env.yaml`, while the OWG Policy settings can be configured from `config/pyb/OWG.yaml`. 

To run a demo language-grasping trial, simply:
```
python demo.py --n_objects=10 --seed=42 --verbose=1 --vis=1
```
use the `vis` flag to visualize intermediate visual prompts to the VLM and `verbose` to print the VLM response in console. 

## Acknowledgements
Our project is made possible due to following works:
1. [GR-ConvNet](https://github.com/skumra/robotic-grasping) implementation
2. Visualizer utilities from robotflow's [supervision](https://github.com/roboflow/supervision) and [maestro](https://github.com/roboflow/maestro).
3. Set-of-mark prompting from [SoM](https://github.com/microsoft/SoM).

## Citation
If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.
```
@article{tziafas2024openworldgraspinglargevisionlanguage,
      title={Towards Open-World Grasping with Large Vision-Language Models}, 
      author={Georgios Tziafas and Hamidreza Kasaei},
      year={2024},
      journal={8th Conference on Robot Learning (CoRL 2024)}
```
