# T5 for ScienceWorld
This repository contains a reference implementation of using T5 models for [ScienceWorld](https://www.github.com/allenai/ScienceWorld) environment.

## Quickstart
Clone the repository:
```bash
git clone git@github.com:cognitiveailab/t5-scienceworld.git
cd t5-scienceworld
```

Install Dependencies:
```bash
conda create --name t5-scienceworld python=3.9
conda activate t5-scienceworld
pip install -r requirements.txt
```
You may want to install the pytorch manually if your GPU does not support CUDA 11.


To download our pretrained T5 weights, you need to install gsutil first.
```bash
pip install gsutil
```
Or, go to the [gsutil website](https://cloud.google.com/storage/docs/gsutil_install) for installation instructions. After gsutil is installed, you can download our pretrained T5-11b weights by
```bash
./download_pretrained_model.sh [mode]
```
Replace the \[mode\] with either **bc** or **dt** for Behavior Cloning T5 or Text Decision Transformer T5 model respectively.

Run the T5 agent by
```bash
mkdir logs
python main.py --task_num 0 --env_step_limit 100 --lm_path [lm_model] --simplification_str easy --beams 16 --max_episode_per_file 1000 --mode bc --set test --output_path logs --model_parallelism_size 3
```
Replace the \[lm_path\] with the path to the pretrained T5 model checkpoint. Here is what the rest of the arguments means:
- **task_num:** The ScienceWorld task index (0-29). *See **task list** below*
- **env_step_limit:** the number of maximum steps per episode
- **lm_path:** path to the pretrained T5 model checkpoint
- **simplification_str:** The ScienceWorld simplification string
- **beams:** T5 generation beam size
- **max_episode_per_file:** the maximum number of episodes saved per log file
- **mode:** can be **bc** or **dt**. Set for behavior cloning or decision transformer
- **set:** The data split to run the agent. It can be **test** or **dev**.
- **output_path:** output directory
- **model_parallelism_size** number of GPUs to spread the model on. In our experiments, we used 3 Nvidia A-100 GPUs to run the agent.

## ScienceWorld Task List
```
TASK LIST:
    0: 	                                                 task-1-boil  (30 variations)
    1: 	                        task-1-change-the-state-of-matter-of  (30 variations)
    2: 	                                               task-1-freeze  (30 variations)
    3: 	                                                 task-1-melt  (30 variations)
    4: 	             task-10-measure-melting-point-(known-substance)  (436 variations)
    5: 	           task-10-measure-melting-point-(unknown-substance)  (300 variations)
    6: 	                                     task-10-use-thermometer  (540 variations)
    7: 	                                      task-2-power-component  (20 variations)
    8: 	   task-2-power-component-(renewable-vs-nonrenewable-energy)  (20 variations)
    9: 	                                   task-2a-test-conductivity  (900 variations)
   10: 	             task-2a-test-conductivity-of-unknown-substances  (600 variations)
   11: 	                                          task-3-find-animal  (300 variations)
   12: 	                                    task-3-find-living-thing  (300 variations)
   13: 	                                task-3-find-non-living-thing  (300 variations)
   14: 	                                           task-3-find-plant  (300 variations)
   15: 	                                           task-4-grow-fruit  (126 variations)
   16: 	                                           task-4-grow-plant  (126 variations)
   17: 	                                        task-5-chemistry-mix  (32 variations)
   18: 	                task-5-chemistry-mix-paint-(secondary-color)  (36 variations)
   19: 	                 task-5-chemistry-mix-paint-(tertiary-color)  (36 variations)
   20: 	                             task-6-lifespan-(longest-lived)  (125 variations)
   21: 	         task-6-lifespan-(longest-lived-then-shortest-lived)  (125 variations)
   22: 	                            task-6-lifespan-(shortest-lived)  (125 variations)
   23: 	                               task-7-identify-life-stages-1  (14 variations)
   24: 	                               task-7-identify-life-stages-2  (10 variations)
   25: 	                       task-8-inclined-plane-determine-angle  (168 variations)
   26: 	             task-8-inclined-plane-friction-(named-surfaces)  (1386 variations)
   27: 	           task-8-inclined-plane-friction-(unnamed-surfaces)  (162 variations)
   28: 	                    task-9-mendellian-genetics-(known-plant)  (120 variations)
   29: 	                  task-9-mendellian-genetics-(unknown-plant)  (480 variations)
```

## Citing

If this T5 agent is helpful in your work, please cite the following:

```
@misc{scienceworld2022,
    title={ScienceWorld: Is your Agent Smarter than a 5th Grader?},
    author={Ruoyao Wang and Peter Jansen and Marc-Alexandre C{\^o}t{\'e} and Prithviraj Ammanabrolu},
    year={2022},
    eprint={2203.07540},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2203.07540}
}
```
