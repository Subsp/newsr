# training script NVS datasets

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_NVS
factors = C.FACTORS_NVS
TRAIN_DATA = f'{C.DATA_DIR}/mip360'

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS}'
OUT_DIR = 'output'

DRY_RUN = False

configs = {
    "SOF_STD": "--splatting_config configs/hierarchical.json --lambda_smoothness 0.0 --lambda_opacity_field 0.0 --lambda_extent 0.0 --lambda_distortion 100",
}

# jobs as a cross product of scenes and configs
jobs = [
    (scenes[idx], factors[idx], f'{OUT_DIR}/{config_name}', config_args) 
    for idx,_ in enumerate(scenes) 
    for config_name, config_args in configs.items()
]

def train_scene(gpu, scene, factor, out_dir, args):
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python train.py -s {TRAIN_DATA}/{scene} \
            --eval -i images_{factor} \
            -m {out_dir}/{scene}  \
            {STD_ARGS} {args} \
            --port {6009+gpu}"
    os.system(cmd)
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python render.py -m {out_dir}/{scene} \
            -s {TRAIN_DATA}/{scene} --init_type sfm \
            --data_device cpu --skip_train"
    os.system(cmd)
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {out_dir}/{scene}"
    os.system(cmd)
    
    # marching tets
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python extract_mesh_tets.py -m {out_dir}/{scene} \
            --iteration {ITERATIONS} \
            --bounding_mode STP --opacity_cutoff_tetra 0.0039"
    # by default, not run
    # os.system(cmd)
    
    return True


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)