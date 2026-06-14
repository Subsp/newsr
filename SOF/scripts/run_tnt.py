# training script for TNT dataset

import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import constants as C
from constants import dispatch_jobs

scenes = C.SCENES_TNT
factors = C.FACTORS_TNT

excluded_gpus = set([])

ITERATIONS = 30000
STD_ARGS = f'--iterations {ITERATIONS} --lambda_distortion 100 --eval --far_plane 100.'
OUT_DIR = 'output'
# gt data is assumed to be in the same directory as train data
TNT_GT_DATA = f'{C.DATA_DIR}/TNT_GOF/'
TNT_TRAIN_DATA = f'{C.DATA_DIR}/TNT_GOF'

DRY_RUN = False

configs = {
    "SOF_STD": "--splatting_config configs/hierarchical.json --use_decoupled_appearance --detach_alpha False",
}

# jobs as a cross product of scenes and configs
jobs = [
    (scenes[idx], factors[idx], f'{OUT_DIR}/{config_name}', config_args) 
    for idx,_ in enumerate(scenes) 
    for config_name, config_args in configs.items()
]

def train_scene(gpu, scene, factor, out_dir, args):
    # optimization
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python train.py -s {TNT_TRAIN_DATA}/{scene} \
            -m {out_dir}/{scene} \
            -r {factor} \
            {STD_ARGS} {args} \
            --port {6009+gpu}"
    os.system(cmd)

    # marching tets
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python extract_mesh_tets.py -m {out_dir}/{scene} \
            --iteration {ITERATIONS} \
            --data_device cpu \
            --bounding_mode STP --opacity_cutoff_tetra 0.0039"
    os.system(cmd)
    
    # evaluate
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} \
            python mesh_utils/eval_TNT.py \
            --dataset-dir {TNT_GT_DATA}/{scene} \
            --ply-path {out_dir}/{scene}/test/ours_{ITERATIONS}/mesh_faster_binary_search_7.ply \
            --traj-path {TNT_GT_DATA}/{scene}/{scene}_traj_path.log \
            --out-dir {out_dir}/{scene}/eval"
    os.system(cmd)
    
    return True


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor, train_scene)