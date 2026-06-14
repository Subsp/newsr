import json
import numpy as np
import constants as C
import subprocess

scenes = C.SCENES_DTU

output_dirs = [
    "output/SOF_STD",
]
KEYS = {"mean_d2s", "mean_s2d", "overall"}

print(output_dirs)

def show_results(o):
    all_metrics = {"mean_d2s": [], "mean_s2d": [], "overall": []}
    for scene in scenes:

        json_file = f"{o}/scan{scene}/TSDF/results.json"
        data = json.load(open(json_file))
        
        for k in KEYS:
            all_metrics[k].append(data[k])

    print(f'\t{C.YELLOW}{o}{C.RESET}')
    for z in KEYS:
        latex = []
        for k in KEYS:
            numbers = np.asarray(all_metrics[k]).mean(axis=0).tolist()
            
            numbers = all_metrics[k] + [numbers]
            
            numbers = [f"{x:.3f}" for x in numbers]
            if k == z:
                latex.extend(numbers)
            
        
        print(f'{C.RED}{z}:{C.RESET}: ' + " & ".join([str(s) for s in scenes]))
        print(" & ".join(latex))
        
    # print the number of gaussians as well
    num_gaussians = []
    for scene in scenes:
        # the 3rd line always contains the number of gaussians
        result = subprocess.run(['head', '-n', '3', f'{o}/scan{scene}/point_cloud/iteration_30000/point_cloud.ply'], stdout=subprocess.PIPE, text=True)
        # get 3rd line, remove 'element vertex ' and convert to int
        num_gaussians += [int(result.stdout.split('\n')[-2][15:])]
        
    print(f'{C.RED}primitives{C.RESET}: ' + " & ".join([str(s) for s in scenes]))
    
    # add average
    num_gaussians += [int(np.asarray(num_gaussians).mean())]
    
    formatted = [C.human_format(int(n)) for n in num_gaussians]
    print(" & ".join(formatted))

for o in output_dirs:
    print('')
    show_results(o)