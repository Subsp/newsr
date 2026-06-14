import json
import numpy as np
import constants as C
import subprocess

scenes = C.SCENES_TNT

output_dirs = [
    "output/SOF_STD"
]
KEYS = ["precision", "recall", "fscore"]

print(output_dirs)

def show_results(o):
    all_metrics = {"precision": [], "recall": [], "fscore": []}
    for scene in scenes:

        json_file = f"{o}/{scene}/eval/results.json"
        import os
        if not os.path.exists(json_file):
            [all_metrics[k].append(0.0) for k in KEYS]
            continue
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
            
        
        print(f'{C.RED}{z}:{C.RESET} ' + " & ".join([str(s) for s in scenes]))
        print(" & ".join(latex))

for o in output_dirs:
    print('')
    show_results(o) 