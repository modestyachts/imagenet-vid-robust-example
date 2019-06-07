import argparse
import json
import pathlib
import numpy as np
from shutil import copyfile



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_file")
    parser.add_argument("--dataset_loc", default="./imagenet-vid-robust")
    args = parser.parse_args()
    results = {}
    with open(args.prediction_file, "r") as f:
        preds = json.loads(f.read())

    with open(args.dataset_loc + "/metadata/pmsets.json", "r") as f:
        pmsets = json.loads(f.read())

    with open(args.dataset_loc + "/metadata/labels.json" , "r") as f:
        labels = json.loads(f.read())

    with open(args.dataset_loc + "/misc/imagenet_vid_class_index.json" , "r") as f:
        cls_idx = json.loads(f.read())

    correct_anchor = 0
    correct_pmk = 0
    N = len(pmsets)
    wrong_map = {}
    for anchor, pmset in pmsets.items():
        pmset_correct = 0
        wrongs = []
        for elem in pmset:
            if np.argmax(preds[elem]) in labels[elem]:
                pmset_correct += 1
            else:
                wrongs.append(elem)

        if np.argmax(preds[anchor]) in labels[anchor]:
            correct_anchor  += 1
            pmset_correct += 1
            if len(wrongs) > 0:
                wrong_map[anchor] = wrongs[-1]

        if pmset_correct == len(pmset) + 1:
            correct_pmk += 1

    print(f"Benign Accuracy: {correct_anchor/N}")
    print(f"PM-10 Accuracy: {correct_pmk/N}")
    dataset_path = pathlib.Path(args.dataset_loc)
    model_name = pathlib.Path(args.prediction_file).name.strip(".json")
    img_folder = pathlib.Path(f"{model_name}_predictions/")
    img_folder.mkdir(exist_ok=True)
    for anchor, pmk in wrong_map.items():
        anchor_path = pathlib.Path(anchor)
        vid_name = str(anchor_path.parent.name)
        cls_name0 = cls_idx[str(np.argmax(preds[anchor]))][1]
        copy_path0 = img_folder / pathlib.Path(vid_name + f"_pred_{cls_name0}.jpeg")
        copyfile(str(dataset_path / anchor), copy_path0)
        cls_name1 = cls_idx[str(np.argmax(preds[pmk]))][1]
        copy_path1 = img_folder / pathlib.Path(vid_name + f"_pred_{cls_name1}.jpeg")
        copyfile(str(dataset_path / pmk), copy_path1)





if __name__ == "__main__":
    main()





