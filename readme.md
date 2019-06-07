ImagNet-Vid-Robust playground
=============================

Read our paper [A systematic framework for natural perturbations from videos](https://modestyachts.github.io/natural-perturbations-website/results.html) for background information.

Download the dataset first:
```
./download_dataset.sh
```

Compute Robust Accuracy + Visualize failure cases by scoring predictions (example predictions provided):

```
python score_predictions.py  example_predictions/resnet152_finetune.json
```

Failure examples stored in local directory:
```
ls resnet152_finetune_predictions/
```

The generated folder contains both the benign (anchor) frames and the nearby misclassified frames:

