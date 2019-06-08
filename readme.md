ImageNet-Vid-Robust playground
=============================

Please see our paper [A systematic framework for natural perturbations from videos](https://modestyachts.github.io/natural-perturbations-website) for background information.

Download the dataset (~1G) first:
```
./download_dataset.sh
```

Compute robust accuracy and visualize failure cases by scoring predictions. We provide example predictions from a ResNet-152 model fine-tuned on ImageNet-Vid in [`example_predictions/resnet152_finetune.json`](./example_predictions/resnet152_finetune.json).

```
python score_predictions.py  example_predictions/resnet152_finetune.json
```

Failure examples stored in local directory:
```
ls resnet152_finetune_predictions/
```

The generated folder contains both the benign (anchor) frames and the nearby misclassified frames:
![screen shot](/screenshot.png)

