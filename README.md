# Code for #2 submission in OpenEDS Semantic Segmentation Challenge 2019

This repository is based on [[Dabnet]](https://github.com/Reagan1311/DABNet)

At the time of the challenge Dabnet was most accurate open-sourced lightweight network for semantic segmentation. It was still too big to satisfy competition requirements of under 250k parameters, so number of channels was reduced. Shrinking of the network meant pretrained weights of the original Dabnet is of no use, so training starts with random weights.

## Result

|                                 | Leaderboard IoU | Model File |
|---------------------------------|------------|------------|
| Cross-Entropy loss              | 0.9504     | -          |
| Dice loss                       | 0.9493     | -          |
| Focal loss                      | 0.9516     | -          |
| Focal and Dice losses combo     | 0.9519     | [GoogleDrive](https://drive.google.com/open?id=1qoYYChJ0paJIRmpbfrJlATsmZndAi7T7)           |

## Reproducing results

Executing **./submission.sh** will run inference provided a folder named "test" with images for testing "Eye_segnet_fd_g1_e40_lr_0.010_max_dice.pth.tar" model file are placed in folder **data**. Adding "--device -1" to the command inside the script would run the script on CPU. Zipped file for submission should be in "data" folder after running inference. After placing "train" and "validation" folders inside "data" allows to train the model. Both folders should have "labels" and "images" inside them as in original dataset.

## Unexplored Ideas
- Annotations were provided only for a small portion of the dataset, so given high pixel-wise accuracy on this rather simple task creating masks by predicting labels for unannotated data would have expanded the dataset massively. 
- Knowledge Distillation
- Stronger regularization along with longer training schedule

## TODO
- [ ] Refactor variable parsing with python-fire
- [ ] Add Tensorboard logging
- [ ] Add Docker support
- [ ] Change constants to variables
