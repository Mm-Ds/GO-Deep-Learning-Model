# A Deep Learning Model for GO policy and value prediction.

This project aims at improving a given base model (a ResNet model, given in `golois.py`) that predicts GO games' policy and value.

All files except `final_main.py` were provided. The latter implements the improved model.

Initially changes of architecture have been attempted : Use of Bottleneck blocks (see commented part of the code in `final_main.py`) (and ResNet B blocks) instead of simple ResNet blocks. The base model remains better.

Hence the retained model has the same architecture as the base one (in terms of used blocks) but is wider (larger filter size, yet within a constraint of a maximum of 10^6 parameters), is fine-tuned (mainly with the addition of Learning rate decay) and trained on larger batch sizes for more epochs.
Over 20 epochs it reaches a policy accuracy of 35,8% and a value MSE of 25%, compared to 32% and 24% for the base model.

The final version of it that was reached was trained on 210 epochs and has 41.7 % policy accuracy and 21.7 % value MSE.

**Note: Games data used for training exceeding github's file size limit, it is not provided here.**
