# Final project for the course 02456 Deep learning

## SEMANTIC SEGMENTATION OF CELLS FROM CLINICAL IMAGES USING DEEPLEARNING
Project 9: Segmentation of cell images
Supervised by: Peter Jensen, CTO of Cellari.io
By: Mikkel Lehmann (S153048), Sameer Agarwal (S192274), Nicklas Lund (S145129)

## Notes:
Notebooks for each of the two models can be found in the notebooks folder. Please note that saved models are saved using Git LFS. 

## Abstract
As part of the course 02456 Deep Learning at DTU, this project has set out to build and evaluate two deep learning models with supporting functions and architecture for semantic segmentation of cells from clinical images. The two models implemented were (1) U-Net (2015) and (2) Mask R-CNN (2018). Significant support functions include a data pre-processing pipeline and a memory-efficient batch generator. The dataset used is from the *GlaS@MICCAI'2015: Gland Segmentation Challenge Contest* and evaluation was done using the official challenge evaluation metrics and scoreboard. Only limited tuning and performance improving work was done for the two models. Despite this, our DTU-UNET was found to perform similar to peers of its time, finishing at a 7th-place out of 12 peers, while the DTU-MASKRCNN was found to outclass all peers, finishing at 1st-place out of the 12, with a clear win in each of the 6 performance metrics. The findings show that the performance of semantic segmentation algorithms has significantly improved from 2015 to 2018. Based on this, it can be assumed that future work might see new segmentation algorithms implemented in many real world settings in just a few years from now, as robustness and performance is closing in on a level where the algorithm can work in concert with humans, and maybe even work autonomously.
