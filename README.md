# Final project for the course 02456 Deep learning

## SEMANTIC SEGMENTATION OF CELLS FROM CLINICAL IMAGES USING DEEPLEARNING
Project 9: Segmentation of cell images
Supervised by: Peter Jensen, CTO of Cellari.io
By: Mikkel Lehmann (S153048), Sameer Agarwal (S192274), Nicklas Lund (S145129)

## Notes:
Notebooks for each of the two models can be found in the notebooks folder. Please note that saved models are saved using Git LFS. 

## Abstract
As  part  of  the  course  02456  Deep  Learning  at  DTU,  thisproject  has  set  out  to  build  and  evaluate  two  deep  learningmodels  with  supporting  functions  and  architecture  for  se-mantic segmentation of cells from clinical images.  The twomodels  implemented  were  (1)  U-Net  (2015)  and  (2)  MaskR-CNN (2018).  Significant support functions include a datapre-processing  pipeline  and  a  memory-efficient  batch  gen-erator.   The dataset used is from theGlaS@MICCAI’2015:Gland Segmentation Challenge Contestand  evaluation  wasdone using the official challenge evaluation metrics and score-board. Only limited tuning and performance improving workwas done for the two models.  Despite this, our DTU-UNETwas found to perform similar to peers of its time,  finishingat a 7th-place out of 12 peers, while the DTU-MASKRCNNwas  found  to  outclass  all  peers,  finishing  at  1st-place  outof  the  12,  with  a  clear  win  in  each  of  the  6  performancemetrics.  The findings show that the performance of seman-tic segmentation algorithms has significantly improved from2015 to 2018.  Based on this, it can be assumed that futurework  might  see  new  segmentation  algorithms  implementedin many real world settings in just a few years from now, asrobustness and performance is closing in on a level where thealgorithm can work in concert with humans, and maybe evenwork autonomously.
