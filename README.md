# IPRec
The code and data of SIGIR21 paper "Package Recommendation with Intra- and Inter-Package Attention Networks". 

## Dataset
Download from https://drive.google.com/file/d/12FO3IzmsqDYqa_RHpqdYC-2PrdZ3NCcf/view?usp=sharing 
Data format: user_id \t timestamp \t article_id \t label1 (user whether click) \t publish_media \t label2 (user whether subscribe the media) \t friend_list

## How to run
1. Generate datasets with data_process.ipynb file
2. Train the IPRec model with train.py file

## Requirements
Python == 3.6
Tensorflow == 1.12
