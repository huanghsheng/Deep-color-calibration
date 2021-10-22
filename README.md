# Pytorch_Adain_from_scratch
Pytorch implementation of [Deep Color Calibration for UAV Imagery in Crop Monitoring Using Semantic Style Transfer with Local to Global Attention](https://www.sciencedirect.com/science/article/pii/S030324342100297X).

This project aims to build a general framework for color calibration in agricultural UAV remote sensing.

------

## Requirements

- Python 3.6
- PyTorch 1.5.1
- TorchVision
- Anaconda environment recommended here!
- GPU environment is required for training and testing


## Usage
------
## Dataset preparation
The dataset used in this study includes the UAV imagery of rice, bean, and cotton(CropUAVDataset). The imagery were collected at different sites and different dates, where the ortho-mosaics present siginifcant color cast and color inconsistency. The dataset is available at https://drive.google.com/drive/folders/15TETGMJxQvuBOqjCTGQPu5OTCizF4pPK.
After downloading, the dataset should be placed at the "data" folder under the project.
For more details on our dataset, you can refer to our paper [Deep Color Calibration for UAV Imagery in Crop Monitoring Using Semantic Style Transfer with Local to Global Attention](https://www.sciencedirect.com/science/article/pii/S030324342100297X).


## Testing
Use the script of test_seg5.py, where the following parameters have to be defined in the script:
    # args.model_state_path is the directory where the trained model was saved
    # In this implementation, three trained models (for rice, bean, and cotton) were saved under the directory of ./result/model_state
        args.model_state_path = './result/model_state/rice.pth'
    
    # args.content_dir is the file that record the testing samples
    args.content_dir = './data/val_rice.txt'
    
    # args.style is the reference image
    args.style = './data/2017-9-30/images/DJI_0426_4_4.png'
    
    args.out_dir = './result/test_outputs'


## Evaluation
evaluate1.py: Evaluation for the color cast with the metrics of KL divergence and Hellinger distance.

evaluate2.py: Evaluation for the loss of spatial details.


## Traning
Use the script of train_seg5.py, where the following parameters have to be defined in the script:

    args.batch_size = 1
    
    args.epoch = 100 
    
    args.snapshot_interval = 100
    
    # args.semantic_train_txt is the file containing the training samples	
    args.semantic_train_txt = './data/train_rice.txt'
    
    # args.semantic_train_txt is the file containing the testing samples
    args.semantic_test_txt = './data/val_rice.txt'


## References
- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [Pytorch_Adain_from_scratch](https://github.com/irasin/Pytorch_AdaIN) 

