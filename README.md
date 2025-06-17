# AlpsCropFormer: Early/In-Season Crop Classification from Satellite Imagery Time Series

This project adapts a sota model to Swiss agricultural data with a specific focus on early/in-season classification of various crops.

## Research Foundation

This project is based on the research presented in:

- [ViTs for SITS: Vision Transformers for Satellite Image Time Series](https://openaccess.thecvf.com/content/CVPR2023/html/Tarasiou_ViTs_for_SITS_Vision_Transformers_for_Satellite_Image_Time_Series_CVPR_2023_paper.html) - Featured at CVPR 2023, this paper explores the application of Vision Transformers to Satellite Image Time Series analysis. The original repository can be found [here](https://github.com/michaeltrs/DeepSatModels).

## Environment Setup

### Installation of Miniconda
For the initial setup, please follow the instructions for downloading and installing Miniconda available at the [official Conda documentation](https://www.anaconda.com/docs/getting-started/miniconda/install).

### Environment Configuration
1. **Creating the Environment**: Navigate to the code directory in your terminal and create the environment using the provided `.yml` file by executing:

        conda env create -f deepsatmodels_env_*.yml

2. **Activating the Environment**: Activate the newly created environment with:

        source activate deepsatmodels

## Experiment Setup

- **Configuration**: Specify the base directory and paths for training and evaluation datasets within the `data/datasets.yaml` file.
- **Experiment Configuration**: Use a distinct `.yaml` file for each experiment, located in the `configs` folder. These configuration files encapsulate default parameters aligned with those used in the featured research. Modify these `.yaml` files as necessary to accommodate custom datasets.

### Training: Semantic Segmentation

To train for semantic segmentation, execute the following command, replacing `**` with the appropriate directory names:

        python train_and_eval/segmentation_training_transf.py --config_file configs/**/TSViT.yaml


### Pre-trained checkpoints
Download 5-fold PASTIS24 [pre-trained models and tensorboard files](https://drive.google.com/file/d/1AzWEtHxojuCjaIsekja4J54LuEb9e7kw/view?usp=share_link).


## BibTex
If you incorporate any data or code from this repository into your project, please acknowledge the source by citing the following work:

```
```

## License and Copyright

This project is made available under the Apache License 2.0. Please see the [LICENSE](https://github.com/jeffzwe/CropFormer/blob/main/LICENSE.txt) file for detailed licensing information.
