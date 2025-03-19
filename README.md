# Image classification using Euro-SAT data

## Overview

This project focuses on image classification using the Euro-SAT dataset. The Euro-SAT dataset consists of Sentinel-2 satellite images covering 13 spectral bands and includes 10 classes with a total of 27,000 labeled images.

## Dataset

The Euro-SAT 🚀 dataset can be downloaded from [here](https://github.com/phelber/eurosat). It contains images categorized into the following classes:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea/Lake

In this project the dataset is managed by 🤗 [Datasets](https://huggingface.co/docs/datasets/installation) library.

## Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.10+
- PyTorch 2.6
- PyTorch Lightning
- Datasets
- Transforms

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/tuanmp/euro-sat.git 
    cd euro-sat
    ```
2. Install the requirements
    ```
    conda create env --name euro-sat python=3.11
    pip install -r requirements.txt
    ```

2. Train an example the model:
    ```bash
    python lightning_train.py
    ```

<!-- 4. Evaluate the model:
    ```bash
    python evaluate_model.py
    ``` -->

<!-- ## Results

The trained model achieves an accuracy of X% on the test set. Below is a sample confusion matrix and classification report:

![Confusion Matrix](path_to_confusion_matrix_image) -->

<!-- ## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes. -->

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The Euro-SAT dataset creators
