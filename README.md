# Using Autoencoders and CNNs for CAPTCHA Recognition   
## 2022 FA CS534 Final Project 

### Contributers 
Katherine Keegan, Anna Pritchard, and Hong kyu Lee


### Abstract  
An autoencoder is a promising option for replacing the traditional
image pre-processing methods in CAPTCHA, and with further
training could be utilized to recognize a variety of CAPTCHA se-
quence types. More broadly, if models such as ours can decode
CAPTCHA sequences with high accuracy, complexity needs to be
added to the sequences to enhance website security. Given these
findings, we anticipate autoencoder + CNN models may be used as
a technique to test the strength of CAPTCHA sequences, providing
insight to the strengths and weaknesses of AI in breaching website
security.  

### How to run

1. Download the dataset. The model can run [small dataset](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images) and [large dataset](https://www.kaggle.com/datasets/parsasam/captcha-dataset). 

2. Place the datasets into /dataset/CAPTCHA_SIMPLE and /dataset/CAPTCHA_LARGE respecitvely.

3. Run vaious exeriments by calling .py file such as `python vanila_cnn_large.py`. Each code has pre-defined parameters encapsulated in a dictionary. If you'd like to try various hyperparameters, edit the code before running it.


| Code | What it does| Hyperparamters you can change|
|------|-------------|---------------|
| vanila_cnn_large.py | Train a vanila Resnet-18 model on large dataset | epoch, <br> learning rate, <br>  batch size |
| vanila_cnn_small.py | Train a vanila Resnet-18 model on small dataset | epoch, <br> learning rate, <br> batch size |
| train_preprocess_baseline_small.py | Train a pre-processing + Resnet-18 model on the small dataset | epoch, <br> learning rate,<br> batch size, <br> **denoise mode** |
| train_preprocess_baseline_large.py | Train a pre-processing + Resnet-18 model on the large dataset | epoch, <br> learning rate, <br> batch size, <br> **denoise mode** |
| train_preprocess_encoder.py | Train an autoencoder on the small dataset using selected pre-processing options | epoch, <br> learning rate, <br> batch size, <br> **multiple denoise modes*, <br>  **layers config** , <br>  **cnn config**|
| train_preprocess_encoder_large.py | Train an autoencoder on the large dataset using selected pre-processing options | epoch, <br> learning rate, <br> batch size, <br> **multiple denoise modes**,  <br> **layers config** , <br>  **cnn config**|
| encoder_classifier_small.py | Train a proposed autoencoder + Resnet 18 model on the small dataset | epoch, <br> learning rate, <br> batch size,  <br> **layers config** , <br>  **cnn config** <br> **load encoder**, **encoderpath**, <br> **encoder name**|
| encoder_classifier_large.py | Train a proposed autoencoder + Resnet 18 model on the large dataset | epoch, <br> learning rate, <br> batch size,  <br> **layers config** , <br>  **cnn config**, <br> **load encoder**, <br> **encoderpath**,  <br> **encoder name**|
| encoder_classifier_auto.py | Train a proposed autoencoder + Resnet 18 model from scratch on the large dataset | epoch, <br> learning rate, <br> batch size,  <br> **layers config**, <br>  **cnn config**, <br> **load encoder**, **encoderpath**,  <br> **encoder name**|


#### More explanation on parameters

* denoise mode
    * type: int  
    * When running on  the small dataset, it can be one of 1, 2, 3 and 4. <br> When running on the large dataset, it can be 1, or 2. <br> It reflects the pre-processing options from the report. Details can be found from `/dataset/dataset.py`  

* ultiple denoise modes
    * type: Iist of int
    * Same settings with denoise mode, but numbers are included in a list. <br> The autoencoder trainer will use all options in the list.

* layers config
    * type: List of int
    * Defines a structure of an autoencoder.
        * -1 will append a CNN that reduces output dimension
        * +1 will append a CNN that increases output dimension
        * 0 will append a CNN that preserves input dimension
    * The number of -1 and +1 in the list has to be same. The code will run even though they're not same, but will not perform well.

* CNN config
    * type: int
    * There are three pre-defined CNN preset in `/model/encoder.py`. It is simply a dictionary with all hyperparamter for CNNs. Use one of three options by `CNN_PRESET[0]`, `CNN_PRESET[1]`, or `CNN_PRESET[2]`

* load encoder
    * type : Bool
    * If true, the encoder_classifer file will load the autoencoder from hyperparameter `encoder_path`. <br>
    * If false, the encoder_classifier will create a new autoencoder.

* encoder path 
    * type : String
    * Path to the encoder that will be used for the training a CNN. Do not include the full path. Instead, place the path to the folder that has the model. 

* encoder name
    * type: String
    * Name of the encoder model file. Usually the autoencoder models are saved as `model.pt` but sometimes they're not. 