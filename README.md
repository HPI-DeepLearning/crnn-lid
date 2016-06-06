# Master Thesis
Automatic spoken language identification (LID) using deep learning.

## Motivation
I want to classify the spoken language within audio files, a process that usually serves as the first step for NLP or speech transcription.

I used two deep learning approaches using the Tensorflow and Caffe frameworks for different model configuration.

## Repo Structure

- **/data**
  - Scripts to download training data from Voxforge and Youtube. For usage details see below.  
- **/Evaluation**
  - Prediction scripts for single audio files or list of files using Caffe
- **/Preprocessing**
  - Includes all scripts to convert a WAV audio file into spectrogram and mel-filter spectrogram images using a Spark Pipeline.
  - All scripts to create/extract the audio features
  - To convert a directory of WAV audio files using the Spark pipeline run: `./run.sh --inputPath {input_path} --outputPath {output_path} | tee sparkline.log -`
- **/models**
  - All our Caffe models: Berlin_net, Topcoder, VGG_M
  - Berlin_net: 3Conv + Batch Normalisation, 2 FullyConnected Layer (Shallow Architecture)
  - Topcoder_net: (Deep Architecture) inspired by [Topcoder's spoken language identification challenge](https://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks/)
  - Finetuning of [Inception-v3](https://github.com/tensorflow/models/tree/master/inception)
- **/tensorflow**
  - All the code for setting up and training various models with Tensorflow.
  - Includes training and prediction script. See `train.py` and `predict.py`.
  - Configure your learning parameters in `config.yaml`.
  - Add or change network under `/tensorflow/networks/instances/`.
- **/tools**
  - Some handy scripts to clean filenames, normalize audio files and other stuff.


## Requirements
- Caffe 
- TensorFlow
- Spark
- Python 2.7
- OpenCV 2.4+
- youtube_dl

```
// Install additional Python requirements
pip install -r requirements.txt
pip install youtube_dl
```

## Datasets
Downloads training data / audio samples from various sources.

#### Voxforge
- Downloads the audio samples from www.voxforge.org for some languages
```bash
/data/voxforge/download-data.sh
/data/voxforge/extract_tgz.sh {path_to_german.tgz} german
```

#### Youtube
- Downloads various news channels from Youtube.
- Configure channels/sources in `youtube/sources.yml`

```python
python /data/youtube/download.py
```

## Models

I trained models for 2/4 languages (English, German, French, Spanish). 



#### Training

```
// Caffe:
/models/{model_name}/training.sh
```


```
// Tensorflow:
python /tensorflow/train.py
```

#### Labels
```
0 English, 
1 German, 
2 French, 
3 Spanish
```


## Training Data
For training I downloaded news reel videos from Youtube. Check out the **/data** directory.

## LICENSE
TBD

