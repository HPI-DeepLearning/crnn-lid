# Master Thesis
Language Identification Using Deep Convolutional Recurrent Neural Network
[Read the complete thesis here.](https://github.com/hotzenklotz/master-thesis/blob/master/thesis/Masters%20Thesis%20Tom%20Herold.pdf)


## Repo Structure

- **/data**
  - Scripts to download training data from Voxforge, European Parliament Speech Repository and YouTube. For usage details see below.
- **/keras**
  - All the code for setting up and training various models with Keras/Tensorflow.
  - Includes training and prediction script. See `train.py` and `predict.py`.
  - Configure your learning parameters in `config.yaml`.
  - More below
- **/tools**
  - Some handy scripts to clean filenames, normalize audio files and other stuff.
- **/webserver**
  - A demo project for language identification. A small web server with a REST interface for classification and a small web frontend to upload audio files.
- **/notebooks**
  - Various jupyter notebooks with various audio experiments
- **/thesis**
  - Latex sources & figures for my thesis
- **/paper**
  - Some papers / related worked I used in preparation for this thesis. Not complete.



## Requirements
- Keras 1
- TensorFlow
- Python 3.4
- youtube_dl
- sox


## Training Data
Downloads training data / audio samples from various sources.

#### Voxforge
- Downloads the audio samples from www.voxforge.org for some languages
```bash
/data/voxforge/download-data.sh
/data/voxforge/extract_tgz.sh {path_to_german.tgz} german
```

#### Youtube
- Downloads various news channels from YouTube.
- Configure channels/sources in `/data/sources.yml`

```python
python /data/download_youtube.py
```

#### European Speech Repository
- Downloads various speeches and press conferences from [European Speech Repository](https://webgate.ec.europa.eu/sr/).
- needs WebDriver/Selenium & Firefox

```python
python /data/download_europe_speech_repository.py
```

## Convert Audio Files to Spectrograms

Make sure you have [SoX](http://sox.sourceforge.net/) installed. To create 500x129x1 grayscale spectrogram images run the following script.

```
python /data/wav_to_spectrogram.py --source <path> --target <path>
```

The above script uses different spectrogram generators to augment the data with additional noise or background music if needed. Adjust the imports accordingly.


## Models

I trained models for 4 languages (English, German, French, Spanish) and 6 languages (English, German, French, Spanish, Chinese, Russian).
They might be released later.


#### Training & Prediction

To start a training run, set all the desired properties and hyperparameters i the config.yaml file and train with Keras:
```
python /keras/train.py --config <config.yaml>
```

To predict a single audio file run:
```
python /keras/train.py --model <path/to/model> --input <path/to/speech.mp3>
```
Audio files can be in any format understood by SoX. The pretrained model files need to be caomptible with Keras v1.


#### Labels
```
0 English,
1 German,
2 French,
3 Spanish,
4 Mandarin Chinese,
5 Russian
```

## LICENSE
TBD

