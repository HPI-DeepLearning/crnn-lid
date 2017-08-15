# Language Identification Using Deep Convolutional Recurrent Neural Networks

This repository contains the code for the paper "Language Identification Using Deep Convolutional Recurrent Neural Networks", which will be presented at the 24th International Conference on Neural Information Processing (ICONIP 2017).

## Structure of the Repository

- **/data**
  - Scripts to download training data from Voxforge, European Parliament Speech Repository and YouTube. For usage details see the README in that folder.
- **/keras**
  - All the code for setting up and training various models with Keras/Tensorflow.
  - Includes training and prediction script. See `train.py` and `predict.py`.
  - Configure your learning parameters in `config.yaml`.
  - More below
- **/tools**
  - Some handy scripts to clean filenames, normalize audio files and other stuff.
- **/web-server**
  - A demo project for language identification. A small web server with a REST interface for classification and a small web frontend to upload audio files. For more information see README in that folder.

## Requirements

You can install all python requirements with `pip install -r requirements.txt` in the respective folders. You will additionally need to install the following software:
- youtube_dl
- sox

## Models

The repository contains a model for 4 languages (English, German, French, Spanish) and a model for 6 languages (English, German, French, Spanish, Chinese, Russian). You can find these models in the folder `web-server/model`.


#### Training & Prediction

To start a training run, go into the `keras` directory, set all the desired properties and hyperparameters in the config.yaml file and train with Keras:
```
python train.py --config <config.yaml>
```

To predict a single audio file run:
```
python predict.py --model <path/to/model> --input <path/to/speech.mp3>
```
Audio files can be in any format understood by SoX. The pretrained model files need to be caomptible with Keras v1.

To evaluate a trained model you can run:
```
python evaluate.py --model <path/to/model> --config <config.yaml> --testset True
```

You can also create a visualisation of the clusters the model is able to produce by using our `tsne.py` script:
```
python tsne.py --model <path/to/model> --config <config.yaml>
```
In case you are interested in creating a visualization of what kind of patterns excite certain layers the most, you can create such a visualization with the following command:
```
python visualize_conv.py --model <path/to/model>
```

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

GPLv3 see `LICENSE` for more information.

## Citation

If you find this code useful, please cite our paper:

```
@article{HPI_crnn-lid,
  Author = {Christian Bartz, Tom Herold, Haojin Yang, Christoph Meinel},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {TODO},
  Title = {Language Identification Using Deep Convolutional Recurrent Neural Networks},
  Year = {2017}
}
```

