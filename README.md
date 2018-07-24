# Instrument Recognition
The instrument recognition model in this repo is trained on MusicNet dataset, which contains 7 kinds of instrument - Piano, Violin, Viola, Cello, Clarinet, Horn and Bassoon.

## File structure

### For training and evaluation
- **./data/**: store the pre-train models' parameters. Model's parameters will also store in this directory during training process.
- **./function/**: store all the python files related to training and testing
    - **evl.py**: for score computation and evaluation
    - **fit.py**: training process
    - **lib.py**: loss function and model initialization
    - **model.py**: model structure
    - **norm_lib.py**: data normalization
- **process.py**: data pre-processing
- **config.py**: configuration option
- **run.py**: start the training process
- **test_frame.py**: start the evaluation process

## For real music testing
- **./mp3/**: folder to put the mp3 files you want to predict
- **./plot/**: folder to store the result graphic
- **./result/**: folder to store the result raw data
- **predict_pitch.py**: pitch extraction
- **prediction.py**: start prediction process

## Requirement
- librosa==0.6.0
- matplotlib==2.2.0
- numpy==1.14.2
- pytorch==0.3.1
- mir-eval==0.4
- scikit-learn==0.18.1
- scipy==1.0.1

## Prediction Process
This section is for those who want to load the pre-train model directly for real music testing
1. Put MP3/WAV files in the "mp3" folder
2. Run the 'predict_pitch' python file with the name of the song as the first arg
```
python predict_pitch.py test.mp3
```
Run the prediction python file with the name of the song as the first arg
```
python prediction.py test.mp3
```
3. Prediction result will be shown as a picture and stored in the "plot" folder. Prediction raw data will be stored in the "result" folder

Training and Evaluation Process
1. Download MusicNet dataset (https://homes.cs.washington.edu/~thickstn/start.html)
2. Follow the guid in 'process.py' to process the data
3. Modify 'config.py' for training and evaluation configuration
4. Run the python script 'run.py' to start the training
5. Run the python script 'test_frame.py' to start evaluation

Paper
* Yun-Ning Hung and Yi-Hsuan Yang, "FRAME-LEVEL INSTRUMENT RECOGNITION BY TIMBRE AND PITCH" to appear at International Society for Music Information Retrieval Conference (ISMIR), 2018

Reference
Please cite these two papers when you use the MusicNet dataset and the Pitch estimator.
* John Thickstun, Zaid Harchaoui, and Sham M.Kakade. Learning features of music from scratch. In Proc. Int. Conf. Learning Representations, 2017. [Online] https://homes.cs.washington.edu/~thickstn/musicnet.html
* John Thickstun, Zaid Harchaoui, Dean P. Foster, and Sham M. Kakade. Invariances and data augmentation for supervised music transcription. In Proc. Int. Conf. Acoustics, Speech, and Signal Processing, 2018
