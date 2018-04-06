# MOSiamFC - TensorFlow

Forked version of SiamFC that supports Multi Object Tracking. Also this version is made to be compatible with Python 3.x .

**Note1**: Tested in Windows platform, using the [Anaconda Platform](https://www.anaconda.com/download/).

**Note2**: This fork also applied the [pull request](https://github.com/torrvision/siamfc-tf/pull/5) to use OpenCV to show the frame results.

For more information, please refer to the original [Repository](https://github.com/torrvision/siamfc-tf).

## Settings things up with virtualenv
1) Get virtualenv if you don't have it already
`pip install virtualenv`
1) Create new virtualenv with Python 3.6
`virtualenv --python=/usr/bin/python3.6 mo-siam`
1) Activate the virtualenv
`source ~/mo-siam/bin/activate`
1) Clone the repository
`git clone https://github.com/lukaswals/siamfc-tf.git`
1) `cd siamfc-tf`
1) Install the required packages
`sudo pip install -r requirements.txt`
1) `mkdir pretrained data`
1) Download the [pretrained networks](https://bit.ly/cfnet_networks) in `pretrained` and unzip the archive (we will only use `baseline-conv5_e55.mat`)
1) Download [video sequences](https://drive.google.com/file/d/0B7Awq_aAemXQSnhBVW5LNmNvUU0/view) in `data` and unzip the archive.

## Bounding Box Input
It's important to note that the provided Ground Truth for the video sequences are only for one object.

## Running the tracker
1) Set `video` from `parameters.evaluation` to `"all"` or to a specific sequence (e.g. `"vot2016_ball1"`)
1) See if you are happy with the default parameters in `parameters/hyperparameters.json`
1) Enable Multi-Object tracking by setting `multi_object` from `parameters.evaluation` to 1 (default value)
1) Optionally enable visualization in `parameters/run.json`
1) Call the main script (within an active virtualenv session)
`python run_tracker_evaluation.py`


## Fork Authors
* Lucas Wals

## Original Authors

* [**Luca Bertinetto**](https://www.robots.ox.ac.uk/~luca)
* [**Jack Valmadre**](http://jack.valmadre.net)

## References
If you find their work useful, please consider citing

↓ [Original method] ↓
```
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850--865},
  year={2016}
}
```
↓ [Improved method and evaluation] ↓
```
@article{valmadre2017end,
  title={End-to-end representation learning for Correlation Filter based tracking},
  author={Valmadre, Jack and Bertinetto, Luca and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip HS},
  journal={arXiv preprint arXiv:1704.06036},
  year={2017}
}
```

## License
This code can be freely used for personal, academic, or educational purposes.

