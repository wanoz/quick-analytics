# Quick Analytics

## Background and motivation
When creating scripts and building tools for data analytics or machine learning tasks, there seems to be many components that are commonly used across multiple frameworks and libraries. These components are often re-used and re-deployed in a customised fashion to suit different use-cases. The intention here is to streamline and aggregate useful functions of these scripts in one place so as to reduce the amount of code one has to do in data analytics or model experimentation.

## Content inclusions
Currently, only Python scripts and Jupyter notebooks are included in this package. These cover Sklearn and Keras/Tensorflow implementations of:
- Data preprocessing (PCA, feature engineering, visualisation)
- Preliminary model training and evaluation

### Note
Place input data (csv, xlsx files) in the "Data" folder if using the Jupyter notebook template examples.

## Requirements
- Python 3 (download from: https://www.python.org/downloads/)

- Install library packages:
  - Numpy
  - Scipy
  - Pandas
  - Matplotlib
  - Seaborn
  - Sklearn
  - Tensorflow
  - Skimage
```
$ pip install numpy
$ pip install scipy
$ pip install pandas
$ pip install matplotlib
$ pip install seaborn
$ pip install scikit-learn
$ pip install tensorflow
$ pip install scikit-image
```

## Future updates
- More Python Jupyter notebook examples that cover different explorative analytics environments and different model implementations.
- Functionality that allows comparison of model performance under collaboration (by recording model parameters that are used by different Jupyter notebooks), aiming improve productivity between potential team members working from a shared workspace.
- Implementations of data preprocessing and modelling in R.
