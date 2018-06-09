# Quick Analytics

## Background and motivation
When creating scripts and building tools for data analytics (including various machine learning tasks), there seems to be many components that are commonly used across multiple frameworks and libraries. These components are often re-used and re-deployed in a customised fashion to suit a spectrum of analytics or machine learning model building tasks. The intention here is to accumulate these scripts in one place and reduce the amount of code one has to do for carrying out analytics tasks.

## Content inclusions
Currently, only Python scripts and Jupyter notebooks are included in this package. These cover Sklearn and Keras/Tensorflow implementations of:
- Data preprocessing (PCA, feature engineering, visualisation)
- Model training (decision tree, logistic regression, SVM, neural networks, convolutional neural networks, etc)
- Model evaluation

### Note
Place input data (csv, xlsx files) in the "Data" folder if using the Jupyter notebook template examples.

## Requirements
- Python 3
- Numpy
- Scipy
- Pandas
- Matplotlib
- Seaborn
- Sklearn
- Tensorflow
```
$ pip install numpy
$ pip install scipy
$ pip install pandas
$ pip install matplotlib
$ pip install seaborn
$ pip install scikit-learn
$ pip install tensorflow
```

## Future updates
- More Python Jupyter notebook examples that cover different model implementations
- Functionality that allows comparison of model performance under collaboration (by recording model parameters that are used by different Jupyter notebooks), aiming improve productivity between potential team members working from a shared workspace.
- Implementations of data preprocessing and modelling in R.
