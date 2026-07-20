import pandas as pd
import os
import cv2
import numpy as np
import pandas as pd
import time
import pickle

from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from cvxopt import matrix, solvers



def linear_kernel(X, Z, **kwargs):
    return X @ Z.T


def polynomial_kernel(X, Z, degree=3, gamma=None, coef0=1.0):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * X.dot(Z.T) + coef0) ** degree


def rbf_kernel(X, Z, gamma=None, **kwargs):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Z_norm = np.sum(Z ** 2, axis=1).reshape(1, -1)
    K = X_norm + Z_norm - 2 * X @ Z.T
    return np.exp(-gamma * K)


def decision_function(X_train, y_train, alpha, b, X_test,
                      kernel_fn, degree=None, gamma=None, coef0=None):
    K = kernel_fn(X_train, X_test, degree=degree, gamma=gamma, coef0=coef0)
    return (alpha * y_train) @ K + b


def load_standard_model(filename):

    with open(filename, "rb") as f:
        model = pickle.load(f)

    return model

def predict_standard(model, X):

    X = model["scaler"].transform(X)
    
    if model["kernel"] == "linear":
        kernel_fn = linear_kernel

    elif model["kernel"] == "poly":
        kernel_fn = polynomial_kernel

    else:
        kernel_fn = rbf_kernel

    scores = decision_function(
        model["X_train"],
        model["y_train"],
        model["alpha"],
        model["b"],
        X,
        kernel_fn,
        model["degree"],
        model["gamma"],
        model["coef0"]
    )

    return (scores > 0).astype(int)

def load_hog_features(csv_file, image_folder,image_size=(128,128)):
	df = pd.read_csv(csv_file, delimiter = ",")
	df = df[df["class_label"].isin(["fire","flood"])]
	#df = df[:100]
	X, y = [], []
	for i, row in df.iterrows():
		img_path = os.path.join(image_folder,row["id"])
		img = cv2.imread(img_path)
		
		if img is None:
			print("Cannot read:",img_path)
			continue
		img = cv2.resize(img,image_size)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		feature = hog(gray,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys')
		X.append(feature)
		y.append(row["class_label"])
	X = np.array(X)
	encoder = LabelEncoder()
	y = encoder.fit_transform(y)
	return X,y,encoder

if __name__ == "__main__":
	#X_tr, y_tr, label_encoder_tr = load_hog_features("C2A_disaster/train.csv", "C2A_disaster/C2A_Dataset/new_dataset3/train/images",image_size=(128,128))
	X_te, y_te, label_encoder_te = load_hog_features("C2A_disaster/test.csv", "C2A_disaster/C2A_Dataset/new_dataset3/test/images",image_size=(128,128))
	'''
	print(X_tr.shape)
	print(y_tr.shape)
	print(X_te.shape)
	print(y_te.shape)
	#cluster_values = [5,10,20, 40, 50, 100]
	cluster_values = [5,10,20]
	run_experiment(X_tr, y_tr,X_te, y_te, name="C2A",cluster_values=cluster_values,kernel_type="linear")
	run_experiment(X_tr, y_tr,X_te, y_te, name="C2A",cluster_values=cluster_values,kernel_type="poly")
	run_experiment(X_tr, y_tr,X_te, y_te, name="C2A",cluster_values=cluster_values,kernel_type="rbf")
	'''
	standard_model = load_standard_model("C2A_disaster/saved_models/standard_svm_linear.pkl")

	y_pred = predict_standard(standard_model, X_te)
	
	print("=============== Standard SVM (Linear) =======================")
	print("Accuracy :", accuracy_score(y_te, y_pred))
	print("Precision:", precision_score(y_te, y_pred))
	print("Recall   :", recall_score(y_te, y_pred))
	print("F1       :", f1_score(y_te, y_pred))
