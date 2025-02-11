import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
#Preparing Data
input_dir = 'C:/Users/arjun/Desktop/Jupyter Notebook/Image_Classification_WOW/Wonders_of_the_world_dataset/Wonders of World'
categories = ['burj_khalifa', 'chichen_itza', 'christ_the_reedemer', 'eiffel_tower', 'great_wall_of_china', 'machu_pichu',
               'pyramids_of_giza', 'roman_colosseum', 'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

data = []
labels = []

# Loop through each category folder
for category_idx, category in enumerate(categories): # Enumerate assigns an index to each category
    category_path = os.path.join(input_dir, category) # Creates a path to the category folder
    print(f"Processing category: {category}")  # Progress indicator
    for file in os.listdir(category_path): # Lists all images in the current category folder
        img_path = os.path.join(category_path, file)  # Builds full path to each image
        print(f"Reading file: {file}")  # Show file being processed
        img = cv2.imread(img_path)  # Reads the image as Numpy array
        img = cv2.resize(img, (150, 200))  # Resize to 150x200 pixels
        img = img.flatten()/255.0 # Converts the 2D image (150×200) into a 1D array (150×200 = 30,000 values) and normalizes pixel values
        data.append(img) 
        labels.append(category_idx)  # Assign a numerical label to each category

# Convert lists to NumPy arrays for efficient storage and faster computation
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

# train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classifier
classifier = SVC()
parameters = [{'gamma':[0.01, 0.001, 0.0001], 'C':[1,10,100,1000]}]

grid_search = GridSearchCV(classifier,parameters)

grid_search.fit(x_train, y_train)

# testing performance
best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print("{}% of the images were correctly classified".format(str(score*100)))
pickle.dump(best_estimator, open(r'C:\Users\arjun\Desktop\Jupyter Notebook\Image_Classification_WOW\model.p', 'wb'))
