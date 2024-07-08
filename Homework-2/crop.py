# import cv2
# import numpy as np
 
# img = cv2.imread('/Users/nick/Documents/Development/UPC/2nd-semester/TOML/HW2/distance-based-output/pictures/1-2.png')
# print(img.shape) # Print image shape
 
# # Cropping an image
# cropped_image = img[90:900, 60:1480]
 
# # Display cropped image
# # cv2.imshow("cropped", cropped_image)
 
# # Save the cropped image
# cv2.imwrite("/Users/nick/Desktop/1-2.png", cropped_image)
 

import os
import cv2

# Folder containing the images
folder_path = "/Users/nick/Desktop/tests"
new_folder_path = "/Users/nick/Desktop/tests/crop"

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (you may need to adjust this condition based on your file types)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read the image
        img = cv2.imread(os.path.join(folder_path, filename))

        # Print the image shape
        print(img.shape)
        
        # Apply the cropping operation
        # cropped_image = img[90:900, 60:1480]
        cropped_image = img[210:1820, 50:2890]

        # Save the cropped image
        cv2.imwrite(os.path.join(new_folder_path, filename), cropped_image)
