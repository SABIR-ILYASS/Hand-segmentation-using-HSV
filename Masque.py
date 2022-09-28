"""
Extraction des masques des mains
Notre but est de segmenter les masques des mains, par un leurs couleurs.
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# paramètres:
espace = 'HSV'
nbr_classes = 180
seuil_min = 10
seuil_max = 25

# lire et affichage de l'image qu'on veut
image_name = 'boeuf1'
chemain = "BDD/" + image_name + ".bmp"
image = cv.imread(chemain)
print(chemain)

# redimensionnement de l'image
dimensions = image.shape
width = dimensions[0]
height = dimensions[1]
image = image[100:int(width * 9 / 10), 100: int(height * 9 / 10)] # extraire une fenetre de l'image
image = cv.resize(image, (int(width / 2), int(height / 2)))
# affichage de l'image
cv.imshow(image_name, image)

# changement de l'espace colorimetrique
image_changed = cv.cvtColor(image, eval("cv.COLOR_BGR2" + espace))
cv.imshow("image in HSV", image_changed)

# calcul et affichage de l'histogramme de la composante H de l'image (en HSV)
histogramme = cv.calcHist([image_changed], [0], None, [nbr_classes], [0, 180])

# roi_x, roi_y, roi_w, roi_h = cv.selectROI('ROI', image, False, False)

cv.normalize(histogramme, histogramme, 0, 255, cv.NORM_MINMAX)
plt.plot(histogramme )
plt.title("histogramme de la composante H (dans HSV) de l'image:  " + chemain)
plt.show(block = False)
plt.pause(0.01)
plt.clf()

# construire le masque
mask = cv.calcBackProject([image_changed], [0], histogramme, [0, nbr_classes], 1)
_, mask2 = cv.threshold(mask, seuil_min, seuil_max, cv.THRESH_BINARY)
mask2 = cv.erode(cv.dilate(mask, None, iterations = 3), None, iterations = 3)
cv.imshow("masque de l'image " + chemain ,mask2)
print("it's ok")
cv.waitKey(0)
cv.destroyWindow()