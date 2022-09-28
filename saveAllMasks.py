"""
Author: SABIR ILYASS
Hand segmentation using HSV 
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob

def FillHole(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        black_image = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv.drawContours(black_image, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
    out = sum(contour_list)
    return out

def area_opening(mask):
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)

    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size to keep an element
    min_size = np.sum(mask > 0) * 0.5;

    # answer image as a np.array
    mask2 = np.zeros((output.shape))

    # for every component in the image, keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask2[output == i + 1] = 255

    mask2 = mask2.astype(np.uint8)

    return mask2

# paramètres:
espace = 'HSV'
nbr_classes = 180
seuil_min = 85
seuil_max = 225


for img in glob.glob('BDD/*.bmp'):
    # lire et affichage de l'image qu'on veut
    image = cv.imread(img)
    print(img)
    chemain = img[4:]

    # redimensionnement de l'image
    dimensions = image.shape
    width = dimensions[0]
    height = dimensions[1]
    image = image[100:int(width * 9 / 10), 100: int(height * 9 / 10)] # extraire une fenetre de l'image
    image = cv.resize(image, (int(width / 2), int(height / 2)))
    # affichage de l'image
    # cv.imshow(img, image)

    # changement de l'espace colorimetrique
    image_changed = cv.cvtColor(image, eval("cv.COLOR_BGR2" + espace))
    (h, s, v) = cv.split(image_changed)
    v[:] = seuil_min
    # h[:] = seuil_max
    img = cv.merge((v, v, s))
    rgb = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    rgb[:, :, 0] = rgb[:, :, 0] * (0.2989/0.587)
    rgb[:, :, 2] = rgb[:, :, 2] * (0.333/0.587)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # construire le masque
    kernel = np.ones((5, 5), np.float32) / 25
    gray = cv.filter2D(gray, -1, kernel)
    es = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    gray = cv.erode(gray, es, iterations = 1)
    gray = cv.dilate(gray, es, iterations = 1)
    gray = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    #mask = cv.dilate(cv.erode(gray, None, iterations=3), None, iterations=3)
    _, mask = cv.threshold(gray, seuil_min, seuil_max, cv.THRESH_BINARY)
    mask = area_opening(mask)
    mask = FillHole(mask)

    # save masks
    chemain_mask = "Masques/masque_" + chemain
    status = cv.imwrite(chemain_mask, mask)

    if cv.waitKey() & 0xFF == ord('q'):
        break