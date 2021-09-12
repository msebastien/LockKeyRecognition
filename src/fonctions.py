from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import math
from numpy import diff
import json

NFEATURES_MATCHING = 100000  # Nombre de points clés pour le feature matching
NB_MAX_KEY_POINTS = 70  # Nombre de points de correspondance gardés


def resize_auto(img):
    scale_percent = compute_scale_percent(img)  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)  # resize image
    return resized


def compute_scale_percent(img):  # scale image to be maximum 600 pixels along the longest dimension
    longest_dim = img.shape[0]
    if longest_dim < img.shape[1]:
        longest_dim = img.shape[1]

    scale_percent = 100  # default case, image size is already ok so we keep 100% of it
    if longest_dim > 600:  # if too large, compute scale_percent
        scale_percent = (600 * 100) / longest_dim
    return scale_percent


def rotate(img, angle):
    h = img.shape[0]  # get image height, width
    w = img.shape[1]
    center = (w / 2, h / 2)  # calculate the center of the image
    M = cv.getRotationMatrix2D(center, angle, 1)

    # Perform the counter clockwise rotation holding at the center
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS,
                            borderMode=cv.BORDER_REPLICATE)
    return rotated


def hough_transform(img, angular_res, canny_threshold1, canny_threshold2):
    img2 = np.copy(img)

    canny = cv.Canny(img2, canny_threshold1, canny_threshold2)
    ht = cv.cvtColor(canny, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(canny, 1, np.pi / 180 * angular_res, 150)  # Standard Hough Line Transform
    thetas = []
    if lines is not None:
        for i in range(0, len(lines)):
            theta = lines[i][0][1]
            thetas.append(theta)
            rho = lines[i][0][0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(ht, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)
    theta_median = np.median(thetas)
    return ht, theta_median * 180 / math.pi


def feature_match_crop(key, pattern, canny_threshold1, canny_threshold2):
    # Conversion en niveaux de gris
    rotated_key_photo = cv.cvtColor(key, cv.COLOR_BGR2GRAY)
    pattern = cv.cvtColor(pattern, cv.COLOR_BGR2GRAY)

    # Detection des contours de la photo et du pattern
    key_edges = cv.Canny(rotated_key_photo, canny_threshold1, canny_threshold2)
    pattern_edges = cv.Canny(pattern, canny_threshold1, canny_threshold2)

    # Recherche de points clés sur les images
    orb = cv.ORB_create(nfeatures=NFEATURES_MATCHING, scoreType=cv.ORB_FAST_SCORE)
    kp1 = orb.detect(pattern_edges, None)
    kp1, des1 = orb.compute(pattern_edges, kp1)
    kp2 = orb.detect(key_edges, None)
    kp2, des2 = orb.compute(key_edges, kp2)

    # Recherche de correspondances et tri dans l'ordre
    bf = cv.BFMatcher(cv.NORM_HAMMING2)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Affichage des 70 meilleures correspondances
    matches_img = cv.drawMatches(pattern_edges, kp1, key_edges, kp2, matches[:NB_MAX_KEY_POINTS], None, flags=2)

    # Rognage de la photo de la clé à partir des points trouvés
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:NB_MAX_KEY_POINTS]])
    min_x, min_y = np.int32(dst_pts.min(axis=0))
    max_x, max_y = np.int32(dst_pts.max(axis=0))

    return key[min_y:max_y, min_x:max_x], matches_img


# Recherche de cercle puis rogagne de l'image
def find_circles_and_crop(img):
    w = img.shape[1]

    # copie de l'image pour afficher les cercles
    img2 = np.copy(img)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # Rayons minimum et maximum du cercle à chercher
    radius_min = int(0.8 * (w / 2))
    radius_max = int(w / 2)

    # Recherche de cercles dans l'image
    circles = cv.HoughCircles(img2, cv.HOUGH_GRADIENT, 1.9, 180, param1=100, param2=60, minRadius=radius_min,
                              maxRadius=radius_max)

    # Si un cercle a été trouvé
    if circles is not None and circles.shape[0] == 1:
        circles = np.round(circles[0, :]).astype("int")
        # print("Circles :", circles)
        # Affichage pour débugage
        for (center_circle_x, center_circle_y, rayon) in circles:
            cv.circle(img2, (center_circle_x, center_circle_y), rayon, (0, 255, 0), 4)
            cv.imshow("Cercles trouvés", img2)
            cv.waitKey()
        # Si le centre du cercle est dessous du milieu de l'image (verticalement), la clé est à l'envers
        if circles[0][1] > img.shape[0] / 2:
            print("Clé a l'envers")
            img = img[:circles[0][1] - circles[0][2], :img.shape[1]]
            img = cv.flip(img, 0)
        else:
            img = img[circles[0][1] + circles[0][2]:, :img.shape[1]]
        return img
    else:
        print("Aucun cercle trouvé...")
        return None


# Recherche de la profondeur des crans
def measure_cuts(img, contours, cuts_centers, w):
    # Position "en largeur" sur l'image des crans
    x_position_cuts = []
    for i in range(len(cuts_centers)):
        x = [x[1] for x in contours if x[0] == cuts_centers[i]][0]
        cv.circle(img, (x, cuts_centers[i]), 5, (255, 0, 0), 5)
        x_position_cuts.append(x)
    return x_position_cuts, img


# Calcul et traçage des positions (en hauteur) des crans.
# Calculé à partir de la position du premier cran et de l'espacement défini selon le type de clé
def cuts_positions(mm_in_pixel, first_notch, notch_space):
    cuts_centers = []
    for i in range(5):
        cut_center = round(first_notch + notch_space * i * mm_in_pixel)
        cuts_centers.append(cut_center)
    return cuts_centers


# Recherche du "shoulder", le pixel blanc le plus à droite de l'image après avoir été rognée
def find_shoulder_position(polygon_edges):
    print(polygon_edges)
    list_max_x = max(polygon_edges, key=lambda x: x[1])
    return list_max_x[1], list_max_x[0], polygon_edges.index(list_max_x)


# Recherche du premier cran de la clé
def get_first_notch(polygon_edges):
    list_min_x = min(polygon_edges, key=lambda x: x[1])
    return list_min_x[1], list_min_x[0]


# Calcul difference en horizontal entre le "shoulder" et les crans puis conversion en mm
def get_cuts_depth(crans, shoulder, mm_in_pixel):
    depths_in_mm = []
    for cran in crans:
        depth = shoulder - cran
        depths_in_mm.append(round((depth / mm_in_pixel), 3))
    return depths_in_mm


# Calcul des seuils pour la détection de contours
def get_threshold_values(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, th = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    high_val = ret
    low_val = ret * 0.5
    # print(high_thresh_val, low_thresh_val)
    return low_val, high_val


def get_key_measures(img_edges, shoulder_tip_mm, cut_spacing, w, key_edges_right, auto_find_cuts):
    img2 = np.copy(img_edges)
    shoulder_x, shoulder_y, index_shoulder = find_shoulder_position(key_edges_right)  # position x et y du "shoulder"

    # Conversion mm en pixels
    mm_in_pixels = (img2.shape[0] - shoulder_y) / shoulder_tip_mm
    print("mm pixel :", mm_in_pixels)
    print("Shoulder y:", shoulder_y)

    # Recherche du premier cran (présent en dessous du "shoulder")
    # Une marge de 50px est pris en dessous du shoulder, pour ne pas détecter le second cran
    first_notch_x, first_notch_y = get_first_notch(key_edges_right[index_shoulder:index_shoulder + 50])
    print("Position Y premier cran :", first_notch_y)

    if not auto_find_cuts:
        cuts_centers = cuts_positions(mm_in_pixels, first_notch_y, cut_spacing)  # Position des crans en hauteur
    else:
        cuts_centers = get_pos_cuts_auto(key_edges_right)
        if cuts_centers[0] < first_notch_y:
            del cuts_centers[0]
        cuts_centers[0] = first_notch_y

    for cut in cuts_centers:
        cv.line(img2, ((int)(0.25 * w), cut), ((int)(0.75 * w), cut), (255, 0, 0), 1, cv.LINE_AA)
    print("Positions Y des crans :", cuts_centers)

    # position des crans en largeur
    cuts_x_pos, img_res = measure_cuts(img_edges, key_edges_right, cuts_centers, w)

    # Profondeur en mm des crans
    result = get_cuts_depth(cuts_x_pos, shoulder_x, mm_in_pixels)
    return img_res, img2, result, mm_in_pixels


def polygon_detection(img):
    img2 = np.copy(img)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)

    (h, w) = img.shape[0], img.shape[1]
    key_edges_left = []
    key_edges_right = []
    # i --> y et j-->x
    # Left edge
    left_edge_pos = 0
    for i in range(5, h - 5):
        j = 0
        while img[i][j] == 0 and j < w - 1:
            j += 1
        if j == w - 1:
            j = 0
        key_edges_left.append((i, j))
        img2[i][j] = (255, 0, 0)
        if i == 10:
            left_edge_pos = j

    # Right edge
    right_edge_pos = 0
    for i in range(5, h - 5):
        j = w - 1
        while img[i][j] == 0 and j > 0:
            j -= 1
        if j == 0:
            j = w - 1
        key_edges_right.append((i, j))
        img2[i][j] = (0, 255, 0)
        if i == 10:
            right_edge_pos = j

    key_width_px = right_edge_pos - left_edge_pos
    print("key_width_px: " + str(key_width_px))
    return img2, key_edges_left, key_edges_right, key_width_px


# Obtenir la position des crans de façon automatique
def get_pos_cuts_auto(polygon_edges):
    # coordonnées x et y des contours à droite de l'image
    x = [item[1] for item in polygon_edges]
    y = [item[0] for item in polygon_edges]

    # calcul différentiel pour obtenir les pentes et lieux stationnaires
    dydx = diff(x) / diff(y)
    # print("dydx", dydx)

    # Mise en lien dérivée<->coordonnée y
    listDeriv = []
    for i in range(len(dydx)):
        listDeriv.append([dydx[i], y[i]])
    value_deriv = [item[0] for item in listDeriv]
    coord_y = [item[1] for item in listDeriv]

    # Calcul du nombre de points consécutifs négatifs ou nuls (correspond à un cran)
    consecutive_points = []
    for i in range(len(listDeriv)):
        if i == 0:
            if value_deriv[i] == 0:
                consecutive_points.append([1, coord_y[i]])
            else:
                consecutive_points.append([0, coord_y[i]])
        else:
            if value_deriv[i] == 0 and value_deriv[i - 1] == 0:
                consecutive_points.append([consecutive_points[i - 1][0] + 1, coord_y[i]])
            else:
                consecutive_points.append([0, coord_y[i]])

    # Recherche des endroits où il y a plus de 4 points consécutifs de stationnaire
    res = ([item[1] for item in consecutive_points if item[0] > 2])
    print(res)
    cuts = []

    # Calcul de la médiane des points séparés de 1 px.
    temp = []
    for i in range(len(res)):
        if i < len(res) - 1:
            if res[i + 1] - res[i] == 1:
                temp.append(res[i])
            else:
                print("prov :", temp)
                if len(temp) > 0:
                    cuts.append(int(np.median(temp)))
                else:
                    cuts.append(res[i])
                temp.clear()
        else:
            cuts.append(int(np.median(temp)))
    print("Crans trouvés auto :", cuts)
    return cuts


def get_key_code(file, name, depths, key_width):
    code = []

    # Chargement du fichier JSON
    with open(file) as json_file:
        keys_info = json.load(json_file)

        # On parcourt la liste des clés pour trouver la clé correpondante
        for key in keys_info['keys']:
            if key['name'] == name:
                key_data = key['data']
                print("name: " + key['name'])

                # Pour chaque cran mesuré, on cherche le code correspondant dans le JSON
                for i in range(0, len(depths)):
                    depths[i] = abs(key_width - depths[i])  # Distance entre le dos de la clé et le bas du cran
                    print("=== " + str(depths[i]) + "mm ===")

                    # Recherche du code
                    for e in range(0, len(key_data)):
                        if key_data[e]['depth'] - 0.25 <= depths[i] < key_data[e]['depth'] + 0.25:
                            found = key_data[e]['code']
                            code.append(found)
                            print("Cran n°" + str(i) + " : Code trouvé (" + str(found) + ")")
                            break

                        # Si on arrive à la fin de la liste des codes, on indique que le code n'a pas été trouvé
                        elif e == len(key_data) - 1:
                            code.append(-1)
                            print("Cran n°" + str(i) + " : Code non trouvé")
    return code
