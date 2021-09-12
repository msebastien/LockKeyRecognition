from fonctions import *
import matplotlib.pyplot as plt

CUT_SPACING = 3.8  # Espace entre les crans
SHOULDER_TIP_DISTANCE = 26.7  # Espace en mm shoulder - tip
ANGULAR_RES = 0.1  # Résolution de l'angle d'orientation de la clé
AUTO_FIND_CUTS = False  # Utiliser le système de recherche automatique des crans
KEY_NAME = "abus"  # current possible values in json file : "evva", "thirard", "abus", "city"

# Chargement des images
KeyPhoto = cv.imread('dataset/Abus-2/Abus-2-cote-2.jpg')
Pattern = cv.imread('model-3D.png')

# Redimensionnement des images
KeyPhoto = resize_auto(KeyPhoto)
Pattern = resize_auto(Pattern)

# Calcul des seuils pour la détection de bords avec Canny
canny_threshold1, canny_threshold2 = get_threshold_values(KeyPhoto)

# Calcul de l'angle d'orientation de la clé
houghTransformScene, rotation_angle = hough_transform(KeyPhoto, ANGULAR_RES, canny_threshold1, canny_threshold2)
plt.subplot(3, 4, 1), plt.title("Hough Transform"), plt.imshow(houghTransformScene)
print("Angle de la clé : ", rotation_angle)

# Réorientation de l'image
if rotation_angle != 0:
    rotated_key_photo = rotate(KeyPhoto, rotation_angle)
else:  # la clé est déjà à la verticale (0°)
    rotated_key_photo = KeyPhoto
plt.subplot(342), plt.title("Clé retournée"), plt.imshow(rotated_key_photo)

# Segmentation de l'image avec le feature matching
cropped_key, matches_img = feature_match_crop(rotated_key_photo, Pattern, canny_threshold1, canny_threshold2)
plt.subplot(3, 4, 3), plt.title("Matches"), plt.imshow(matches_img)
plt.subplot(3, 4, 4), plt.title("Cle rognee 1"), plt.imshow(cropped_key)

# Recherche de la partie circulaire supérieure de la clé
# pour ne garder que les crans (et réorienter la clé si besoin)
cropped_key = find_circles_and_crop(cropped_key)
cropped_key_width = cropped_key.shape[1]
cropped_key_height = cropped_key.shape[0]
plt.subplot(3, 4, 5), plt.title("Clé rognée 2"), plt.imshow(cropped_key)

# Détection des contours des crans de la clé
canny_threshold1, canny_threshold2 = get_threshold_values(cropped_key)
detected_edges = cv.Canny(cropped_key, canny_threshold1, canny_threshold2)

# Mesure profondeur des crans
img_polygon, key_edges_left, key_edges_right, key_width_px = polygon_detection(detected_edges)
plt.subplot(3, 4, 6), plt.title("Contours"), plt.imshow(img_polygon)

# Si la variance du côté droit est inférieur à celle du côté gauche,
# les crans de la clé sont du mauvais côté
if np.var(key_edges_right) < np.var(key_edges_left):
    detected_edges = cv.flip(detected_edges, 1)
    img_polygon, key_edges_left, key_edges_right, _ = polygon_detection(detected_edges)
    img_res, cuts_img, result, mm_in_pixels = get_key_measures(detected_edges, SHOULDER_TIP_DISTANCE, CUT_SPACING, cropped_key_width, key_edges_right, AUTO_FIND_CUTS)
    print("clé dans le mauvais coté")
else:
    img_res, cuts_img, result, mm_in_pixels = get_key_measures(detected_edges, SHOULDER_TIP_DISTANCE, CUT_SPACING, cropped_key_width, key_edges_right, AUTO_FIND_CUTS)
print("Profondeur des crans : ", result)

key_width_mm = round(key_width_px / mm_in_pixels, 2)  # Largeur de la clé en mm
print("keyWidthMm: " + str(key_width_mm))

# Récupération des codes de la clé avec la mesure des profondeurs des crans
# et les données du fabricant stockées dans un fichier JSON
keyCode = get_key_code("key-codes.json", KEY_NAME, result, key_width_mm)
print("code : " + str(keyCode))


plt.subplot(3, 4, 11), plt.title("Crans"), plt.imshow(img_res)
plt.subplot(3, 4, 10), plt.title("Centre des dents"), plt.imshow(cuts_img)
plt.show()



