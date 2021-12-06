import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

src_img = './images/'


#Modifie le chemin d'accès des images et csv
def change_path_root(path):
    global src_img
    src_img = path



#Fonction pour récupération de données parmi les images de la base mendeley
def get_mendeley_cells(size = 1000, stratify_category=True,flatten = True, color=False):
    df_normal_cells = pd.read_csv('../mendeley_cells_redim.csv')
    """Fonction pour récupération de données parmi les images normales
    Paramètres:
        - size : nombre d'images
        - stratify : même nombre d'images par type d'images
    """
    img_cells = []
    data_cells = []
    img_selected = []
    if (stratify_category):
        nb_img_cat = int(size / len(df_normal_cells['category'].value_counts()))
        # pour chaque category on récupère nb_img_cat (stocke index dans un tableau)
        for category in df_normal_cells['category'].value_counts().index:
            img_selected.extend(np.random.choice(df_normal_cells[df_normal_cells['category'] == category].index, nb_img_cat))
    else:
        img_selected = np.random.choice(df_normal_cells.index, size)

    for index in img_selected:
        filename = df_normal_cells.loc[index]['filename']
        data_cells.append((filename, df_normal_cells.loc[index]['category'], 'mendeley'))
        if color:
            img = cv2.imread(src_img + filename)
            img_cells.append(img) # Si couleur, pas d'applatissement possible
        else:
            img = cv2.imread(src_img + filename, cv2.IMREAD_GRAYSCALE)
            if flatten:
                img_cells.append(img.flatten())
            else:
                img_cells.append(img.reshape(256, 256, 1)) # en gris

    # mise en forme pour sortie
    df_data_cells = pd.DataFrame(data_cells, columns=['filename', 'category', 'class'])
    data = np.array(img_cells)

    return df_data_cells, data


# Affichage de plusieurs images sans définir colonnes/lignes => 9 images par lignes
def display_cells(a_filenames):
    """
        Affichage de plusieurs images sans définir colonnes/lignes => 9 images par lignes

    Paramètres :
        a_filenames : tableau des noms de fichiers images
    """
    nb = len(a_filenames)
    nb_lines = nb // 9 + 1
    fig = plt.figure(figsize=(18, nb_lines * 2))
    for i, filename in enumerate(a_filenames):
        fig.add_subplot(nb // 9 + 1, 9, i + 1)
        img = plt.imread(src_img + filename)
        plt.imshow(img)
        plt.axis('off')
