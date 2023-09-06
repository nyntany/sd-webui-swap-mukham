import os

## _____________________ GALLERY ____________________

root_sd = os.getcwd()
txt2img_dir = os.path.join(root_sd, "Outputs", "txt2img-images")
img2img_dir = os.path.join(root_sd, "Outputs", "img2img-images")
swap_dir = os.path.join(root_sd, "Outputs", "swap-mukham")
others_dir = os.path.join(root_sd, "Outputs")
txt2img_img_list = [] 
img2img_img_list = [] 
swap_img_list = [] 
others_img_list = [] 



def gallery_txt2img():
    txt2img_img_list = []  # Lista vacía para guardar las imágenes
    for root, dirs, files in os.walk(txt2img_dir):  # Recorrer la carpeta y subcarpetas
        for file in files:  # Recorrer los archivos
            if file.endswith((".jpg", ".png", ".jpeg")): 
                img_path = os.path.join(root, file) 
                txt2img_img_list.append(img_path)
    yield gr.Gallery.update(value=txt2img_img_list)  # Devolver el nuevo valor de la galería
def gallery_img2img():
    img2img_img_list = []  # Lista vacía para guardar las imágenes
    for root, dirs, files in os.walk(img2img_dir):  # Recorrer la carpeta y subcarpetas
        for file in files:  # Recorrer los archivos
            if file.endswith((".jpg", ".png", ".jpeg")): 
                img_path = os.path.join(root, file) 
                img2img_img_list.append(img_path)
    yield gr.Gallery.update(value=img2img_img_list)  # Devolver el nuevo valor de la galería

def gallery_swap():
    swap_img_list = []  # Lista vacía para guardar las imágenes
    for root, dirs, files in os.walk(swap_dir):  # Recorrer la carpeta y subcarpetas
        for file in files:  # Recorrer los archivos
            if file.endswith((".jpg", ".png", ".jpeg")): 
                img_path = os.path.join(root, file) 
                swap_img_list.append(img_path)
    yield gr.Gallery.update(value=swap_img_list)  # Devolver el nuevo valor de la galería

def gallery_others():
    global txt2img_img_list, img2img_img_list, swap_img_list
    others_img_list = []  # Lista vacía para guardar las imágenes
    
    # Recorrer la carpeta y subcarpetas de 'others_dir'
    for root, dirs, files in os.walk(others_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                
                # Verificar si el archivo ya está en alguna de las listas generadas previamente
                if img_path not in txt2img_img_list and img_path not in img2img_img_list and img_path not in swap_img_list:
                    others_img_list.append(img_path)
    
    yield gr.Gallery.update(value=others_img_list)  # Devolver el nuevo valor de la galería


def update_galleries():
    # Crear tres variables vacías para guardar las imágenes de cada tab
    txt2img_img_list = []
    img2img_img_list = []
    swap_img_list = []
    others_img_list = [] 
    # Escanear las imágenes de cada tab_dir y guardarlas en la variable correspondiente
    for root, dirs, files in os.walk(txt2img_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                txt2img_img_list.append(img_path)
    for root, dirs, files in os.walk(img2img_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                img2img_img_list.append(img_path)
    for root, dirs, files in os.walk(swap_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                swap_img_list.append(img_path)
    for root, dirs, files in os.walk(others_dir):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                if img_path not in txt2img_img_list and img_path not in img2img_img_list and img_path not in swap_img_list:
                    others_img_list.append(img_path)
                    
    return txt2img_img_list, img2img_img_list, swap_img_list, others_img_list


