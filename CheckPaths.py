# Import necessary libraries
import os
import sys
import shutil
import numpy as np
from time import sleep
# Import local files
from UtilFunctions import get_expected_path_from_conf


def is_files_renamed(conf_dict, folder_path: str):
    """This function will rename all files inside the source_folder and save it as x_y coordinates.
    argument:
    folder_path: the path of folder in string format"""
    # Obtains all contents in mentioned path
    files_list = os.listdir(folder_path)
    folder_contents = [(os.path.join(folder_path, file)) for file in files_list]
    is_file_dir = [os.path.isdir(f) for f in folder_contents]
    image_holder_folder_name = folder_contents[is_file_dir.index(True)]

    # Get all image paths by indexing the folder which returns true
    image_paths = os.listdir(image_holder_folder_name)

    # Counts the number of terms in a filename
    terms_in_name = image_paths[0].split('_')
    terms_in_name[-1] = terms_in_name[-1][:-4]

    sys.stdout.write(f"\nHere is an example filename: {image_paths[0]}\n")
    rename = input("Do you want to rename files? to eg. Xcoor_Ycoor.png [Y or N]:  ").lower()
    if rename not in ["y", "n"]:
        sys.stderr.write(f"Invalid input either input 'y' or 'n'.\n\n")
        sleep(0.1)
        return is_files_renamed(conf_dict, folder_path)

    if rename == 'n':  # If user doesn't want to rename we do nothing and exit out
        validation = is_folder_ready_for_stitching(conf_dict, image_paths)
        image_paths = [os.path.join(folder_path, image_holder_folder_name, _path) for _path in image_paths]
        return validation, image_paths

    # To maintain homogeneity in file names
    sys.stdout.write(f"Enter the axis index in og name between 1-{len(terms_in_name)}: \n")
    x_index = int(input("X-axis: ")) - 1
    y_index = int(input("Y-axis: ")) - 1

    for k in range(len(image_paths)):
        if '_' in image_paths[k]:  # Splits filenames wrt '_'
            current_path = os.path.join(folder_path, image_holder_folder_name, image_paths[k])
            terms_in_name = image_paths[k].split('_')
            terms_in_name[-1] = terms_in_name[-1][:-4]
            new_name = str(terms_in_name[x_index] + "_" + terms_in_name[y_index] + ".png")
            new_path = os.path.join(folder_path, image_holder_folder_name, new_name)
            shutil.move(current_path, new_path)

    renamed_image_paths = os.listdir(folder_contents[is_file_dir.index(True)])
    sys.stdout.write(f"Files renamed. Example file name: {renamed_image_paths[0]}\n")

    validation = is_folder_ready_for_stitching(conf_dict, renamed_image_paths)
    renamed_image_paths = [os.path.join(folder_path, image_holder_folder_name, _path) for _path in renamed_image_paths]
    return validation, renamed_image_paths


def is_folder_ready_for_stitching(conf_dict, image_paths: list):
    """This function will check whether there exists an image for every point mentioned in conf file.
    argument:
    image_paths: list of paths of images."""

    paths_expec = get_expected_path_from_conf(conf_dict)
    list_image_paths = image_paths[:]
    terms_in_path = len(list_image_paths[0].split("_"))
    if terms_in_path >= 3:
        sys.stderr.write(
            f"""Your given file names is not following the x-coor_y-coor.png format you need to replace in order to"
            prepare the folder\n""")
        sys.stderr.write(f"Your Path isn't Ready For Stitching")

    _extra_paths = []
    for _path in list_image_paths:
        if _path not in paths_expec:
            sys.stderr.write(f"Extra path found --> {_path}\n")
            _extra_paths.append(_path)
    if len(_extra_paths) != 0:
        return False

    errors: int = 0
    filename: str

    # Creates filenames with xcor and ycor lists and checks if they exist in list
    for _path in paths_expec:
        try:
            list_image_paths.remove(_path)
        except ValueError:
            errors += 1
            sys.stderr.write(f"Data Ambiguity found. Expected file not found--> {_path}\n")

    if len(list_image_paths) > 0:
        sys.stdout.write(f"List of \n{list_image_paths}\n")
        errors += len(list_image_paths)

    if errors == 0:
        sys.stdout.write(f"Dataset verified you may go ahead\n\n")
        return True
    else:
        return False
