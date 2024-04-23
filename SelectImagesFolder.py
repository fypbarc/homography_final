# Import necessary libraries
import os
import sys
import time
from tkinter import filedialog, messagebox

SCAN_FILE = "scan_details.conf"

class GetConfData:
    """Obtains the config file data present in chosen folder and gets zoom level, X and Y-axes steps.
    The above parameters are then used for Stitching."""

    def __init__(self):
        folder_path = get_folder_path()
        self.folder = folder_path
        self.folder_contents = os.listdir(self.folder)  # Get folder contents
        self.path_is_valid = self.check_path_validity(self.folder_contents)  # Check if scan_detail.conf exists or not
        self.conf_dict = None

        if not self.path_is_valid:  # If .conf doesn't exist, throws a warning and makes user choose again
            warning_message = """Select the folder having the following items:
            1) image_folder (having all scanned images)
            2) scan.conf file with .conf extension"""
            messagebox.showwarning(title="Warning", message=warning_message)
            _get_conf_data = GetConfData()
            self.folder = _get_conf_data.folder
            self.folder_contents = _get_conf_data.folder_contents
            self.path_is_valid = _get_conf_data.path_is_valid
            self.conf_dict = _get_conf_data.conf_dict
        else:  # If scan_detail.conf exists alongside with images scan_details are extracted
            self.conf_dict = self.return_stitching_dict()

    @staticmethod
    def check_path_validity(contents: list):
        """Checks if there exists a scan.conf file in the path and a folder containing image files"""
        config_files = list()
        for file in contents:
            if file.endswith(".conf"):
                config_files.append(file)

        if len(config_files) != 1:
            if len(config_files) == 0:
                return False
            sys.stdout.write("""Here is a list of available scan.config files available in the given folder.
                             Please input the index number of the correct file\n""")
            for index, element in enumerate(config_files):
                sys.stdout.write(f"{index}: {element}\n")
            sys.stdout.write("\n")

            while True:
                index = input("Enter the index in numeric format: ")
                try:
                    global SCAN_FILE
                    SCAN_FILE = str(config_files[int(index)])
                    return True
                except Exception:
                    sys.stderr.write(f"Invalid input please choose from displayed file indexes.\n")
                    time.sleep(0.1)
        return True

    def return_stitching_dict(self):
        """ Reads through the scan_details.conf and returns a dict of start and end coordinates of scan, its zoom value
        and step_size in each axis and returns a dictionary
        data_dict format is as follows
        key: value --> 'Key': [start_point, end_point, step_size]"""
        scan_path = [path for path in self.folder_contents if SCAN_FILE in path][0]
        scan_path = os.path.join(self.folder, scan_path)
        # Read and dictionary the contents of conf file
        data_dict = dict()
        with open(scan_path, 'r') as f:
            # Opens the conf file and obtains X, Y, Zoom ranges
            conf_content = f.readlines()
            conf_content = [param.replace("\n", "") for param in conf_content if
                            "#" not in param and "FOCUS" not in param]

        data_dict["X_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "X_RANGE" in ent][0].split("|")
        data_dict["Y_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "Y_RANGE" in ent][0].split("|")
        data_dict["ZOOM_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "ZOOM_RANGE" in ent][0].split("|")

        return data_dict


def get_folder_path():
    """Opens a window to let user select the folder containing images and scan.conf file.
    Note: Select the folder having 'image_folder' and scan_detail.conf file"""

    # Returns path of selected folder
    selected_folder_path = filedialog.askdirectory(title="Select Parent Folder containing Images")

    if selected_folder_path == "":
        # If user cancels the select folder path the code ends
        warning_message = "Operation quit by user. Terminating..."
        messagebox.showwarning(title="Warning", message=warning_message)
        sys.exit()

    return selected_folder_path
