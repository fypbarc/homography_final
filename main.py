# Import program files
import torch
from Stitcher import *
from CheckPaths import *
from SelectImagesFolder import *
from threading import Thread
from Denoising import DenoiseImage
from Convert3D import create3D

import cv2 as cv
from time import time


# Create folder to save batch_images
if not os.path.exists(BATCH_SAVE_IMAGE_FOLDER):
    os.mkdir(BATCH_SAVE_IMAGE_FOLDER)
else:
    for file_path in os.listdir(BATCH_SAVE_IMAGE_FOLDER):
        os.remove(os.path.join(BATCH_SAVE_IMAGE_FOLDER, file_path))

# CONSTANTS
RESIZE_RATIO = 1  # Images will be resized to RESIZED_RATIO
FINAL_RESIZE_RATIO = 1
DENOISED_IMAGE_FILENAME = "denoised_filtered"
try_use_gpu = torch.cuda.is_available()


sys.stdout.write(f"Stitching to generate 3D-panorama is a 10-stage process.\n\n")


# Get folder path and validate it
conf_extractor = GetConfData()
CONF_DICT = conf_extractor.conf_dict
status, filepaths_list = is_files_renamed(CONF_DICT, conf_extractor.folder)
if not status:
    sys.stderr.write(f"\nYour image folder isn't what the stitcher requires. Exiting...")
    sys.exit()
sys.stdout.write(f"STAGE 1: Images Acquisition completed successfully.\n")

# Initialize timer
start = time()

# Obtains the image data from paths and resizes them
image_data_list = [cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) for filename in filepaths_list]
resize_images(image_data_list, size_factor=RESIZE_RATIO)

# We start with making batches of input images
ROW_PANO = {}
x_range, y_range = get_expected_path_from_conf(CONF_DICT, raw_range=True)
total_images = len(x_range) * len(y_range)
if STITCHING_BATCH_SIZE > total_images / 2:
    STITCHING_BATCH_SIZE = total_images

num_epochs = math.ceil(total_images / STITCHING_BATCH_SIZE) + 1

# Starting threads to stitch batches
row_stitching_threads = {}
for epoch in range(num_epochs):
    if (epoch * STITCHING_BATCH_SIZE) >= total_images:
        break
    row_stitching_threads[epoch] = Thread(target=stitch_batches, args=(epoch, filepaths_list))
    row_stitching_threads[epoch].start()
for thread in list(row_stitching_threads.values()):
    thread.join()

end_time, final_panorama_image = stitch_panorama(BATCH_SAVE_IMAGE_FOLDER)
sys.stdout.write(f"Total time required for stitching: {end_time - start}")


# Obtain denoised image from the panorama
cust_filter = DenoiseImage()
# We need to convert image to grayscale before denoising
final_panorama_gray = cv.cvtColor(final_panorama_image, cv.COLOR_RGB2GRAY)
filtered_image = cust_filter.denoised_image(final_panorama_gray)
save_filtered_path = f"{BATCH_SAVE_IMAGE_FOLDER}/{DENOISED_IMAGE_FILENAME}"
save_image(filtered_image, save_file_name=save_filtered_path)
sys.stdout.write(f"STAGE-10: Denoising completed.\n")

# Obtain FFT of filtered image
fft_img = cust_filter.get_fft(filtered_image)

if np.size(final_panorama_gray) > np.size(filtered_image):
    final_panorama_gray = cv.resize(final_panorama_gray, np.shape(filtered_image)[::-1])
else:
    filtered_image = cv.resize(filtered_image, np.shape(final_panorama_gray)[::-1])
    
value = cust_filter.psnr(final_panorama_gray, filtered_image)
sys.stdout.write(f"The PSNR value of denoised image is: {value}\n")

# Convert denoised image to 3D
sys.stdout.write(f"STAGE_10: Obtaining 3D depth map of panorama image and saving in {BATCH_SAVE_IMAGE_FOLDER}...\n")
create3D(save_filtered_path + ".png")
sys.stdout.write("Stitching completed.")
