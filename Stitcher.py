import os
from StageOperations import *

BATCH_SAVE_IMAGE_FOLDER = "batch_images"
STITCHING_BATCH_SIZE = 10

def stitch_batches(epoch_count, filepaths_list):
    """Should be called from inside a thread
    argument:
    epoch_count: current epoch count
    filepaths_list: list of all image paths"""

    sys.stdout.write(f"Thread for epoch:{epoch_count} started.\n")
    filepaths_list_len = len(filepaths_list)
    upper_limit = min((epoch_count + 1) * STITCHING_BATCH_SIZE, filepaths_list_len)
    if upper_limit == filepaths_list_len:
        upper_limit += 1
        
    path_list = filepaths_list[epoch_count * STITCHING_BATCH_SIZE: ]
    images_list = [cv.resize(cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB), None,
                             fx=BATCH_STITCHING_RESIZE_RATIO, fy=BATCH_STITCHING_RESIZE_RATIO) for path in path_list]

    Stages = StageOperations()
    row_epoch_images = {}
    while True:     # Runs loop for stitching and saving rows if error occurs then restarts the thread
        try:
            row_epoch_images[epoch_count] = Stages.stitch_images(images_list, thread_no=epoch_count)
            row_epoch_images[epoch_count] = np.array(row_epoch_images[epoch_count], dtype=np.int16)
            break
        except Exception as e:
            sys.stderr.write(f"Error {e} while stitching thread:{epoch_count}, restarting stitching...\n")

    save_image(save_file_name=f"{BATCH_SAVE_IMAGE_FOLDER}/{epoch_count}", image=row_epoch_images[epoch_count])


def stitch_panorama(batch_images_path):
    """Stitches batch images to obtain final panoramic image
    arguments:
    batch_images_path: folder path where batch images have been saved"""

    batch_images = []
    for path in sorted(os.listdir(batch_images_path)):
        _image = cv.cvtColor(cv.imread(os.path.join(batch_images_path, path)), cv.COLOR_BGR2RGB)
        batch_images.append(_image)

    sys.stdout.write(f"Batch Image Stitching Starts here...\n")
    Stages = StageOperations()
    while True:     # Runs loop for stitching and saving rows if error occurs then restarts the thread
        try:
            final_panorama = None
            if len(batch_images) != 1:
                final_panorama = Stages.stitch_images(batch_images, nfeatures=np.shape(batch_images[0])[1], thread_no=None)
            else:
                final_panorama = batch_images[0]
            end_time = time.time()

            #plot_image(final_panorama)
            save_image(save_file_name=f"{BATCH_SAVE_IMAGE_FOLDER}/panorama", image=final_panorama)
            return end_time, final_panorama
        except Exception as e:
            sys.stderr.write(f"{e}")
