import os
from PIL import Image
import rawpy
import imageio
from tqdm import tqdm

def convert_dng_to_jpg(dng_path, jpg_path):
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess()

    img = Image.fromarray(rgb)
    img.save(jpg_path)

folder_path = 'data/huanghelu_4parts/images/1Q/DJI/1Q_DJI_009/'

for i in tqdm(os.listdir(folder_path), desc='dng 2 jpg'):
    if '.DNG' not in i:
        continue
    # import pdb;pdb.set_trace()
    filename = i.split('.DNG')[0]+'.JPG'
    convert_dng_to_jpg(os.path.join(folder_path, i),os.path.join(folder_path, filename))