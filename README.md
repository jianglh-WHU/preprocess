# Preprocessing Data

### **The structure of the data:**

 ```
   ├── images 
   │   ├── img_1
   │   └── img_2
   ├── images_5 # downsample 5*
   │   ├── img_1
   │   └── img_2
  	├── point_cloud	
   └── json
 ```

### **Scripts:**

+ **Transform cc (context capture) & rc (reality capture) data into json**
  + ccxml2json.json / realitycaputre2json / colmap2json / inphoprj2json 
  + Note: have to adjust the image path based on each cc_xml or rc_csv

```python
python ccxml2json.py --input_path . --input_xml shizi.xml --output_transforms shizi.json

python realitycapture2json.py --input_path data/deyilou/dyl --input_csv data/open_data/am_CSV/zaoshang-dibaoguang.csv --downsample 1 --images_path am/-1 --file_name transforms_am_-1.json
```

+ **Downsample images**
  + Copy the all images and downsample the images to a source path (`cp_imgs.sh` ➡️ `ds_all.py`)
  + downsample the images according to the xml or json (don't need to change the images)

```
python downsample.py --input_path wukang --input_xml wukang_9_qingxie.xml --downsample 4
```

+ **Merge lases to ply and downsample the final point cloud (situation: pc from cc)**

  + ds_multi_las.py

+ **visualize the poses and point cloud**

  + poses.py

  + Note: have to change the ply_path and pos_path mutually


