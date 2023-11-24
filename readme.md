# Preprocessing NeRF Json

## read_xml.py

from cc xml to a dict

```python
def read_xml(input_xml):
	...
	c2w = np.hstack([rot_mat[:3,:4], np.array([[height, width, focal_px]]).T])
	results[id] = {'path': path, 'rot_mat': c2w.tolist()}
```

## downsample.py

downsample image N* (output format: jpg)

```sh
python scripts/downsample.py --input_path wukang --input_xml wukang_9_qingxie.xml --downsample 4
```



## ccxml2json.py

convert cc xml to json for nerfstudio

```sh
python ccxml2json.py --input_path . --input_xml shizi.xml --output_transforms shizi.json
```



## utils.py

orient_and_center_poses function: "pca", "up", "vertical", "none"



## visualize.py

Visualize camera pose & point clouds

camera: Opencv coordinate



