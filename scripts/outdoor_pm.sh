python ccxml2json.py --input_path data/deyilou/project/outdoor/pm/ --input_xml XML/pm.xml --output_transforms pm.json --is_depth
python ccxml2json.py --input_path data/deyilou/project/outdoor/pm/ --input_xml XML/pm.xml --output_transforms pm_4.json --downsample 4 --is_depth
python downsample.py --input_path data/deyilou/project/outdoor/pm/ --input_xml XML/pm.xml --downsample  4 --is_depth