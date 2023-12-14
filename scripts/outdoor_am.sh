python ccxml2json.py --input_path data/deyilou/project/outdoor/am/ --input_xml XML/am.xml --output_transforms am.json --is_depth
python downsample.py --input_path data/deyilou/project/outdoor/am/ --input_xml XML/am.xml --downsample  4 --is_depth
python ccxml2json.py --input_path data/deyilou/project/outdoor/am/ --input_xml XML/am.xml --output_transforms am_4.json --downsample 4 --is_depth