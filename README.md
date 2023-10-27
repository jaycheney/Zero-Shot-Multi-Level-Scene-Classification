# Zero-Shot Multi-Level Scene Classification

Zero-shot multi-level scene classification project modified based [CLIP](https://github.com/openai/CLIP). 
For instance, the scenes are categorized into primary and secondary levels, such as `Indoor - School`.
## RUN
To run the project, execute the `run2layer.py` script. You can specify the `--image_folder` parameter to read images from a designated folder, `{own_image_folder}`, and provide a location to save the resulting scene images using the `--result_folder` parameter.
```bash
python clip2layer.py --image_folder {own_image_folder} --result_folder {own_result_save_folder}
```
## OUTPUT
The output results will be stored at the specified result storage location and will be organized within a multi-level folder structure, as illustrated below:
```
result_folder
|–– Indoor/
|   |–– School
|   |   |–– xxx.jpg
|   |   |–– ....jpg
|   |–– Road/
|   |–– Unlabeled/
|–– Outdoor/
|   |–– School
|   |   |–– xxx.jpg
|   |   |–– ....jpg
|   |–– Road/
|   |–– Unlabeled/
```
Noted: If there are no suitable labels, the scenes will be categorized as `Unlabeled`.
