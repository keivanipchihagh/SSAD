<!-- Title -->
<div align="center">
  <h1 align="center">Semantic Segmentation for Autonomous Driving (SSAD)</h1>
  <h3 align="center">Thesis Project @ Amirkabir University of Technology</h3>
  <p align="center">
    <a href="https://github.com/keivanipchihagh/SSAD/issues">Report Bug</a>
    ·
    <a href="https://github.com/keivanipchihagh/SSAD/issues">Request Feature</a>
  </p>
  <img src="archive\assets\gifs\ss_video.gif" />
</div>

<!-- Table of Contents -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#testing">Testing</a></li>
      </ul>
    </li>
    <li>
      <a href="#data">Data</a>
      <ul>
        <li><a href="#datasets">Datasets</a></li>
      </ul>
    </li>
    <li>
      <a href="#models">Models</a>
      <ul>
        <li><a href="#unet">U-NET</a></li>
        <li><a href="#gan">GAN</a></li>
        <li><a href="#diffusion">Diffusion</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


## About The Project
This project aims to benchmark light-weight models tailored specifically for the task of Semantic Segmentation in autonomous self-driving vehicles. To fulfil our desired behavior, models must be balanced between both precision, computational efficiency and real-time responsiveness, a crucial requirement for safe and effective autonomous navigation systems.


## Installation
...


## Usage
### Training

### Testing
To test on **CamVid**, download the raw videos from [here](http://vis.cs.ucl.ac.uk/Download/G.Brostow/CamVid/) and move them under `data/datasets/CamVid/videos`. Then, run the following script to extract the frames:
```python
python ./data/tools/camvid_video_process.py
```
> :Warning: **Extracted video are very large! (~25GB)**:


## Data
### Datasets
There are many Semantic Segmentation datasets available for the task of Autonomous Driving. The following datasets are used in this project:

#### • Cityscapes ([Kaggle](https://www.kaggle.com/datasets/xiaose/cityscapes))
The [Cityscapes Dataset](https://www.cityscapes-dataset.com/dataset-overview/) focuses on semantic understanding of urban street scenes. In the following, we give an overview on the design choices that were made to target the dataset’s focus. It involves **5000** fine and **20000** coarse annotated images for **30** semantic classes.

#### • CamVid ([Kaggle](https://www.kaggle.com/datasets/carlolepelaars/camvid))
The [Cambridge-driving Labeled Video Database (CamVid)](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) was one of the first semantically segmented datasets to be released in the self-driving space in late 2007. They used their own image annotation software to annotate **700** images from a video sequence of 10 minutes. The camera was set up on the dashboard of a car, with a similar field of view as that of the driver. There are **32** semantic classes for this dataset.

#### • KITTI ([Kaggle](https://www.kaggle.com/datasets/klemenko/kitti-dataset))
[KITTI](https://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) consists of 200 semantically annotated train as well as 200 test images corresponding to the KITTI Stereo and Flow Benchmark 2015. The data format and metrics are conform with The [Cityscapes Dataset](https://www.cityscapes-dataset.com/dataset-overview/).

#### • DUS
The [Daimler Urban Segmentation Dataset (DUS)](https://paperswithcode.com/dataset/dus) is a dataset for semantic segmentation. It consists of video sequences recorded in urban traffic. The dataset consists of **5000** rectified stereo image pairs with a resolution of 1024x440. 500 frames (every 10th frame of the sequence) come with pixel-level semantic class annotations into 5 classes: ground, building, vehicle, pedestrian, sky.

#### • Mapillary ([Kaggle](https://www.kaggle.com/datasets/kaggleprollc/mapillary-vistas-image-data-collection))
[Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas) is a diverse street-level imagery dataset with pixel‑accurate and instance‑specific human annotations for understanding street scenes around the world. It contains **25000** high-resolution images and **124** semantic object categories collected from 6 continents with a variety of weather, season, time of day, camera, and viewpoints.

## Models
...


## Contributing
Thank you for considering contributing to this project! Contributions are welcome and encouraged.

Please ensure that your pull request adheres to the following guidelines:

- Describe the problem or feature in detail.
- Make sure your code follows the project's coding style and conventions.
- Include relevant tests and ensure they pass.
- Update the documentation to reflect your changes if necessary.

By contributing to this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Thank you for your contributions to making this project better!


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The MIT License is a permissive open-source license that allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software. It is a simple and flexible license that is widely used in the open-source community.

By contributing to this project, you agree that your contributions will be licensed under the MIT License unless explicitly stated otherwise.


## Contact
If you have any questions, suggestions, or feedback, don't hesitate to get in touch!
<p align="center">
    <a href="mailto:keivanipchihagh@gmail.com">Email</a>
    ·
    <a href="https://www.linkedin.com/in/keivanipchihagh">LinkedIn</a>
    ·
    <a href="https://github.com/keivanipchihagh">GitHub</a>
    ·
    <a href="https://stackoverflow.com/users/14733503/keivan-ipchi-hagh">StackOverFlow</a>
</p>


## Acknowledgments
..