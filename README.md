# Forgery Detection in JPEG images

This code implements a JPEG forgery detection algorithm taken from the paper : [JPEG Grid Detection based on the Number of DCT Zeros and its Application to Automatic and Localized Forgery Detection](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Nikoukhah_JPEG_Grid_Detection_based_on_the_Number_of_DCT_Zeros_CVPRW_2019_paper.pdf).

The general idea is the following : we compute the best 8x8 jpeg grid that fits our image (by computing the number of zeros in the 2D dct2 of each 8x8 block) and then we find regions of the image that follow a different grid. We apply a statistical test to detect whether these are the result of forgeries.

This project was carried out for the master MVA course "Introduction à l'imagerie numérique".

## Installation
- clone repo
- cd ForgeryDetection
- install dependencies : `pip install numpy opencv-python pip install scipy tqdm matplotlib`

## Usage
`python main.py --path <path_to_file>`

The analysis may take a couple minutes to perform depending on the resolution of the image (this implementation could easily benefit from multiprocessing for quicker computation). 
- First, the algorithm computes the 8x8 JPEG grid `G` that fits the image the best and applies a statistical test to detect whether this grid is significant
- Then, for each region where the best grid is different from `G` we apply a statistical test to detect whether it is the result of a forgery or simply a random occurence.
- Finally, we plot 3 images :
  - The original image
  - The vote map for each pixel, where we apply a unique random color for each grid
  - The forgery mask, where the detected forged regions are displayed in white

## Example
You can run the following command : 
`python main.py --path "images/tampered2.ppm"`

This outputs the following figures : 
|Image | Vote map | Forgery |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![Image](https://github.com/MatthieuMoreau0/ForgeryDetection/blob/master/example/usage/image.png "Image") | ![Vote map](https://github.com/MatthieuMoreau0/ForgeryDetection/blob/master/example/usage/votes.png "Vote map")  | ![Forgery](https://github.com/MatthieuMoreau0/ForgeryDetection/blob/master/example/usage/forgery.png "Forgery")|


In the top right corner the best fitting grid is different from the main grid of the image. The statistical test reveals this cannot be explained by randomness and thus this is the result of a forgery.


