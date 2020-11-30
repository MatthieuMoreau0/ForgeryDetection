# ForgeryDetection

This code implements a JPEG forgery detection algorithm taken from the paper : [JPEG Grid Detection based on the Number of DCT Zeros and its Application to Automatic and Localized Forgery Detection](http://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Nikoukhah_JPEG_Grid_Detection_based_on_the_Number_of_DCT_Zeros_CVPRW_2019_paper.pdf).

The general idea is the following : we compute the best 8x8 jpeg grid that fits our image (by computing the number of zeros in the 2D dct2 of each 8x8 block) and then we find regions of the image that follow a different grid. We apply a statistical test to detect whether these are the result of forgeries.

This project is still a work in progress for the MVA course "Introduction à l'imagerie numérique".

Inline-style:
![alt text](https://github.com/MatthieuMoreau0/ForgeryDetection/blob/master/images/pelican.ppm "Logo Title Text 1")
