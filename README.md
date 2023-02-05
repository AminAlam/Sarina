<div align="center">
<img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/text_on_contour.png" width=400>
<br/>
<h1>Sarina</h1>
<br/>
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="built with Python3" />
<img src="https://img.shields.io/badge/C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="built with C++" />

</div>

----------
Sarina: An ASCII Art generator command line tool to create word clouds from text words based on contours of the given image.

<table border="0">
 <tr>
    <td>The program is dedicated to <a href="https://en.wikipedia.org/wiki/Death_of_Sarina_Esmailzadeh">Sarina Esmailzadeh</a>, a 16-year-old teenager who lost her life during the <a href="https://en.wikipedia.org/wiki/Mahsa_Amini_protests">Mahsa Amini protests</a> protests, as a result of violence inflicted by the IRGC forces. Her memory serves as a reminder of the importance of justice and human rights.

</td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/assets/images/Sarina.png" alt="Sarina Esmailzadeh" width=400 /></td>
 </tr>
</table>

----------
## Table of contents			
   * [Introduction](https://github.com/AminAlam/Sarina#overview)
   * [Installation](https://github.com/AminAlam/Sarina#installation)
   * [Usage](https://github.com/AminAlam/Sarina#usage)
   * [How It Works](https://github.com/AminAlam/Sarina#how-it-works)

----------
## Overview
<p align="justify">
 Sarina is an ASCII art generator written in Python3 and C++. It transforms an input image and a text file containing words and their weights into a unique ASCII art representation. The algorithm behind Sarina is randomized, ensuring that every output is distinct, even for identical inputs.
</p>

----------
## Installation

### PyPI
- Check [Python Packaging User Guide](https://packaging.python.org/installing/)     
- Run `pip install sarina-cli` or `pip3 install sarina-cli`

### Source code
- Clone the repository or download the source code.
- Run `pip3 install -r requirements.txt` or `pip install -r requirements.txt`

## Usage

### Default image and words
```console
Amin@Maximus:Sarina $ sarina
Sarina is generating your word cloud...
100%|███████████████████████████████████████████████████████████| 132/132 [01:09<00:00,  1.89it/s]
Done!
Images are saved in ./results
```
<table border="0">
<tr>
<td> Input Image </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
</tr>
 <tr>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/assets/images/iran_map.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/iran_map/just_text.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/iran_map/just_text_reverse.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/iran_map/text_on_contour.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/iran_map/text_on_contour_reverse.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/iran_map/text_on_main_image.png" width=400 /></td>
 </tr>
</table>

### Custom image and options
```console
Amin@Maximus:Sarina $ sarina  -if 'assets/images/Sarina.png' -ct 100 -ft 20 -tc [255,255,255] -pc -cs
Enter the contour indices to keep (+) or to remove (-) (separated by space): +1 -2 -3 -4
Sarina is generating your word cloud...
100%|███████████████████████████████████████████████████████████| 132/132 [01:06<00:00,  1.98it/s]
Done!
Images are saved in ./results
```
<table border="0">
<tr>
<td> Input Image </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
<td> Generated Output </td>
</tr>
 <tr>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/assets/images/Sarina.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/just_text.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/just_text_reverse.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/text_on_contour.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/text_on_contour_reverse.png" width=400 /></td>
    <td><img src="https://github.com/AminAlam/Sarina/blob/dev/other_files/sarina/text_on_main_image.png" width=400 /></td>
 </tr>
</table>




To learn more about the options, you can use the following command:
```console
Amin@Maximus:Sarina $ sarina --help
Usage: sarina [OPTIONS]

  Sarina: An ASCII Art Generator to create word clouds from text files based
  on image contours

Options:

  -if, --img_file PATH            Path to image file
  
  -tf, --txt_file PATH            Path to text file. Each line of the text
                                  file should be in the following format:
                                  WORD|WEIGHT

  -cs, --contour_selection        Contour selection - if selected, user will
                                  be prompted to enter the contours index. For
                                  example, if you want to keep the contours
                                  with index 0, 3, 4, and remove contours with
                                  index 1, 2, you should enter +0 +3 +4 -1 -2
                                  
  -ct, --contour_treshold INTEGER RANGE
                                  Threshold value to detect the contours.
                                  Sarina uses intensity thresholding to detect
                                  the contours. The higher the value, the more
                                  contours will be detected but the less
                                  accurate the result will be  [default: 100;
                                  0<=x<=255]
                                  
  --max_iter INTEGER RANGE        Maximum number of iterations. Higher number
                                  of iterations will result in more consistent
                                  results with the given texts and weights,
                                  but it will take more time to generate the
                                  result  [default: 1000; 100<=x<=10000]
                                  
  --decay_rate FLOAT RANGE        Decay rate for font scale. Higher decay rate
                                  will result in more consistent results with
                                  the given texts and weights, but it will
                                  take more time to generate the result
                                  [default: 0.9; 0.1<=x<=1.0]
                                  
  -ft, --font_thickness INTEGER   Font thickness. Higher values will make the
                                  texts font thicker. Choose this value based
                                  on the size of the image  [default: 10]
                                  
  --margin INTEGER RANGE          Margin between texts in pixels. Higher
                                  values will result in more space between the
                                  texts  [default: 20; 0<=x<=100]
                                  
  -tc, --text_color TEXT          Text color in RGB format. For example,
                                  [255,0,0] is red. Note to use square
                                  brackets and commas. Also, just enter the
                                  numbers, do not use spaces  [default:
                                  [0,0,0]]
                                  
  -pc, --plot_contour             Plot contour on the generated images. If
                                  selected, the generated images will be
                                  plotted with the detected/selected contours
                                  
  -op, --opacity                  If selected, opacity of each text will be
                                  selected based on its weight  [default:
                                  True]
                                  
  -sp, --save_path PATH           Path to save the generated images. If not
                                  selected, the generated images will be saved
                                  in the same results folder in the directory
                                  as the function is called.
```

