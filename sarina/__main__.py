import cv2 
import numpy as np
import sys
from tqdm import tqdm
import random 
import click
import os
import pathlib
from PIL import Image, ImageDraw, ImageFont
file_dir = pathlib.Path(__file__).parent.absolute()

from py2cpp import CppBackend
from parsers import parse_words



def get_text_dimensions(text_string, font):
    # https://stackoverflow.com/a/46220683/9263761
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)

@click.command(help='Sarina: An ASCII Art Generator to create word clouds from text files based on image contours')
@click.option('--img_file', '-if', default=os.path.join(file_dir, 'assets', 'images', 'iran_map.png'), type=click.Path(exists=True), help='Path to image file')
@click.option('--txt_file', '-tf', default=os.path.join(file_dir, 'assets', 'texts', 'heroes_of_iran.txt'), type=click.Path(exists=True), help='Path to text file. Each line of the text file should be in the following format: WORD|WEIGHT')
@click.option('--font_file', '-ff', default=os.path.join(file_dir, 'assets', 'fonts', 'ARIAL.TTF'), type=click.Path(exists=True), help='Path to font file')
@click.option('--contour_selection', '-cs', is_flag=True, default=False, show_default=True, help='Contour selection - if selected, user will be prompted to enter the contours index. For example, if you want to keep the contours with index 0, 3, 4, and remove contours with index 1, 2, you should enter +0 +3 +4 -1 -2')
@click.option('--contour_treshold', '-ct', default=100, type=click.IntRange(0, 255), show_default=True, help='Threshold value to detect the contours. Sarina uses intensity thresholding to detect the contours. The higher the value, the more contours will be detected but the less accurate the result will be')
@click.option('--max_iter', default=1000, type=click.IntRange(100, 10000), show_default=True, help='Maximum number of iterations. Higher number of iterations will result in more consistent results with the given texts and weights, but it will take more time to generate the result')
@click.option('--decay_rate', default=0.9, type=click.FloatRange(0.1, 1.0), show_default=True, help='Decay rate for font scale. Higher decay rate will result in more consistent results with the given texts and weights, but it will take more time to generate the result')
@click.option('--font_thickness', '-ft', default=10, show_default=True, help='Font thickness. Higher values will make the texts font thicker. Choose this value based on the size of the image')
@click.option('--margin', default=20, type=click.IntRange(0, 100), show_default=True, help='Margin between texts in pixels. Higher values will result in more space between the texts')
@click.option('--text_color', '-tc', default='[0,0,0]', type=click.STRING, show_default=True, help='Text color in RGB format. For example, [255,0,0] is red. Note to use square brackets and commas. Also, just enter the numbers, do not use spaces')
@click.option('--plot_contour', '-pc', is_flag=True, default=False, show_default=True, help='Plot contour on the generated images. If selected, the generated images will be plotted with the detected/selected contours')
@click.option('--opacity', '-op', is_flag=True, default=True, show_default=True, help='If selected, opacity of each text will be selected based on its weight')
@click.option('--save_path', '-sp', default = None, type=click.Path(exists=True), help='Path to save the generated images. If not selected, the generated images will be saved in the same results folder in the directory as the function is called.')
def main(img_file, txt_file, font_file, contour_selection, contour_treshold, max_iter, decay_rate, font_thickness, margin, text_color, plot_contour, opacity, save_path):
    rgb_img = cv2.imread(img_file)
    main_img = rgb_img.copy()
    text, weights = parse_words(txt_file)

    w_img, h_img, _ = rgb_img.shape

    color_original = [int(c) for c in text_color[1:-1].split(',')]
    color_original = [color_original[2], color_original[1], color_original[0]]
    color_original = tuple(color_original)

    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img, contour_treshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    num_contours = len(contours)

    font_size = min([w_img/500, h_img/500, 5])
    font_thickness_contour = 1+int(min([w_img/500, h_img/500, 5]))

    if not contour_selection:
        keep_contours = [0]
        if len(contours)>1 and abs(cv2.contourArea(contours[0]) - w_img*h_img) < 0.01*w_img*h_img:
            keep_contours = [1]
        remove_contours = []
        selected_contours = keep_contours
    else:
        legend_h = 0
        for i in range(np.min([num_contours, 5])):
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            cv2.drawContours(rgb_img, contours, i, color, 3)
            (w, h), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness_contour)
            # put the text on a rectangle as legend on the image
            cv2.rectangle(rgb_img, (0, legend_h), (w, legend_h+h), color, -1)
            cv2.putText(rgb_img, str(i), (0, legend_h+h), cv2.FONT_HERSHEY_SIMPLEX, font_size, [abs(255-c) for c in color], font_thickness_contour)
            legend_h = legend_h + h
        cv2.imshow('contours', rgb_img)
        cv2.waitKey(1)
        selected_contours = input('Enter the contour indices to keep (+) or to remove (-) (separated by space): ')
        cv2.destroyAllWindows()
        selected_contours = [i for i in selected_contours.split(' ')]
        keep_contours = [int(i[1:]) for i in selected_contours if i[0]=='+']
        remove_contours = [int(i[1:]) for i in selected_contours if i[0]=='-']
        selected_contours = [int(i[1:]) for i in selected_contours if i[0]=='-' or i[0]=='+']
        
    print('Sarina is generating your word cloud...')

    contour = np.concatenate(([contours[i] for i in keep_contours]), axis=0)
    
    just_text_img = (rgb_img*0+1)*255
    fontScale = 1
    max_iter_original = max_iter
    thickness_original = font_thickness
    margin = np.min([margin, w_img//100, h_img//100])


    txt_info = []

    S_contour = cv2.contourArea(contour)

    max_weight = max(weights)
    min_weight = min(weights)

    weights = [30 + 170 * (weight - min_weight) / (max_weight - min_weight) for weight in weights]
    max_weight = max(weights)
    min_weight = min(weights)

    S_total_txt = 0
    for indx, txt in enumerate(text):
        weight = weights[indx]
        # thickness = 1+int(thickness_original*(max_weight/weight)**0.5)
        # (w, h), _ = cv2.getTextSize(txt, font, int(1+fontScale*weights[indx]), thickness)
        font = ImageFont.truetype(font_file, int(1+fontScale*weights[indx]))
        (w, h) = get_text_dimensions(txt, font)
        S_total_txt += w * h
        txt_info.append([w, h, indx, txt, weights[indx]])
    print(S_contour, S_total_txt)
    txt_info.sort(key=lambda x: x[-1], reverse=True)
    C_ratio = S_contour / S_total_txt
    
    resize_factor = 1/C_ratio**0.5
    contour = contour * resize_factor
    contour = contour.astype(np.int32)
    just_text_img = cv2.resize(just_text_img, (0, 0), fx=resize_factor, fy=resize_factor)
    main_img = cv2.resize(main_img, (0, 0), fx=resize_factor, fy=resize_factor)

    min_x = np.min(contour[:, 0, 0])
    min_y = np.min(contour[:, 0, 1])
    max_x = np.max(contour[:, 0, 0])
    max_y = np.max(contour[:, 0, 1])


    filled_area = just_text_img*0
    filled_area = cv2.cvtColor(filled_area, cv2.COLOR_BGR2GRAY)
    for contour_indx in keep_contours:
        contours[contour_indx] = contours[contour_indx] * resize_factor
        contours[contour_indx] = contours[contour_indx].astype(np.int32)
        cv2.drawContours(filled_area, contours, contour_indx, (255, 255, 255), -1)
    for contour_indx in remove_contours:
        contours[contour_indx] = contours[contour_indx] * resize_factor
        contours[contour_indx] = contours[contour_indx].astype(np.int32)
        cv2.drawContours(filled_area, contours, contour_indx, (0, 0, 0), -1)

    filled_area = (filled_area / 255 - 1)*-1
    filled_area = filled_area*255
    text_on_contour_img = filled_area.copy()

    filled_area = np.array(filled_area, dtype=np.uint8)

    max_weight = np.max(weights)
    cpp_backend = CppBackend(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)
    
    for txt in tqdm(txt_info):
        w, h, indx, txt, weight = txt
        max_iter = int(max_iter_original*(max_weight/weight)**0.5)
        thickness = 1+int(thickness_original/(max_weight/weight)**0.5)
        fontScale_tmp = fontScale
        while True:
            x, y, status = cpp_backend.get_fontScale(filled_area, w, h, max_iter, margin)
            print(w, h, txt, status, x, y)
            if status:
                font = ImageFont.truetype(font_file, int(fontScale_tmp*weight))
                (w, h) = get_text_dimensions(txt, font)
                # (w, h), _ = cv2.getTextSize(txt, font, fontScale_tmp*weight, thickness)
                break
            else:
                fontScale_tmp = fontScale_tmp * decay_rate
                font = ImageFont.truetype(font_file, int(fontScale_tmp*weight))
                (w, h) = get_text_dimensions(txt, font)
                # (w, h), _ = cv2.getTextSize(txt, font, fontScale_tmp*weight, thickness)
        if opacity:
            alpha = 0.5*(1+weight/max_weight)
        else:
            alpha = 1

        color = (0, 0, 0)
        overlay = just_text_img.copy()

        font = ImageFont.truetype(font_file, int(fontScale_tmp*weight))

        img_draw_interface = ImageDraw.Draw(Image.fromarray(overlay))
        img_draw_interface.text((x, y+h), txt, font=font, fill=color)

        # cv2.putText(overlay, txt, (x, y+h), font, fontScale_tmp*weight, color, thickness, cv2.LINE_AA)
        just_text_img = cv2.addWeighted(overlay, alpha, just_text_img, 1 - alpha, 0)

        color = int(alpha*255)
        img_draw_interface = ImageDraw.Draw(Image.fromarray(text_on_contour_img))
        img_draw_interface.text((x, y+h), txt, font=font, fill=color)

        # cv2.putText(text_on_contour_img, txt, (x, y+h), font, fontScale_tmp*weight, color, thickness, cv2.LINE_AA)
        img_draw_interface = ImageDraw.Draw(Image.fromarray(main_img))
        img_draw_interface.text((x, y+h), txt, font=font, fill=color_original)
        filled_area = Image.fromarray(filled_area)
        img_draw_interface = ImageDraw.Draw(filled_area)
        img_draw_interface.text((x, y+h), txt, font=font, fill=255)
        filled_area = np.array(filled_area, dtype=np.uint8)

        cv2.imshow('filled_area', filled_area)
        cv2.waitKey(0)

        # cv2.putText(main_img, txt, (x, y+h), font, fontScale_tmp*weight, color_original, thickness, cv2.LINE_AA)


        # cv2.putText(filled_area, txt, (x, y+h), font, fontScale_tmp*weight, [255, 255, 255], thickness, cv2.LINE_AA)

    alpha = 0.3
    overlay = just_text_img.copy()
    if plot_contour:
        for contour_indx in selected_contours:
            cv2.drawContours(main_img, contours, contour_indx, color_original, 10)
            cv2.drawContours(overlay, contours, contour_indx, (0, 0, 0), 10)
        just_text_img = cv2.addWeighted(overlay, alpha, just_text_img, 1 - alpha, 0)
            


    main_img = cv2.resize(main_img, (0, 0), fx=1/resize_factor, fy=1/resize_factor)
    if save_path is None:
        home_dir = pathlib.Path.home()
        save_path = os.path.join(home_dir, 'sarina_results')
        if os.path.isdir(save_path) is False:
            os.mkdir(save_path)

    just_text_img_reverse = (just_text_img*-1+255).astype(np.uint8)
    text_on_contour_img_reverse = (text_on_contour_img*-1+255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path, 'just_text.png'), just_text_img)
    cv2.imwrite(os.path.join(save_path, 'text_on_contour.png'), text_on_contour_img)
    cv2.imwrite(os.path.join(save_path, 'text_on_main_image.png'), main_img)
    cv2.imwrite(os.path.join(save_path, 'just_text_reverse.png'), just_text_img_reverse)
    cv2.imwrite(os.path.join(save_path, 'text_on_contour_reverse.png'), text_on_contour_img_reverse)

    print('Done!')
    print(f'Images are saved in {save_path}')



if __name__ == "__main__":
    main()
    

    