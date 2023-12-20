import PyPDF2
from pdf2image import convert_from_path


def detect_color(img):
    img.save('temp.png')
    top_right_pixel = img.getpixel((img.width - 10, 10))
    print(top_right_pixel)
    if top_right_pixel == (255, 54, 54):  # Red
        return 'red'
    elif top_right_pixel == (20, 71, 255):  # Blue
        return 'blue'
    elif top_right_pixel == (153, 51, 255):  # Purple
        return 'purple'
    else:
        return 'other'



