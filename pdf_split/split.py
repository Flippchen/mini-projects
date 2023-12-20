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


def split_pdf_by_color(pdf_path):
    images = convert_from_path(pdf_path)
    reader = PyPDF2.PdfReader(pdf_path)
    writers = {
        'red': PyPDF2.PdfWriter(),
        'blue': PyPDF2.PdfWriter(),
        'purple': PyPDF2.PdfWriter(),
        'all': PyPDF2.PdfWriter(),
        'other': PyPDF2.PdfWriter()
    }

    for page_num, image in enumerate(images):
        page = reader.pages[page_num]
        color = detect_color(image)

        if color in writers:
            writers[color].add_page(page)
        else:
            writers['other'].add_page(page)

        writers['all'].add_page(page)

    for color, writer in writers.items():
        with open(f'{color}_pages.pdf', 'wb') as f:
            writer.write(f)


if __name__ == '__main__':
    split_pdf_by_color('DWH_Folien.pdf')
