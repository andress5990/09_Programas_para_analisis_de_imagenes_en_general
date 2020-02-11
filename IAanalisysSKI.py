# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open('Color.jpg')
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()