from PIL import Image

img = Image.open("C:/Users/ADMIN/MySGNet/data/kaist-cvpr15-preview/train/gt/._I00027.jpg")
print("Format:", img.format)
print("Size:", img.size)
print("Mode:", img.mode)
