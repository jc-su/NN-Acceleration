import os
import time

import numpy
from PIL import Image
 
def create_image(width = 1920, height = 1080, num_of_images = 1):
    width = int(width)
    height = int(height)
    num_of_images = int(num_of_images)
 
    for n in range(num_of_images):
        filename = 'images/{}.jpg'.format(width)
        rgb_array = numpy.random.rand(height,width,3) * 255
        image = Image.fromarray(rgb_array.astype('uint8')).convert('RGB')
        image.save(filename)
 
def main(args):
    create_image(width = int(args[0]), height = int(args[1]), num_of_images = args[2])
    return 0
 
if __name__ == '__main__':
    import sys 
    status = main(sys.argv[1:])
    sys.exit(status)