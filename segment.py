import numpy as np
import cv2
import argparse
from utils import *
from fuzzyseg import *
import time

parser = argparse.ArgumentParser(
    description='Hierarchical segmentation based o FuzzySegmentation',
)

parser.add_argument('-i', '--img', action="store",
                    dest='image', default='images/snb.jpg',
                    # dest='image', default='images/snb.jpg',
                    help='Path to the image to be segmented')
parser.add_argument('-o', '--output', action="store",
                    dest='output', default='output.png',
                    help='Path to save the segmented image, default = output.png')
parser.add_argument('-s', '--seeds', action="store",
                    dest='seeds', default='seeds.in',
                    help='Path to the file containing the seed with the format: x y class, default = seeds.in')
parser.add_argument('-n', '--num-aff', action="store",
                    dest='num_aff', default=1000,
                    help='Number of discretized affinities, default = 1000')
parser.add_argument('-a', '--avg-size', action="store",
                    dest='avg_size', default=2,
                    help='Average size used on affinity calc, default = 2')

args = parser.parse_args()

image = cv2.imread(args.image, 0)
output = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8)

fs = FuzzySegmentation(image, num_aff=args.num_aff, avg_size=args.avg_size)

seeds = open(args.seeds, 'r')

for seed in seeds:
    x, y, c = seed.split()
    fs.add_seed(int(x), int(y), int(c))

seeds.close()

start_time = time.time()
fs.fuzzy_seg()
elapsed_time = time.time() - start_time
print('Segmentation time: ', elapsed_time)
# fs.print_conts()
flag = True
for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        r = 255*((fs.final_class[i, j]&1) != 0)
        g = 255*((fs.final_class[i, j]&2) != 0)
        b = 255*((fs.final_class[i, j]&4) != 0)
        output[i, j][0] = min(r*(fs.get_normalized_affinity(i, j)), 255)
        output[i, j][1] = min(g*(fs.get_normalized_affinity(i, j)), 255)
        output[i, j][2] = min(b*(fs.get_normalized_affinity(i, j)), 255)

cv2.imshow('Segmentation', output)
cv2.imwrite(args.output, output)
cv2.waitKey(0)
cv2.destroyAllWindows()
# show_generated_seeds(image)

