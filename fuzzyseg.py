import numpy as np
import time


class Seed(object):

    def __init__(self, x, y, class_seed):
        self.x = x
        self.y = y
        self.class_seed = class_seed


class FuzzyFragment(object):

    def __init__(self):
        self.x = -1
        self.y = -1
        self.class_frag = -1


class FuzzyFragmentList(object):

    def __init__(self):
        self.buf = [FuzzyFragment()]
        self.idx = 0

    def get(self, i):
        if(i >= len(self.buf)):
            for i in range(i*2):
                self.buf.append(FuzzyFragment())
        return self.buf[i]

    def set(self, id, v):
        self.buf[id] = v


class FuzzySegmentation(object):

    def __init__(self, image, num_aff=1000, avg_size=2):
        self.num_aff = int(num_aff)
        self.avg_size = int(avg_size)

        self.fragment_set = [FuzzyFragmentList() for i in range(self.num_aff)] # list of fragment lists for each discretized affinity

        self.seed_set = [] # structure to keep the chosen seeds
        self.final_class = np.zeros((image.shape[0], image.shape[1]), dtype=np.int) # final class of each pixel
        self.greatestk = np.zeros((image.shape[0], image.shape[1])) # affinity matrix

        self.image = image
        self.width = image.shape[1]
        self.height = image.shape[0]


    def media(self, x, y, size):
        sum = 0
        qtd = 0
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if(self.is_valid(x+j, y+i)):
                    sum += self.image[y+i, x+j]
                    qtd += 1
        return sum/qtd

    def add_fragment(self, k, x, y, class_frag):
        self.fragment_set[k].get(self.fragment_set[k].idx).x = x
        self.fragment_set[k].get(self.fragment_set[k].idx).y = y
        self.fragment_set[k].get(self.fragment_set[k].idx).class_frag = class_frag
        self.fragment_set[k].idx += 1
        self.greatestk[y, x] = k


    def is_valid(self, x, y):
        return x < self.width and y < self.height and x >= 0 and y >= 0

    def aff_distance_factor(self, dist):
        aff_df = 980
        return 1 * (1.0 - dist / aff_df)

    def get_affinity(self, x1, y1, x2, y2):
        p1 = self.media(x1, y1, self.avg_size)
        p2 = self.media(x2, y2, self.avg_size)
        disp = float(p2 - p1)

        af = (self.num_aff-1)*((1.0 - np.fabs(disp)/255.0))
        return max(0, min(self.num_aff - 1, int(af)))

    def get_normalized_affinity(self, x, y):
        return (float(self.greatestk[y, x]) / (self.num_aff-1))

    def reset(self):
        self.fragment_set = [FuzzyFragmentList() for i in range(self.num_aff)]  # list of fragment lists for each discretized affinity
        self.final_class = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.int)  # final class of each pixel
        self.greatestk = np.zeros((self.image.shape[0], self.image.shape[1]))  # affinity matrix

    def add_fragment_from_seed(self):
        for i in range(len(self.seed_set)):
            self.add_fragment(self.num_aff-1, self.seed_set[i].x, self.seed_set[i].y, self.seed_set[i].class_seed)

    def add_seed(self, x, y, seed_class):
        self.seed_set.append(Seed(x, y, seed_class))

    def proces_fragment(self, k, x, y, dx, dy, f):
        if(self.is_valid(x + dx, y + dy) and self.final_class[y + dy, x + dx] == 0):
            affinity = self.get_affinity(x, y, x+dx, y+dy)
            a = min(k, affinity)
            if(a > self.greatestk[y+dy, x+dx]):
                self.add_fragment(a, x + dx, y + dy, self.fragment_set[k].get(f).class_frag)

    def fuzzy_seg(self):
        print('Segmentation started ...')
        self.reset()
        self.add_fragment_from_seed()
        for k in range(self.num_aff-1, 0-1, -1):
            f = 0
            while(f < self.fragment_set[k].idx):
                x = self.fragment_set[k].get(f).x
                y = self.fragment_set[k].get(f).y
                if(self.final_class[y, x] == 0):
                    self.final_class[y, x] = self.fragment_set[k].get(f).class_frag
                self.proces_fragment(k, x, y, 1, 0, f)
                self.proces_fragment(k, x, y, -1, 0, f)
                self.proces_fragment(k, x, y, 0, 1, f)
                self.proces_fragment(k, x, y, 0, -1, f)
                self.proces_fragment(k, x, y, 1, 1, f)
                self.proces_fragment(k, x, y, 1, -1, f)
                self.proces_fragment(k, x, y, -1, 1, f)
                self.proces_fragment(k, x, y, -1, -1, f)
                f += 1
        print('Segmentation done!')
        # uncomment if want to save the classes and affinities on text files
        # np.savetxt('final_classes.out', self.final_class, delimiter=',', fmt='%i')
        # np.savetxt('affinity.out', self.greatestk, delimiter=',', fmt='%1.3f')


