import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image

class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 34 + 1 #label + edge
        opt.contain_dontcare_label = True
        opt.semantic_nc = 36 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 2.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.instances, self.paths = self.list_images() 

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        instance = Image.open(os.path.join(self.paths[1], self.instances[idx]))

        image, label, instance = self.transforms(image, label, instance)

        label = label * 255 + 1
        label[label == 256] = 0 # unknown class should be zero for correct losses

        return {"image": image, "label": label, "instance": instance, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot, "leftImg8bit", mode)
        for city_folder in sorted(os.listdir(path_img)):
            cur_folder = os.path.join(path_img, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                images.append(os.path.join(city_folder, item))
        labels = []
        instances = []
        path_lab = os.path.join(self.opt.dataroot, "gtFine", mode)
        for city_folder in sorted(os.listdir(path_lab)):
            cur_folder = os.path.join(path_lab, city_folder)
            for item in sorted(os.listdir(cur_folder)):
                if item.find("labelIds") != -1:
                    labels.append(os.path.join(city_folder, item))
                if item.find("instanceIds") != -1:
                    instances.append(os.path.join(city_folder, item))
        assert len(images)  == len(labels) == len(instances), "different len of images and labels and instances %s - %s - %s" % (len(images), len(labels), len(instances))
        for i in range(len(images)):
            assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
                '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, instances, (path_img, path_lab)

    def transforms(self, image, label, instance):
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        instance = TR.functional.resize(instance, (new_width, new_height), Image.NEAREST)

        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
                instance = TR.functional.hflip(instance)

        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        if instance.mode == 'L':
            instance = TR.functional.to_tensor(instance) * 255
        else :
            instance = TR.functional.to_tensor(instance)

        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label, instance