import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image
import numpy as np

class CocoStuffDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):

        opt.load_size = 256
        opt.crop_size = 256
        opt.label_nc = 67+1
        opt.contain_dontcare_label = True
        opt.semantic_nc = 69 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.images, self.labels, self.instances, self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.paths[0], self.images[idx])).convert('RGB')
        label = Image.open(os.path.join(self.paths[1], self.labels[idx]))
        instance = Image.open(os.path.join(self.paths[2], self.instances[idx]))

        image, label, instance = self.transforms(image, label, instance)

        label = label * 255 + 1
        label[label == 256] = 0 # unknown class should be zero for correct losses

        return {"image": image, "label": label, "instance": instance, "name": self.images[idx]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train"
        path_img = os.path.join(self.opt.dataroot, mode+"_img")
        path_lab = os.path.join(self.opt.dataroot, mode+"_label")
        path_inst = os.path.join(self.opt.dataroot, mode+"_inst")
        images = sorted(os.listdir(path_img))
        labels = sorted(os.listdir(path_lab))
        instances = sorted(os.listdir(path_inst))
        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        for i in range(len(images)):
            assert os.path.splitext(images[i])[0] == os.path.splitext(labels[i])[0], '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, instances, (path_img, path_lab, path_inst)


    def transforms(self, image, label, instance):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        instance = TR.functional.resize(instance, (new_width, new_height), Image.NEAREST)
        # crop
        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
        instance = instance.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
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