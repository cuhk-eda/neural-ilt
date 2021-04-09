import os, csv
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale, Grayscale
import torch.utils.data as data
from fnmatch import fnmatch
from PIL import Image
import numpy as np


class ILTRefineDataset(data.Dataset):
    r"""
    Data loader for Neural-ILT (on-neural-network ILT correction)
    Args:
        data_root: root to layout dir
        margin: reserved margin for on-neural-network ILT correction (cropped_bbox_size = bbox_size + margin)
        scale_dim_w/scale_dim_h: the size after scaling (the NN input/output size)
        split: the list name of the dataset
        read_ref: read the pre-computed cropped bbox information or not
    """
    def __init__(
        self,
        data_root,
        margin=128,
        scale_dim_w=512,
        scale_dim_h=512,
        split="ibm_opc_test",
        read_ref=False,
    ):
        super(ILTRefineDataset, self).__init__()

        self.transforms = Compose(
            [
                Grayscale(num_output_channels=1),
                ToTensor(),
            ]
        )

        self.data_root = data_root
        self.margin = margin
        self.scale_dim_h = scale_dim_h
        self.scale_dim_w = scale_dim_w
        self.split = split
        self.read_ref = read_ref
        if self.read_ref:
            self.ref_crop_box_dict = self.read_ref_crop_box_dict()
        self.layouts = self.load_refine_dataset(
            self.data_root, self.split
        )

    def __getitem__(self, index):
        split_list = ['train', 'test', 'val', 'ibm_opc_test', 'ibm_opc_test_ext',
                      'train_via', 'test_via', 'baseline_via']
        if self.split in split_list:
            layout_design_name = self.layouts[index].split("/")[-1].split("_")[0]
            layout_name = self.layouts[index].split("/")[-1]
            if self.read_ref:
                layout, scale_factor, new_cord = self.load_img(
                    self.layouts[index],
                    ref_info=self.ref_crop_box_dict[layout_design_name]
                )
            else:
                layout, scale_factor, new_cord = self.load_img(
                    self.layouts[index], ref_info=None, slient=False
                )
            target = self.load_target_img(self.layouts[index])
            return layout, target, scale_factor, new_cord, layout_name
        else:
            raise NotImplementedError("Parameter {split} must be specificed correctly")

    def __len__(self):
        return len(self.layouts)

    def load_refine_dataset(self, data_root, split):
        split_list = ['train', 'test', 'val', 'ibm_opc_test', 'ibm_opc_test_ext',
                      'train_via', 'test_via', 'baseline_via']
        if split not in split_list:
            raise NotImplementedError(
                "Parameter {split} should be one of %s" % split_list
            )

        split_list_path = os.path.join(data_root, "%s_list.txt" % split)
        layouts = []

        with open(split_list_path, "r") as split_list:
            for line in split_list.readlines():
                line = line.split("\n")[0]
                image_path, _ = line.split("\t")[0], line.split("\t")[1]
                layouts.append(os.path.join(os.getcwd(), image_path))
        return layouts

    def load_img(self, filepath, ref_info=None, slient=True):
        r"""
        Load the target layout (filepath) into image tensor, process its cropped bbox on-the-fly
        Args:
            filepath: the image file path
            ref_info: read the pre-computed cropped bbox or not, if True, skip the bbox pre-processing
        Return:
            Image tensor of the target layout, cropped bbox information of the target layout
        Cropped bbox:
            [original_size, original_size] -> [corp_bbox_size, corp_bbox_size] -> [scale_size, scale_size]
            corp_bbox_size = max(margin + bbox_size_w, margin + bbox_size_h)
        """
        img = Image.open(filepath)
        img = img.convert("L")

        if ref_info is not None:
            scale_factor, new_cord = ref_info
            cropped_img = img.crop(box=new_cord)
            resized_img = cropped_img.resize(
                size=(self.scale_dim_w, self.scale_dim_h), resample=Image.NEAREST
            )
            image_tensor = self.transforms(resized_img)
            return image_tensor, scale_factor, new_cord

        x1, y1, x2, y2 = img.getbbox()
        max_width, max_height = img.size
        w, h = x2 - x1, y2 - y1
        max_len = max(w, h)
        x_new = int(x1 + w / 2) - int(max_len / 2)
        y_new = int(y1 + h / 2) - int(max_len / 2)
        new_cord = (
            x_new - self.margin,
            y_new - self.margin,
            x_new + max_len + self.margin,
            y_new + max_len + self.margin,
        )
        cropped_dim_w = max_len + 2 * self.margin
        cropped_dim_h = cropped_dim_w
        if (
            (x_new - self.margin) <= 0
            or (y_new - self.margin) <= 0
            or (x_new + max_len + self.margin) >= max_width
            or (y_new + max_len + self.margin) >= max_height
        ):
            new_cord = (0, 0, max_width, max_height)
            cropped_dim_w = max_width
            cropped_dim_h = max_height

        cropped_img = img.crop(box=new_cord)
        resized_img = cropped_img.resize(
            size=(self.scale_dim_w, self.scale_dim_h), resample=Image.NEAREST
        )

        scale_factor_w = cropped_dim_w / self.scale_dim_w
        scale_factor_h = cropped_dim_h / self.scale_dim_h
        scale_factor = (scale_factor_w, scale_factor_h)

        image_tensor = self.transforms(resized_img)

        if not slient:
            print(
                "\nProcessing {} with size of".format(filepath.split("/")[-1]),
                new_cord,
                "and scale factor = [{}, {}]".format(scale_factor_w, scale_factor_h),
            )
        return image_tensor, scale_factor, new_cord

    def load_target_img(self, filepath):
        img = Image.open(filepath)
        img = img.convert("L")
        image_tensor = self.transforms(img)  # 1 * 2048 * 2048
        return image_tensor

    def read_ref_crop_box_dict(self, csv_name="ref_crop_box_dict.csv"):
        r"""
        Read the pre-computed/reference cropped bbox information for each layout
        """
        ref_crop_box_dict_read = {}
        with open(os.path.join(self.data_root, csv_name)) as csv_file:
            reader = csv.reader(csv_file)
            for line in reader:
                name = line[0]
                scale = (float(line[1]), float(line[2]))
                cord = (int(line[3]), int(line[4]), int(line[5]), int(line[6]))
                ref_crop_box_dict_read[name] = (scale, cord)
        return ref_crop_box_dict_read

    def is_image(self, filename):
        return any(
            filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
        )
