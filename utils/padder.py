from typing import Tuple
import numpy as np
import cv2


class Padder():
    def __init__(self, tile_size):
        self.top = -1
        self.bottom = -1
        self.left = -1
        self.right = -1
        self.tile_size = tile_size
        self.padded_img_shape = ()

    @property
    def padded_img_shape(self) -> Tuple:
        return self._padded_img_shape

    @padded_img_shape.setter
    def padded_img_shape(self, value:Tuple):
        self._padded_img_shape = tuple(value)

    def cal_new_img_size_to_tile_multiple(self, image) -> Tuple:
        img_height = image.shape[0]
        img_width = image.shape[1]
        num_tiles_col = (img_height // self.tile_size) + 1 if img_height % self.tile_size else img_height // self.tile_size
        num_tiles_row = (img_width // self.tile_size) + 1 if img_width % self.tile_size else img_width // self.tile_size
        new_img_height = num_tiles_col * self.tile_size
        new_img_width = num_tiles_row * self.tile_size
        self.padded_img_shape = (new_img_height, new_img_width)
        return self.padded_img_shape

    def cal_padding_dim(self, image) -> Tuple:
        (new_img_height, new_img_width) = self.cal_new_img_size_to_tile_multiple(image)
        vertical_pad = new_img_height - image.shape[0]
        horizontal_pad = new_img_width - image.shape[1]
        top, bottom, left, right = vertical_pad//2, vertical_pad-vertical_pad//2, horizontal_pad//2, horizontal_pad-horizontal_pad//2
        self.top, self.bottom, self.left, self.right = top, bottom, left, right
        return (self.top, self.bottom, self.left, self.right)

    def pad_image(self, image) -> np.ndarray:
        if (self.top == -1 or self.bottom == -1 or self.left == -1 or self.right == -1):
            self.cal_padding_dim(image)
        border_type = cv2.BORDER_REFLECT
        image = cv2.copyMakeBorder(image, self.top, self.bottom, self.left, self.right, border_type)
        return np.array(image)
    
    def pad_image_to_tile_multiple(self, ds) -> np.ndarray:
        ds = [self.pad_image(ds[i]) for i in range(ds.shape[0])]
        return np.array(ds)

    def print_instance_attributes(self):
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)