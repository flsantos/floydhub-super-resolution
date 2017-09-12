from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np



def super_resolve(input_image, output_filename, model, cuda):
	img = Image.open(input_image).convert('YCbCr')
	y, cb, cr = img.split()

	model = torch.load(model)
	input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

	if cuda:
		model = model.cuda()
		input = input.cuda()

	out = model(input)
	out = out.cpu()
	out_img_y = out.data[0].numpy()
	out_img_y *= 255.0
	out_img_y = out_img_y.clip(0, 255)
	out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

	out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
	out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
	out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

	out_img.save(output_filename)
	return output_filename



if __name__ == "__main__":
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
	parser.add_argument('--input_image', type=str, required=True, help='input image to use')
	parser.add_argument('--model', type=str, required=True, help='model file to use')
	parser.add_argument('--output_filename', type=str, help='where to save the output image')
	parser.add_argument('--cuda', action='store_true', help='use cuda')
	opt = parser.parse_args()

	print(opt)

	out = super_resolve(opt.input_image, opt.output_filename, opt.model, opt.cuda)
	print('output image saved to ', out)
