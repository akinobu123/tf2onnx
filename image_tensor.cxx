#include <stdio.h>
#include <string.h>
#include <iostream>
#include "image_tensor.hxx"

ImageTensor::ImageTensor(ImageTensor::ETensorType type, int n, int w, int h, int c, unsigned char* buff)
	: type_(type)
	, batch_(n)
	, width_(w)
	, height_(h)
	, channel_(c)
	, buff_(buff)
	, is_manage_buff_(false)
	, batch_index_(0)
	, img_size_(0)
	, line_size_(0)
	, plane_size_(0)
	, pix_size_(0)
{
	// check
	if(buff != nullptr && n != 1) {
		printf("ImageTensor::ImageTensor() ERROR : buffer is specified and n is not 1.\n");
		throw std::exception();
	}
	// initialize buffer
	if(buff == nullptr) {
		is_manage_buff_ = true;
		buff_ = (unsigned char *)calloc(n * h * w * c, sizeof(unsigned char));
	} else {
		is_manage_buff_ = false;
		buff_ = buff;
		batch_index_ = 1;	// image must be only one.
	}
	// calc parts size
	if(type == ImageTensor::NHWC) {
		pix_size_	= channel_;
		line_size_	= width_ * pix_size_;
		plane_size_	= line_size_ * height_;
		img_size_	= plane_size_ * 1;
	} else if(type == ImageTensor::NCHW) {
		pix_size_	= 1;
		line_size_	= width_ * pix_size_;
		plane_size_	= line_size_ * height_;
		img_size_	= plane_size_ * channel_;
	} else {

	}
}

ImageTensor::~ImageTensor()
{
	if(is_manage_buff_ == true) {
		free(buff_);
	}
}

// add
void ImageTensor::add(ImageTensor* data)
{
	// check
	if(batch_ < (batch_index_ + data->batch_)) {
		printf("ImageTensor::add() ERROR : batch is full.\n");
		throw std::exception();
	}
	if(data->height_ != height_ || data->width_ != width_ || data->channel_ != channel_) {
		printf("ImageTensor::add() ERROR : batch is full.\n");
		throw std::exception();
	}
	// memory copy
	void *dst = buff_ + (img_size_ * batch_index_);
	void *src = data->buff_;
	size_t size = img_size_ * data->batch_;
	memcpy(dst, src, size);
	// increment index
	batch_index_ += data->batch_;
}

// cropできるのはbatchサイズが1の時だけ
void ImageTensor::crop(int x, int y, int w, int h, ImageTensor* out)
{
	// check
	if(batch_ != 1 || out->batch_ != 1) {
		printf("ImageTensor::crop() ERROR : batch_ is not 1.\n");
		throw std::exception();
	}
	if(out == nullptr) {
		printf("ImageTensor::crop() ERROR : out is null.\n");
		throw std::exception();
	}
	if(x >= width_ || (x+w) > width_) {
		printf("ImageTensor::crop() ERROR : x and/or w is out of image.\n");
		throw std::exception();
	}
	if(y >= height_ || (y+h) > height_) {
		printf("ImageTensor::crop() ERROR : y and/or h is out of image.\n");
		throw std::exception();
	}
	if(out->width_ != w || out->height_ != h || out->channel_ != channel_) {
		printf("ImageTensor::crop() ERROR : out is different at size or type.\n");
		throw std::exception();
	}
	// move data to out
	unsigned char data;
	if(this->type_ == ImageTensor::NHWC && out->type_ == ImageTensor::NHWC) {
		for(int xi = 0; xi < w; xi++) {
			for(int yi = 0; yi < h; yi++) {
				for(int ci = 0; ci < channel_; ci++) {
					data = this->getDataByNHWC(0, x+xi, y+yi, ci);
					out->setDataToNHWC(0, xi, yi, ci, data);
				}
			}
		}
	} else if(this->type_ == ImageTensor::NHWC && out->type_ == ImageTensor::NCHW) {
		for(int xi = 0; xi < w; xi++) {
			for(int yi = 0; yi < h; yi++) {
				for(int ci = 0; ci < channel_; ci++) {
					data = this->getDataByNHWC(0, x+xi, y+yi, ci);
					out->setDataToNCHW(0, xi, yi, ci, data);
				}
			}
		}
	} else if(this->type_ == ImageTensor::NCHW && out->type_ == ImageTensor::NHWC) {
		for(int xi = 0; xi < w; xi++) {
			for(int yi = 0; yi < h; yi++) {
				for(int ci = 0; ci < channel_; ci++) {
					data = this->getDataByNCHW(0, x+xi, y+yi, ci);
					out->setDataToNHWC(0, xi, yi, ci, data);
				}
			}
		}
	} else if(this->type_ == ImageTensor::NCHW && out->type_ == ImageTensor::NCHW) {
		for(int xi = 0; xi < w; xi++) {
			for(int yi = 0; yi < h; yi++) {
				for(int ci = 0; ci < channel_; ci++) {
					data = this->getDataByNCHW(0, x+xi, y+yi, ci);
					out->setDataToNCHW(0, xi, yi, ci, data);
				}
			}
		}
	}
}

void ImageTensor::print()
{
	if(this->type_ == ImageTensor::NHWC) {
		for(int ni = 0; ni < batch_; ni++) {
			printf("batch=%d\n", ni);
			for(int yi = 0; yi < height_; yi++) {
				for(int xi = 0; xi < width_; xi++) {
					for(int ci = 0; ci < channel_; ci++) {
						unsigned char data = this->getDataByNHWC(ni, xi, yi, ci);
						printf("%03u, ", data);
					}
					printf("  ");
				}
				printf("\n");
			}
			printf("\n\n");
		}
	} else if(this->type_ == ImageTensor::NCHW) {
		for(int ni = 0; ni < batch_; ni++) {
			printf("batch=%d\n", ni);
			for(int ci = 0; ci < channel_; ci++) {
				for(int yi = 0; yi < height_; yi++) {
					for(int xi = 0; xi < width_; xi++) {
						unsigned char data = this->getDataByNCHW(ni, xi, yi, ci);
						printf("%03u, ", data);
					}
					printf("\n");
				}
				printf("\n");
			}
			printf("\n");
		}
	}
}

void ImageTensor::print_raw() {
	unsigned long data_count = img_size_ * batch_index_;
	for(int i=0; i < data_count; i++) {
		printf("%03u, ", buff_[i]);
	}
}
