class ImageTensor
{
public:
	enum ETensorType {
		NHWC	= 0,	// batch, height, width, channel(1 or 3)
		NCHW	= 1,	// batch, channel(1 or 3), height, width
	};
	enum ETensorDataType {
		D0_255	= 0,	// batch, height, width, channel(1 or 3)
		D_1_1	= 1,	// batch, channel(1 or 3), height, width
	};

	ImageTensor(ETensorType type, int n, int w, int h, int c, unsigned char* buff = nullptr);
	virtual ~ImageTensor();

	void *getBuff() { return buff_; };
	unsigned long getBuffSize() { return img_size_ * batch_index_; };
	ETensorType getType() { return type_; };

	// addできるのは同じサイズの画像データのみ
	void add(ImageTensor* data);

	// cropできるのはbatchサイズが1の時だけ
	void crop(int x, int y, int w, int h, ImageTensor* out);

	void print();
	void print_raw();

private:
	inline unsigned char getDataByNHWC(int n, int x, int y, int c) {
		return buff_[img_size_*n + line_size_*y + pix_size_*x + c];
	}
	inline unsigned char getDataByNCHW(int n, int x, int y, int c) {
		return buff_[img_size_*n + plane_size_*c + line_size_*y + x];
	}
	inline void setDataToNHWC(int n, int x, int y, int c, unsigned char data) {
		buff_[img_size_*n + line_size_*y + pix_size_*x + c] = data;
	}
	inline void setDataToNCHW(int n, int x, int y, int c, unsigned char data) {
		buff_[img_size_*n + plane_size_*c + line_size_*y + x] = data;
	}

private:
	ETensorType type_;
	int batch_;
	int width_;
	int height_;
	int channel_;
	unsigned char *buff_;
	bool is_manage_buff_;
	int batch_index_;
	unsigned long img_size_;	// by bytes
	unsigned long line_size_;	// by bytes
	unsigned long plane_size_;	// by bytes
	unsigned long pix_size_;	// by bytes
};