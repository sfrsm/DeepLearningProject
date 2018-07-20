import cv2
import math

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# Quantization Matrix
QUANTIZATION_MAT = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

def Image2DCT(img):

# reading image in grayscale style
# img = cv2.imread('frame001.jpg', cv2.IMREAD_GRAYSCALE)

# You can try with this matrix to understand working of DCT
# img = np.array([[255,255,227,204,204,203,192,217],[215,189,167,166,160,135,167,244],[169,115,99,99,99,82,127,220],[146,90,86,88,84,63,195,189],[255,255,231,239,240,182,251,232],[255,255,21,245,226,169,229,247],[255,255,222,251,174,209,174,163],[255,255,221,184,205,248,249,220]])


# get size of the image
    [h, w] = img.shape

# No of blocks needed : Calculation

    height = h
    width = w
    h = np.float32(h)
    w = np.float32(w)

    nbh = math.ceil(h / block_size)
    nbh = np.int32(nbh)

    nbw = math.ceil(w / block_size)
    nbw = np.int32(nbw)

# Pad the image, because sometime image size is not dividable to block size
# get the size of padded image by multiplying block size by number of blocks in height/width

# height of padded image
    H = block_size * nbh

# width of padded image
    W = block_size * nbw

# create a numpy zero matrix with size of H,W
    padded_img = np.zeros((H, W))
    padded_img2 = np.zeros((nbh, nbw))

# copy the values of img into padded_img[0:h,0:w]
# or this other way here
    padded_img[0:height, 0:width] = img[0:height, 0:width]


# start encoding:
# divide image into block size by block size (here: 8-by-8) blocks
# To each block apply 2D discrete cosine transform
# reorder DCT coefficients in zig-zag order
# reshaped it back to block size by block size (here: 8-by-8)

    for i in range(nbh):

        # Compute start and end row index of the block
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            # Compute start & end column index of the block
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            # block = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]
            block = padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2]

            # apply 2D discrete cosine transform to the selected block
            DCT = cv2.dct(block)

            DCT_normalized = np.divide(DCT, QUANTIZATION_MAT).astype(int)

            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reordered = zigzag(DCT_normalized)

            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshaped = np.reshape(reordered, (block_size, block_size))

            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1: row_ind_2, col_ind_1: col_ind_2] = reshaped
            padded_img2[i, j] = padded_img[row_ind_1 , col_ind_1]

    return np.uint8(padded_img2)
