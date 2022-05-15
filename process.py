import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def apply_clahe(
    path,
    out_path
):

    images=[i for i in os.listdir(path) if not (i.startswith('.'))]
    for img in images:
        image_c = cv2.imread('/Users/osama-mac/Desktop/UrbanExplorer/bright/'+img)

        #Generating the histogram of the original image
        hist_c,bins_c = np.histogram(image_c.flatten(),256,[0,256])

        #Generating the cumulative distribution function of the original image
        cdf_c = hist_c.cumsum()
        cdf_c_normalized = cdf_c * hist_c.max()/ cdf_c.max()

        #Converting the image to YCrCb
        image_yuv = cv2.cvtColor(image_c, cv2.COLOR_BGR2YUV)

        #Creating CLAHE 
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))

        # Applying Histogram Equalization on the original imageof the Y channel
        image_yuv[:,:,0] = clahe.apply(image_yuv[:,:,0])

        # convert the YUV image back to RGB format
        image_c_clahe = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        #Generating the histogram of the image after applying CLAHE
        hist_c_clahe, bins_c_clahe = np.histogram(image_c_clahe.flatten(),256,[0,256])

        #Generating the cumulative distribution function of the original image
        cdf_c_clahe = hist_c_clahe.cumsum()
        cdf_c_clahe_normalized = cdf_c_clahe * hist_c_clahe.max()/ cdf_c_clahe.max()

        #Plotting the Original and Histogram Equalized Image, Histogram and CDF
        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB))
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Original Image')

        axs[0, 1].imshow(cv2.cvtColor(image_c_clahe, cv2.COLOR_BGR2RGB))
        sav=os.path.join(out_path,img)
        cv2.imwrite(sav,image_c_clahe)
        axs[0, 0].axis('off')
        axs[0, 1].set_title('Image after CLAHE')


        axs[1, 0].plot(cdf_c_normalized, color = 'b')
        axs[1, 0].hist(image_c.flatten(),256,[0,256], color = 'r')
        axs[1, 0].legend(('cdf','histogram'), loc = 'upper left')



        axs[1, 1].plot(cdf_c_clahe_normalized, color = 'b')
        axs[1, 1].hist(image_c_clahe.flatten(),256,[0,256], color = 'r')
        axs[1, 1].legend(('cdf_clahe','histogram_clahe'), loc = 'upper left')

def get_arg():
    parser = argparse.ArgumentParser(description='Apply CLAHE to IHC images')
    parser.add_argument('--input_path', type=str, help='images path ./images/')
    parser.add_argument('--output_path', type=str, help='output save path ./output')
    return parser.parse_args()

if __name__ == '__main__':
    args=get_arg()
    apply_clahe(args.input_path,args.output_path)
    print('Finished Sucessfully!')


