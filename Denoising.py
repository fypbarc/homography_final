import cv2
import pywt
import numpy as np


ho_sym6 = np.array(
    [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466,
     0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578,
     0.0017677118642428036, -0.007800708325034148])
h1_sym6 = np.array(
    [0.007800708325034148, 0.0017677118642428036, -0.04472490177066578, -0.021060292512300564, 0.07263752278646252,
     0.3379294217276218, -0.787641141030194, 0.4910559419267466, 0.048311742585633, -0.11799011114819057,
     -0.0034907120842174702, 0.015404109327027373])
g0_sym6 = np.array(
    [-0.007800708325034148, 0.0017677118642428036, 0.04472490177066578, -0.021060292512300564, -0.07263752278646252,
     0.3379294217276218, 0.787641141030194, 0.4910559419267466, -0.048311742585633, -0.11799011114819057,
     0.0034907120842174702, 0.015404109327027373])
g1_sym6 = np.array(
    [0.015404109327027373, -0.0034907120842174702, -0.11799011114819057, 0.048311742585633, 0.4910559419267466,
     -0.787641141030194, 0.3379294217276218, 0.07263752278646252, -0.021060292512300564, -0.04472490177066578,
     0.0017677118642428036, 0.007800708325034148])
h0_bior3_5 = np.array(
    [-0.013810679320049757, 0.04143203796014927, 0.052480581416189075, -0.26792717880896527, -0.07181553246425874,
     0.966747552403483, 0.966747552403483, -0.07181553246425874, -0.26792717880896527, 0.052480581416189075,
     0.04143203796014927, -0.013810679320049757])
h1_bior3_5 = np.array(
    [0.0, 0.0, 0.0, 0.0, -0.1767766952966369, 0.5303300858899107, -0.5303300858899107, 0.1767766952966369, 0.0, 0.0,
     0.0, 0.0])
g0_bior3_5 = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.1767766952966369, 0.5303300858899107, 0.5303300858899107, 0.1767766952966369, 0.0, 0.0, 0.0,
     0.0])
g1_bior3_5 = np.array(
    [-0.013810679320049757, -0.04143203796014927, 0.052480581416189075, 0.26792717880896527, -0.07181553246425874,
     -0.966747552403483, 0.966747552403483, 0.07181553246425874, -0.26792717880896527, -0.052480581416189075,
     0.04143203796014927, 0.013810679320049757])

alpha = 0.65


def normal(array):
    normalized_array = array / np.linalg.norm(array)
    return normalized_array


class DenoiseImage():
    def __init__(self):
        self.myOtherWavelet = None

    def denoised_image(self, image):
        cust_h0 = np.add(np.multiply(alpha, ho_sym6), np.multiply((1.0 - alpha), h0_bior3_5))
        cust_h0_norm = normal(cust_h0)
        cust_h1 = np.add(np.multiply(alpha, h1_sym6), np.multiply((1 - alpha), h1_bior3_5))
        cust_h1_norm = normal(cust_h1)
        cust_g0 = np.add(np.multiply(alpha, g0_sym6), np.multiply((1.0 - alpha), g0_bior3_5))
        cust_g0_norm = normal(cust_g0)
        cust_g1 = np.add(np.multiply(alpha, g1_sym6), np.multiply((1 - alpha), g1_bior3_5))
        cust_g1_norm = normal(cust_g1)
        filter_bank =  [cust_h0_norm, cust_h1_norm, cust_g0_norm, cust_g1_norm]
        
        self.myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)

        cA, (cH, cV, cD), (cH1, cV1, cD1), (cH2, cV2, cD2) = pywt.wavedec2(image, self.myOtherWavelet, level=3)
        [r3, c3] = np.shape(cD2)
        cV2 = np.zeros((r3, c3))
        cH2 = np.zeros((r3, c3))
        cD2 = np.zeros((r3, c3))
        coeffs = cA, (cH, cV, cD), (cH1, cV1, cD1), (cH2, cV2, cD2)
        denoised_img = pywt.waverec2(coeffs, self.myOtherWavelet)
            
        return denoised_img


    @staticmethod
    def get_fft(img):
        new_img_fourier = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        new_img_fourier_shift = np.fft.fftshift(new_img_fourier)
        new_img_magnitude = 20 * np.log(cv2.magnitude(new_img_fourier_shift[:, :, 0], new_img_fourier_shift[:, :, 1]))
        new_img_magnitude = cv2.normalize(new_img_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        return np.uint8(new_img_magnitude)        
        

    @staticmethod
    def psnr(original, compressed):
        """MSE is zero means no noise is present in the signal. Therefore, PSNR have no importance."""
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * (np.log10(max_pixel / np.sqrt(mse)))
        return psnr
