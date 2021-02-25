import cv2
import pywt
import scipy.signal
import scipy.stats as stats
from scipy.signal import argrelextrema
from scipy import ndimage
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd



class Buono_Brutto_Cattivo:
    def __init__(self, img, segment_number=99):
        self.segment_number = segment_number
        self.img = img

    def separate(self):
        segment_number = self.segment_number
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        h,w=img.shape[0],img.shape[1]
        dur = h / segment_number
        # bad_dict = {'w': wavlet.y_axis_freq, 'w': dur}
        # good_dict = {'freq': wavlet.y_axis_freq, 'dur_part': dur}
        # fig,ax=plt.subplots()
        # ax.imshow(img)
        newtimeshape = w // segment_number * segment_number
        print(h,w,newtimeshape)
        wavlet_list = np.hsplit(img[:, :newtimeshape], segment_number)
        print(wavlet_list[0].shape[0],wavlet_list[0].shape[1])
        print(wavlet_list[0])
        X = np.zeros((segment_number, wavlet_list[0].shape[0] * wavlet_list[0].shape[1]))
        for id, wavelet_part in enumerate(wavlet_list):
            X[id, :] = wavelet_part.flatten()
        pca = PCA(n_components=2)
        Xnew = pca.fit_transform(X)
        model = IsolationForest(n_estimators=500)
        res = model.fit_predict(Xnew)
        countminus = np.sum(res == -1)
        if countminus > segment_number // 2:
            res = res * -1
        for i in np.argwhere(res == -1):
            x1, x2 = i * newtimeshape // segment_number, (i + 1) * newtimeshape // segment_number
            # ax.axvspan(x1,x2,alpha=0.3, color='red')
            # ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='blue')
        for i in np.argwhere(res == 1):
            x1, x2 = i * newtimeshape // segment_number, (i + 1) * newtimeshape // segment_number
            # ax.axvspan(x1, x2, alpha=0.3, color='blue')
            # ax2.text((x1+x2)/2,wavlet.y_axis_freq[wavlet.y_axis_freq.shape[0]//2],str(i),color='red')
        allfile = pd.DataFrame(X)
        allfile['y'] = res
        good_frames = allfile.loc[allfile['y'] == 1].T
        # print(good_frames.shape)
        bad_frames = allfile.loc[allfile['y'] == -1].T
        # print(bad_frames.shape)
        col_list_bad = bad_frames.columns.tolist()
        col_list = good_frames.columns.tolist()
        corr = pd.concat([good_frames, bad_frames], axis=1).corr()
        c = []
        cor_bad_good = np.abs(corr[col_list].loc[col_list_bad])
        # print(col_list,col_list_bad)
        # print(cor_bad_good)
        # plt.figure()
        # sns.heatmap(cor_bad_good)
        for ind, col in enumerate(col_list):
            if ind == 0:
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind + 1]])))
            elif col_list[ind] == col_list[-1]:
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind - 1]])))
            else:
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind - 1]])))
                c.append(np.abs(good_frames[col].corr(good_frames[col_list[ind + 1]])))
        lim = np.median(c)
        # print(lim)
        good_another = cor_bad_good > lim * 0.95
        for index, row in good_another.iterrows():
            if row[col_list].mean() > lim:
                res[index] = 1
        repared = allfile.copy()
        repared['y'] = res
        # print('Bad_time:')
        bad=[]
        for i in np.argwhere(res == -1):
            x1, x2 = i * newtimeshape // segment_number, (i + 1) * newtimeshape // segment_number
            # ax.axvspan(x1, x2, alpha=0.3, color='black')
            # print(time1)
            # print(time2)
            bad.append(wavlet_list[i[0]])
            # ax2.axvspan(x1,x2,alpha=0.3, color='black')
        good=[]
        for i in np.argwhere(res == 1):
            x1, x2 = i * newtimeshape // segment_number, (i + 1) * newtimeshape // segment_number
            # ax.axvspan(x1, x2, alpha=0.3, color='white')
            # time1 = float(wavlet.x_axis_time[i * newtimeshape // segment_number])
            # time2 = float(wavlet.x_axis_time[(i + 1) * newtimeshape // segment_number])
            # # print(time1)
            # # print(time2)
            good.append(wavlet_list[i[0]])
            # ax2.axvspan(x1,x2,alpha=0.3, color='black')
        # plt.figure()
        # sns.heatmap(np.abs(allfile.loc[allfile['y'] == 1].T.corr()))
        # plt.figure()
        # sns.heatmap(allfile.T.corr())
        np.concatenate(good)
        plt.imshow( np.concatenate(good,axis=1))
        # plt.show()
        # return good_dict, bad_dict


def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    # imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H
def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
def cross_image(im1, im2):

   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

def _rotare_img(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w=img_gray.shape[0],img_gray.shape[1]
    # kernel = np.ones((5, 5), np.uint8)
    # img_gray=cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_edges = auto_canny(img_gray,0)
    # img_edges = cv2.morphologyEx(img_edges, cv2.MORPH_GRADIENT, kernel)
    # plt.imshow(img_edges)
    # plt.imshow(img_edges)
    line_l=max(h,w)
    lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 100, minLineLength=line_l/3, maxLineGap=line_l/50)
    lines_v = cv2.HoughLinesP(img_edges, 1, np.pi / 90, 100, minLineLength=line_l/3, maxLineGap=line_l/50)
    lines_t=[]
    for [[x1, y1, x2, y2]] in lines:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle<185 and angle>175:
            lines_t.append([[x1, y1, x2, y2]])
    lines_v_t=[]
    for [[x1, y1, x2, y2]] in lines_v:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle<92 and angle>88:
            lines_v_t.append([[x1, y1, x2, y2]])
    if type(lines)==type(None):
        lines=[]
    if type(lines_v)==type(None):
        lines_v=[]
    if len(lines_v_t)>len(lines_t):
        median_angle = np.median(lines_v_t)
        img_rotated = ndimage.rotate(img, median_angle + 90)
    else:
        median_angle = np.median(lines_t)
        img_rotated = ndimage.rotate(img, median_angle)
    #     img = ndimage.rotate(img, 90)
    #     h, w = img.shape[0],img.shape[1]
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img_edges = auto_canny(img_gray)
    #     lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 500, minLineLength=w/4, maxLineGap=w/10)
    # angles = []
    for [[x1, y1, x2, y2]] in lines_v_t:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for [[x1, y1, x2, y2]] in lines_t:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # angles.append(angle)
    plt.imshow(img)
    return img_rotated

def crop_ve(img):
    img_real=img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray=cv2.GaussianBlur(img_gray, (7, 7), 0)
    img_gray = np.uint8(img_gray)
    # thresh, im_bw = cv2.threshold(img_gray, 100, 150, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    img_edges = auto_canny(img_gray)
    contours, x = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_areas = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)

        area = cv2.contourArea(cnt)
        all_areas.append(box)

    sorted_contours = sorted(all_areas, key=cv2.contourArea, reverse=True)
    largest_item = sorted_contours[0]
    cv2.drawContours(img_real, [largest_item], 0, (255, 0, 0), 2)
    # plt.imshow(img_real)
    x, y, w, h = cv2.boundingRect(largest_item)
    cropped = img_real[y:y + h, x:x + w]
    return cropped
class Veneer():
    def __init__(self, file_name):
        img_origin = cv2.imread(file_name)
        # h, w = img_origin.shape[0], img_origin.shape[1]
        # img_origin = img_origin[:, :int(w / 2)]
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # img_origin = cv2.filter2D(img_origin, -1, kernel)
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # close = cv2.morphologyEx(img_origin, cv2.MORPH_CLOSE, kernel1)
        # div = np.float32(img_origin) / (close)
        # img_origin = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
        # img_origin = cv2.bilateralFilter(img_origin, 5, 100, 75)
        self.img_origin=crop_ve(_rotare_img(img_origin))


    def conv_an(self):
        mass = []
        conv = cross_image(self.img_origin, self.img_origin)
        self.conv_img=conv
        # kernel = np.ones((1, 2), np.uint8)  # note this is a horizontal kernel
        # d_im = cv2.dilate(conv, kernel, iterations=1)
        # blackAndWhiteImage = cv2.erode(conv, kernel, iterations=1)
        # plt.imshow(blackAndWhiteImage)
        for i in range(conv.shape[1]):
            filt = conv[:, i]
            locminarg = argrelextrema(filt, np.greater, order=5)[0]
            mass.append(len(locminarg))
        fit = stats.norm.pdf(mass, np.mean(mass), np.std(mass))
        # plt.figure()
        # plt.plot(veheer_count, fit, '-o')
        # plt.hist(bins[:-1], bins, weights=counts / 100)
        ans = np.ceil(mass[np.argmax(fit)])
        return ans
    def filt_img(self,img):
        img = img
        size = 8
        if not size % 2:
            size += 1
        kernel = np.ones((size, size), np.float32) / (size * size)
        filtered = cv2.filter2D(img, -1, kernel)
        filtered = img.astype('float32') - filtered.astype('float32')
        filtered = filtered + 127 * np.ones(img.shape, np.uint8)
        filtered = np.uint8(filtered)
        # cv2.imwrite('outputhpass.jpg', filtered)
        # gray = cv2.cvtColor(filtered,cv2.COLOR_BGR2GRAY)
        gray = w2d(filtered, 'db1', 15)
        # cv2.imwrite('gray.jpg', gray)
        # gray = cv2.blur(gray, (5, 3))
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 2), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(blackAndWhiteImage, kernel, iterations=1)
        blackAndWhiteImage = cv2.erode(d_im, kernel, iterations=1)
        return blackAndWhiteImage

    def edge_detector(self, img,bw_img,minLineLength = 50,maxLineGap = 20):
        img = img
        minLineLength = minLineLength
        maxLineGap = maxLineGap
        lines = cv2.HoughLinesP(bw_img, 1, np.pi / 180, 10, minLineLength, maxLineGap)
        new_lienes = np.zeros(img.shape, np.uint8)
        for linee in lines:
            for x1, y1, x2, y2 in linee:
                # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.line(new_lienes, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return img, new_lienes

    def count_veneer(self, new_lienesbw,minLineLength=100,maxLineGap=5):
        new_lienes=new_lienesbw
        # new_lienesbw=cv2.cvtColor(new_lienesbw, cv2.COLOR_BGR2GRAY)
        minLineLength = minLineLength
        new_lienesbw=np.uint8(new_lienesbw)
        new_lienesbwblur=new_lienesbw
        maxLineGap = maxLineGap
        # new_lienesbwblur = cv2.blur(new_lienesbw, (7, 7))
        cv2.imwrite('new_lienesblur.jpg', new_lienesbwblur)
        lines = cv2.HoughLinesP(new_lienesbwblur, 1, np.pi / 180, 10, minLineLength, maxLineGap)
        for linee in lines:
            for x1, y1, x2, y2 in linee:
                cv2.line(new_lienes, (x1, y1), (x2, y2), (0, 255, 0), 1)
        round = 20
        # clear_matrix=np.zeros(new_lienesbw.shape)
        kernel = np.ones((1, 20), np.uint8)  # note this is a horizontal kernel
        d_im = cv2.dilate(new_lienesbw, kernel, iterations=1)
        clear_matrix = cv2.erode(d_im, kernel, iterations=1)
        for i in range(clear_matrix.shape[1]):
            column = new_lienesbw[:, i]
            for ind in range(0, len(column) - round, round):
                if np.sum(column[ind:ind + round]) > 1:
                    column[ind] = 1
                    column[ind + 1:ind + round] = 0
            clear_matrix[:, i] = column
        clear_matrix = np.where(clear_matrix > 1, 1, clear_matrix)
        veheer_count = np.sort(np.sum(clear_matrix, axis=0)) - 1
        # print(clear_matrix)
        # print(veheer_count.shape)
        counts, bins = np.histogram(veheer_count, bins=150)
        # print(veheer_count)
        veheer_count = np.where(veheer_count < 30, veheer_count, veheer_count)
        fit = stats.norm.pdf(veheer_count, np.mean(veheer_count), np.std(veheer_count))
        fig,ax=plt.subplots()
        ax.plot(veheer_count, fit, '-o')
        ax.hist(bins[:-1], bins, weights=counts / 100)
        ans = np.ceil(veheer_count[np.argmax(fit)]/2)+1
        # print(np.ceil(veheer_count[np.argmax(fit)]))
        (h, w) = clear_matrix.shape
        center = (int(w / 2), int(h / 2))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = center
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        clear_matrix = clear_matrix * 255
        cv2.putText(clear_matrix, str(ans), bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        img = self.img_origin
        cv2.putText(img, str(ans), bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        return ans, img, clear_matrix, fig
