import cv2
from skimage import io


def mapcolors(x):
    newEX = cv2.cvtColor(x[0,:,:], cv2.COLOR_GRAY2BGR)*255
    newEX[newEX[:, :, 0] == 255] = [0, 255, 0]
    newHE = cv2.cvtColor(x[1, :, :], cv2.COLOR_GRAY2BGR)*255
    newHE[newHE[:, :, 0] == 255] = [51, 153, 255]
    newMA = cv2.cvtColor(x[2, :, :], cv2.COLOR_GRAY2BGR)*255
    newMA[newMA[:, :, 0] == 255] = [255, 102, 102]
    newSE = cv2.cvtColor(x[3, :, :], cv2.COLOR_GRAY2BGR)*255
    newSE[newSE[:, :, 0] == 255] = [255, 255, 0]
    merge = newEX+newHE+newMA+newSE
    return merge.astype('uint8')

def save_preds(name,gt,pred,dir):
    single_gt = mapcolors(gt)
    single_preds = mapcolors(pred)
    io.imsave(dir+name+"_gt.png", single_gt)
    io.imsave(dir+name + "_pred.png", single_preds)