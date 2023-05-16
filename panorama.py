import numpy as np
import scipy as sp
import cv2
import scipy.ndimage as nd


def getImageCorners(image):
    corners = np.zeros((4, 1, 2), dtype="float32")
    height,weight=image.shape[0],image.shape[1]  # 获取输入图像的高和宽
    
    corners[0,0]=[0,0]                           # 左上角
    corners[1,0]=[0,height-1]                    # 左下角
    corners[2,0]=[weight-1,0]                    # 右上角
    corners[3,0]=[weight-1,height-1]             # 右下角
    
    return corners


def findMatchesBetweenImages(image_1, image_2, num_matches):
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image_1, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image_2, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:num_matches]
    return image_1_kp, image_2_kp, matches


def findHomography(image_1_kp, image_2_kp, matches):
    image_1_points = np.zeros((len(matches), 1, 2), dtype="float32")
    image_2_points = np.zeros((len(matches), 1, 2), dtype="float32")
    for i in range(len(matches)):  # 遍历匹配并将每个匹配关键点的坐标存储在对应的数组中
        q,t=matches[i].queryIdx,matches[i].trainIdx  # 从matchs中获取索引
        image_1_points[i,0]=image_1_kp[q].pt         # 图像1是query图像
        image_2_points[i,0]=image_2_kp[t].pt         # 图像2是train图像
    H,mask = cv2.findHomography(image_1_points,
                                image_2_points,
                                cv2.RANSAC,5.0) # 调用cv2.findhomography()获取homography
    return H


def getBoundingCorners(corners_1, corners_2, homography):
    n_corners_1=cv2.perspectiveTransform(corners_1, homography)  # 对图1的关键点进行变换
    n_corners_2=corners_2
    
    lu=np.array([0,0]).astype('float64')
    rd=np.array([0,0]).astype('float64')
    
    x=np.array([n_corners_1[:,0,0],n_corners_2[:,0,0]])  #获取所有x坐标
    y=np.array([n_corners_1[:,0,1],n_corners_2[:,0,1]])  #获取所有y坐标
    
    lu[0]=np.min(x)  # 左上角
    lu[1]=np.min(y)
    rd[0]=np.max(x)  # 右下角
    rd[1]=np.max(y)
    return lu,rd


def warpCanvas(image, homography, min_xy, max_xy):
    canvas_size = tuple(np.round(max_xy - min_xy+1).astype("int")) # 画布尺寸

    translation_M=np.array([[1, 0, -min_xy[0]],
                            [0, 1, -min_xy[1]],
                            [0, 0, 1]],dtype="float64")  # translation矩阵
    
    M=np.dot(translation_M,homography)  # 带有translation的homography矩阵
    processed = cv2.warpPerspective(image,M,canvas_size)  # 对image进行扭曲，并将结果嵌入到一个足够大的画布中

    return processed


def createImageMask(image):
    h,w,d  = np.atleast_3d(image).shape
    mask = np.zeros((h,w),dtype="int")  # 返回int型array也能进行逻辑运算

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)  #由image生成二值图，阈值为0，即所有大于0的像素点均为255

    contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓
    cv2.drawContours(mask,contours,-1,1,-1)  # 绘制mask
    
    return mask.astype("bool")
    

def createRegionMasks(left_mask,right_mask):
    left_right=np.bitwise_and(left_mask,right_mask)    # True值表示所有的像素是left_mask和right_mask的一部分
    left_only=np.bitwise_and(left_mask,~left_right)    # True值表示所有的像素是left_mask的一部分而不是right_mask的一部分
    right_only=np.bitwise_and(right_mask,~left_right)  # True值表示所有的像素是right_mask的一部分而不是left_mask的一部分

    return left_only,left_right,right_only

def findDistanceToMask(mask):
    mat=~mask # true,false取反
    dis = nd.distance_transform_edt(mat) # 计算图像中非零点到最近背景点（即0）的距离
    dis=dis.astype("float")
    
    return dis

def generateAlphaWeights(left_distance,right_distance):
    res=right_distance/(left_distance+right_distance)
    
    return res



def smooth_blend(left_image,right_image,left_preference):
    left_mask = createImageMask(left_image)            # 获取left_image的有效区域
    right_mask = createImageMask(right_image)          # 获取right_image的有效区域
     
    height, width, channels = np.atleast_3d(left_image).shape

    # 生成只属于left_image的有效区域，混合有效区域和只属于right_image的有效区域
    left_only_mask, overlap_mask, right_only_mask = createRegionMasks(left_mask, right_mask)

    # 获取mask中无效区域的每个点到有效区域的最短距离
    left_distance_mask = findDistanceToMask(left_only_mask)   
    right_distance_mask = findDistanceToMask(right_only_mask)

    # 根据距离获取alpha
    alpha_mask = generateAlphaWeights(left_distance_mask, right_distance_mask)

    # 对于三通道图，mask增加一个维度
    if channels==3:
        left_only_mask = left_only_mask.reshape(height,width,1)
        alpha_mask = alpha_mask.reshape(height,width,1)
        overlap_mask = overlap_mask.reshape(height,width,1)
        right_only_mask = right_only_mask.reshape(height,width,1)
        
    result = left_image * left_only_mask             # 先放入只属于left_image的像素点
    
    left_ratio = alpha_mask * (left_preference + 1)  # 混合图中left_image的权重
    left_ratio[left_ratio>1] = 1                     # 权重最大为1
    right_ratio = 1 - left_ratio                     # 混合图中right_image的权重
    mix_img = left_image.astype("float") * left_ratio\
              + right_image.astype("float") * right_ratio  # 按权重获取混合图像 
    
    result += (mix_img * overlap_mask).astype("uint8")  # 放入属于混合区域的像素点
    
    result += right_image * right_only_mask  # 放入只属于right_image的像素点

    return result



def blendImagePair(image_1, image_2, num_matches):
    kp1, kp2, matches = findMatchesBetweenImages(image_1, image_2, num_matches)
    homography = findHomography(kp1, kp2, matches)
    corners_1 = getImageCorners(image_1)
    corners_2 = getImageCorners(image_2)
    min_xy, max_xy = getBoundingCorners(corners_1, corners_2, homography)
    left_image = warpCanvas(image_1, homography, min_xy, max_xy)
    output_image = np.zeros_like(left_image)
    right_image = np.zeros_like(left_image)
    min_xy = min_xy.astype("int")
    min_x = int(np.round(min_xy[0]))
    min_y = int(np.round(min_xy[1]))
    right_image[-min_y:-min_y + image_2.shape[0],
                 -min_x:-min_x + image_2.shape[1]] = image_2
    
    output_image = smooth_blend(left_image,right_image,0.1)

    return output_image
