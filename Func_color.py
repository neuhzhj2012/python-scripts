#encoding: utf-8
import numpy as np
import cv2

def colormap(rgb=False):
    color_list = np.array(
        [
            255, 0, 0,
            255, 255, 0,
            0, 0, 255,
            255, 0, 255,
            220, 20, 60,
            218, 112, 214,
            50, 205, 50,
            255,192,203,
            0, 139, 139,
            219,112,147,
            218, 165, 32,
            0, 255, 255,
            255, 20, 147,
            255, 165, 0,
            0, 0, 139,
            128, 0, 128,
            95, 158, 160,
            148,0,211,
            100,149,237,
            123,104,238,
            135,206,235,
            127,255,170,
            255, 99, 71,
            205, 133, 63,
            205, 92, 92,
            255, 215, 0
        ]
    ).astype(np.uint8)
    # color_list = color_list.reshape((-1, 3)) * 255 #np.float32
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

if __name__ == '__main__':
    cmap = colormap(rgb=False)
    color_num = cmap.shape[0]

    recth = 10
    rectw = 18
    img = np.ones((color_num * recth * 2 , color_num * rectw * 2, 3), np.uint8) * 255

    imgh, imgw = img.shape[:2]

    color_id = 0
    color_id_tmp = 0
    flag_txt = False

    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(0, imgh, recth*2):
        for x in range(0, imgw, rectw*2):
            color = cmap[color_id % color_num].tolist()

            if int(y //(recth*2)) in [5,6,7,8,9]:
                if not flag_txt:
                    cv2.putText(img, str(color_id_tmp), (x, 7 * (recth*2)), font, 1, color)
                    color_id_tmp += 1
                    if color_id_tmp == color_num:
                        flag_txt = True

            else:
                print ('id: {}, value: {}'.format(color_id, color))
                cv2.rectangle(img, (x, y), (x + rectw, y + recth), color)
            color_id += 1
    cv2.imwrite('color.jpg', img)



