from gettext import translation
import cv2
import numpy as np
import argparse
points = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor', type=int, default=1, help='floor1 or floor2')
    
    args = parser.parse_args()
    return args

class Projection(object):

    def __init__(self, image_path, points):
        """
            :param points: Selected pixels on top view(BEV) image
        """

        if type(image_path) != str:
            self.image = image_path
        else:
            self.image = cv2.imread(image_path)
        self.height, self.width, self.channels = self.image.shape
        
    def top_to_front(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0, fov=90):
        """
            Project the top view pixels to the front view pixels.
            :return: New pixels on perspective(front) view image
        """
        
        ### TODO ###    
        new_pixels = []
        f = 1 / (2/512 * np.tan(np.pi/180*fov/2)) # 1/focal_length = 2/N * tan(hfov/2)
        Z = 2.6 
        theta2rad = np.pi/180*theta

        extrinsic = np.array([[1, 0, 0, dx],
                              [0, np.cos(theta2rad), -np.sin(theta2rad), dy],
                              [0, np.sin(theta2rad), np.cos(theta2rad), dz],
                              [0, 0, 0, 1]])

        N = np.zeros((3, 4))
        N[0, 0], N[1, 1], N[2, 2] = 1, 1, 1

        K = np.zeros((3, 3))
        K[0, 0], K[1, 1], K[2, 2] = -f, -f, 1
        K[0, 2], K[1, 2] = 256, 256

        for u, v in points:
            X = Z * (256-u) / f 
            Y = Z * (256-v) / f
            P = np.array([X, Y ,Z ,1]).reshape((4, 1))
            new_pixel = K @ N @ extrinsic @ P
            new_pixels.append((new_pixel[0,0]/new_pixel[2,0], new_pixel[1,0]/new_pixel[2,0]))
        
        return new_pixels
    

    def show_image(self, new_pixels, i, img_name='projection', color=(0, 0, 255), alpha=0.4):
        """
            Show the projection result and fill the selected area on perspective(front) view image.
        """

        new_image = cv2.fillPoly(
            self.image.copy(), [np.array(new_pixels, dtype=np.int32)], color)
        new_image = cv2.addWeighted(
            new_image, alpha, self.image, (1 - alpha), 0)

        cv2.imshow(
            f'Top to front view projection {img_name}', new_image)
        cv2.imwrite(img_name+i+'.png', new_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return new_image


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        print(x, ' ', y)
        points.append([x, y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(x) + ',' + str(y), (x+5, y+5), font, 0.5, (0, 0, 255), 1)
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:

        print(x, ' ', y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        # cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 1, (255, 255, 0), 2)
        cv2.imshow('image', img)



if __name__ == "__main__":

    pitch_ang = 90
    args = parse_args()
    i = 1
    if args.floor == 1: # floor1
        front_rgb = f"./data_task1/front_view1.png"
        top_rgb = f"./data_task1/top_view1.png"
    else:              # floor2
        front_rgb = f"./data_task1/front_view2.png"
        top_rgb = f"./data_task1/top_view2.png"
        i = 2

    # click the pixels on window
    img = cv2.imread(top_rgb, 1)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)

    projection = Projection(front_rgb, points)
    new_pixels = projection.top_to_front(theta=pitch_ang, dy=1.2252347-0.12523484)
    projection.show_image(new_pixels, str(i))

