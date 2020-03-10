import os
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl


def vis_keypoints(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image_dict, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines=[], input_shape=None, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
        y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
        z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

        if kpt_3d_vis[i1, 0] > 0 and kpt_3d_vis[i2, 0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1, 0] > 0:
            ax.scatter(kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker='o')
        if kpt_3d_vis[i2, 0] > 0:
            ax.scatter(kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker='o')

    # x_r = np.array([0, input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)

    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    # ax.set_xlim([0,input_shape[1]])
    # ax.set_ylim([0,1])
    # ax.set_zlim([-input_shape[0],0])
    ax.legend()

    plt.show()
    # data = fig2data(fig)
    # cv2.imshow(filename, data)
    # cv2.waitKey(0)


def vis_3d(kpt_3d, kps_lines, coordinate="WC", title=None):
    """
    世界坐标系 -> 相机坐标系
    :param kpt_3d:
    :param kps_lines:
    :param coordinate: World coordinates(WC) and camera coordinates(CC)
    :param title:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kpt_3d) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]
    ax.set_title(title)
    if coordinate == "CC":
        index = np.asarray([0, 2, 1])
        kpt_3d = kpt_3d[:, index]
        x_3d = kpt_3d[:, 0]
        y_3d = kpt_3d[:, 1]
        z_3d = -kpt_3d[:, 2]
        ax.set_xlabel('X Label')
        ax.set_ylabel('Z Label')
        ax.set_zlabel('Y Label')
    elif coordinate == "WC":
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        x_3d = kpt_3d[:, 0]
        y_3d = kpt_3d[:, 1]
        z_3d = kpt_3d[:, 2]
    else:
        raise Exception("Error:{}".format(coordinate))
    ax.scatter(x_3d, y_3d, z_3d, c=colors[0], marker='o')
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([x_3d[i1], x_3d[i2]])
        y = np.array([y_3d[i1], y_3d[i2]])
        z = np.array([z_3d[i1], z_3d[i2]])
        ax.plot(x, y, z, c=colors[l], linewidth=2)

    # x_r = np.array([0, input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)
    # ax.set_xlim([0,input_shape[1]])
    # ax.set_ylim([0,1])
    # ax.set_zlim([-input_shape[0],0])
    ax.legend()
    plt.show()
    # data = fig2data(fig)
    # cv2.imshow(filename, data)
    # cv2.waitKey(0)


def fig2data(fig):
    """
    fig = plt.figure()
    image_dict = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image


if __name__ == "__main__":
    kpt_3d = np.asarray([[131.84857, 117.998405, 0.5],
                         [145.72697, 119.57478, 0.52360034],
                         [134.98578, 161.60707, 0.6224624],
                         [148.37857, 208.31126, 0.5981282],
                         [117.716125, 116.393166, 0.47639966],
                         [122.96823, 164.84807, 0.52039963],
                         [128.57422, 214.05252, 0.5434407],
                         [139.20627, 93.79163, 0.4600044],
                         [135.53119, 64.558525, 0.43536133],
                         [134.23317, 53.431286, 0.4631416],
                         [137.92296, 42.45516, 0.42735863],
                         [121.77054, 74.47952, 0.42335498],
                         [109.62302, 103.087, 0.46514747],
                         [100.960884, 122.004486, 0.5518728],
                         [149.35513, 72.8382, 0.46819508],
                         [155.54675, 101.89988, 0.52692795],
                         [153.07935, 123.845924, 0.6085933],
                         [135.68422, 73.65164, 0.4457749]], dtype=np.float32)
    test_kpt_3d = np.asarray([[100, 60, 0.5],
                              [50, 25, 0.6]], dtype=np.float32)
    kps_lines = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
                 (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    # vis_3d(test_kpt_3d)
    vis_3d(kpt_3d, kps_lines)
    kpt_3d_vis = np.zeros(shape=(len(kpt_3d), 1)) + 1

    input_shape = (256, 256)
    vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, input_shape)
