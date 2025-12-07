# GraspSAM源代码结构

# 1.data

## uitls

### 1.evaluation.py

```
import numpy as np
import matplotlib.pyplot as plt

from .grasp_utils import GraspRectangles, detect_grasps


def plot_output(rgb_img, depth_img, grasp_q_img, grasp_angle_img, no_grasps=1, grasp_width_img=None):
    """
    Plot the output of a GG-CNN
    :param rgb_img: RGB Image
    :param depth_img: Depth Image
    :param grasp_q_img: Q output of GG-CNN
    :param grasp_angle_img: Angle output of GG-CNN
    :param no_grasps: Maximum number of grasps to plot
    :param grasp_width_img: (optional) Width output of GG-CNN
    :return:
    """
    gs = detect_grasps(grasp_q_img, grasp_angle_img, width_img=grasp_width_img, no_grasps=no_grasps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(rgb_img)
    for g in gs:
        g.plot(ax)
    ax.set_title('RGB')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(depth_img, cmap='gray')
    for g in gs:
        g.plot(ax)
    ax.set_title('Depth')
    ax.axis('off')

    ax = fig.add_subplot(2, 2, 3)
    plot = ax.imshow(grasp_q_img, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Q')
    ax.axis('off')
    plt.colorbar(plot)

    ax = fig.add_subplot(2, 2, 4)
    plot = ax.imshow(grasp_angle_img, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    ax.set_title('Angle')
    ax.axis('off')
    plt.colorbar(plot)
    plt.show()


def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False
```

### 2.grasp_utils.py

```
import copy
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max


def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


def _grasp_anything_format(grasp: list):
    _, x, y, w, h, theta = grasp
    # index based on row, column (y,x), and the Grasp-Anything dataset's angles are flipped around an axis.
    return Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr


class GraspRectangles:
    """
    Convenience class for loading and operating on sets of Grasp Rectangles.
    """

    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):
        """
        Test if GraspRectangle has the desired attr as a function and call it.
        """
        # Fuck yeah python.
        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Load grasp rectangles from numpy array.
        :param arr: Nx4x2 array, where each 4x2 array is the 4 corner pixels of a grasp rectangle.
        :return: GraspRectangles()
        """
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_ocid_grasp_file(cls, fname):
        """
        Load grasp rectangles from a Cornell dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    gr = np.array([
                        _gr_text_to_no(p0),
                        _gr_text_to_no(p1),
                        _gr_text_to_no(p2),
                        _gr_text_to_no(p3)
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_vmrd_file(cls, fname):
        """
        Load grasp rectangles from a VMRD dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            grasp_lines = f.readlines()
            for grasp_line in grasp_lines:
                # x1, y1, x2, y2, x3, y3, x4, y4 = list(map(lambda x: int(round(float(x))), grasp_line.split(' ')[:8]))
                y1, x1, y2, x2, y3, x3, y4, x4 = list(map(lambda x: int(round(float(x))), grasp_line.split(' ')[:8]))
                try:
                    gr = np.array([
                        [x1, y1],
                        [x2, y2],
                        [x3, y3],
                        [x4, y4],
                    ])

                    grs.append(GraspRectangle(gr))

                except ValueError:
                    # Some files contain weird values.
                    continue
        
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Jacquard dataset file.
        :param fname: Path to file.
        :param scale: Scale to apply (e.g. if resizing images)
        :return: GraspRectangles()
        """
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    @classmethod
    def load_from_grasp_anything_file(cls, fname, scale=1.0):
        """
        Load grasp rectangles from a Grasp-Anything dataset grasp file.
        :param fname: Path to text file.
        :return: GraspRectangles()
        """
        grs = None
        with open(fname, 'rb') as f:
            pos_grasps = torch.load(f)
        # add_fn = fname.replace("positive_grasp", "negative_grasp")
        # with open(fname, 'rb') as f:
        #     neg_grasps = torch.load(f)
        # grasps = torch.cat((pos_grasps, neg_grasps), dim=0).tolist()
        grasps = pos_grasps.tolist()
        grs = list(map(lambda x: _grasp_anything_format(x), grasps))

        grs = cls(grs)
        grs.scale(scale)
        return grs
    
    def append(self, gr):
        """
        Add a grasp rectangle to this GraspRectangles object
        :param gr: GraspRectangle
        """
        self.grs.append(gr)

    def copy(self):
        """
        :return: A deep copy of this object and all of its GraspRectangles.
        """
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):
        """
        Draw all GraspRectangles on a matplotlib plot.
        :param ax: (optional) existing axis
        :param shape: (optional) Plot shape if no existing axis
        """
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        """
        Plot all GraspRectangles as solid rectangles in a numpy array, e.g. as network training data.
        :param shape: output shape
        :param position: If True, Q output will be produced
        :param angle: If True, Angle output will be produced
        :param width: If True, Width output will be produced
        :return: Q, Angle, Width outputs (or None)
        """
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for gr in self.grs:
            rr, cc = gr.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = gr.angle
            if width:
                width_out[rr, cc] = gr.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        """
        Convert all GraspRectangles to a single array.
        :param pad_to: Length to 0-pad the array along the first dimension
        :return: Nx4x2 numpy array
        """
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int32)

    @property
    def center(self):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        points = [gr.points for gr in self.grs]
        return np.mean(np.vstack(points), axis=0).astype(np.int32)


class GraspRectangle:
    """
    Representation of a grasp in the common "Grasp Rectangle" format.
    """

    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        """
        :return: Angle of the grasp to the horizontal.
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi / 2) % np.pi - np.pi / 2

    @property
    def as_grasp(self):
        """
        :return: GraspRectangle converted to a Grasp
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        :return: Rectangle center point
        """
        return self.points.mean(axis=0).astype(np.int32)

    @property
    def length(self):
        """
        :return: Rectangle length (i.e. along the axis of the grasp)
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        :return: Rectangle width (i.e. perpendicular to the axis of the grasp)
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        :param shape: Output Shape
        :return: Indices of pixels within the grasp rectangle polygon.
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        :param shape: Output shape
        :return: Indices of pixels within the centre thrid of the grasp rectangle.
        """
        return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)

    def iou(self, gr, angle_threshold=np.pi / 6):
        """
        Compute IoU with another grasping rectangle
        :param gr: GraspingRectangle to compare
        :param angle_threshold: Maximum angle difference between GraspRectangles
        :return: IoU between Grasp Rectangles
        """
        if abs((self.angle - gr.angle + np.pi / 2) % np.pi - np.pi / 2) > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(gr.points[:, 0], gr.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        """
        :return: Copy of self.
        """
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        """
        Offset grasp rectangle
        :param offset: array [y, x] distance to offset
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int32)

    def scale(self, factor):
        """
        :param factor: Scale grasp rectangle by factor
        """
        if factor == 1.0:
            return
        self.points *= factor
        
    
    def resize(self, current, output):
        copy_points = copy.copy(self.points)
        for idx in range(len(copy_points)):
            pt = copy_points[idx] * float(output/current)
            pt = np.int32(pt)
            copy_points[idx] = pt
            
        self.points = copy_points

    def plot(self, ax, color=None):
        """
        Plot grasping rectangle.
        :param ax: Existing matplotlib axis
        :param color: matplotlib color code (optional)
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int32)


class Grasp:
    """
    A Grasp represented by a center pixel, rotation angle and gripper width (length)
    """

    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_gr(self):
        """
        Convert to GraspRectangle
        :return: GraspRectangle representation of grasp.
        """
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return GraspRectangle(np.array(
            [
                [y1 - self.width / 2 * xo, x1 - self.width / 2 * yo],
                [y2 - self.width / 2 * xo, x2 - self.width / 2 * yo],
                [y2 + self.width / 2 * xo, x2 + self.width / 2 * yo],
                [y1 + self.width / 2 * xo, x1 + self.width / 2 * yo],
            ]
        ).astype(np.float32))

    def max_iou(self, grs):
        """
        Return maximum IoU between self and a list of GraspRectangles
        :param grs: List of GraspRectangles
        :return: Maximum IoU with any of the GraspRectangles
        """
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot Grasp
        :param ax: Existing matplotlib axis
        :param color: (optional) color
        """
        self.as_gr.plot(ax, color)

    def to_jacquard(self, scale=1):
        """
        Output grasp in "Jacquard Dataset Format" (https://jacquard.liris.cnrs.fr/database.php)
        :param scale: (optional) scale to apply to grasp
        :return: string in Jacquard format
        """
        # Output in jacquard format.
        return '%0.2f;%0.2f;%0.2f;%0.2f;%0.2f' % (
            self.center[1] * scale, self.center[0] * scale, -1 * self.angle * 180 / np.pi, self.length * scale,
            self.width * scale)


def detect_grasps(q_img, ang_img, width_img=None, no_grasps=1):
    """
    Detect grasps in a network output.
    :param q_img: Q image network output
    :param ang_img: Angle image network output
    :param width_img: (optional) Width image network output
    :param no_grasps: Max number of grasps to return
    :return: list of Grasps
    """
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    
    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length / 2

        grasps.append(g)

    return grasps
```

### 3.image_utils.py

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import rotate, resize

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class Image:
    """
    Wrapper around an image with some convenient functions.
    """
    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname):
        return cls(imread(fname))

    def copy(self):
        """
        :return: Copy of self.
        """
        return self.__class__(self.img.copy())

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the image to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped image to this size
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        """
        :return: Cropped copy of the image.
        """
        i = self.copy()
        i.crop(*args, **kwargs)
        return i
    
    def normalise(self):
        """
        Normalise the image by converting to float [0,1] and zero-centering
        """
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]
        
        self.img = (self.img - pixel_mean) / pixel_std

    def resize(self, shape):
        """
        Resize image to shape.
        :param shape: New shape.
        """
        if self.img.shape == shape:
            return
        self.img = resize(self.img, shape, preserve_range=True).astype(self.img.dtype)

    def resized(self, *args, **kwargs):
        """
        :return: Resized copy of the image.
        """
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        """
        Rotate the image.
        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle/np.pi*180, center=center, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def rotated(self, *args, **kwargs):
        """
        :return: Rotated copy of image.
        """
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        """
        Plot the image
        :param ax: Existing matplotlib axis (optional)
        :param kwargs: kwargs to imshow
        """
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        """
        :return: Zoomed copy of the image.
        """
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class DepthImage(Image):
    def __init__(self, img):
        super().__init__(img)

    @classmethod
    def from_pcd(cls, pcd_filename, shape, default_filler=0, index=None):
        """
            Create a depth image from an unstructured PCD file.
            If index isn't specified, use euclidean distance, otherwise choose x/y/z=0/1/2
        """
        img = np.zeros(shape)
        if default_filler != 0:
            img += default_filler

        with open(pcd_filename) as f:
            for l in f.readlines():
                ls = l.split()

                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // shape[1]
                c = i % shape[1]

                if index is None:
                    x = float(ls[0])
                    y = float(ls[1])
                    z = float(ls[2])

                    img[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                else:
                    img[r, c] = float(ls[index])

        return cls(img/1000.0)

    @classmethod
    def from_tiff(cls, fname):
        return cls(imread(fname))

    def inpaint(self, missing_value=0):
        """
        Inpaint missing values in depth image.
        :param missing_value: Value to fill in teh depth image.
        """
        # cv2 inpainting doesn't handle the border properly
        # https://stackoverflow.com/questions/25974033/inpainting-depth-map-still-a-black-image-border
        self.img = cv2.copyMakeBorder(self.img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (self.img == missing_value).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(self.img).max()
        self.img = self.img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        self.img = cv2.inpaint(self.img, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        self.img = self.img[1:-1, 1:-1]
        self.img = self.img * scale

    def gradients(self):
        """
        Compute gradients of the depth image using Sobel filtesr.
        :return: Gradients in X direction, Gradients in Y diretion, Magnitude of XY gradients.
        """
        grad_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, borderType=cv2.BORDER_DEFAULT)
        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

        return DepthImage(grad_x), DepthImage(grad_y), DepthImage(grad)

    def normalise(self):
        """
        Normalise by subtracting the mean and clippint [-1, 1]
        """
        self.img = np.clip((self.img - self.img.mean()), -1, 1)


class Mask:
    """
    Wrapper around an mask with some convenient functions.
    """
    def __init__(self, img):
        self.img = img

    def __getattr__(self, attr):
        # Pass along any other methods to the underlying ndarray
        return getattr(self.img, attr)

    @classmethod
    def from_file(cls, fname):
        return cls(imread(fname))
    
    @classmethod
    def from_npy_file(cls, fname):
        return cls(np.load(fname).astype(np.float32))
    
    
    def random(self):
        obj_ids = np.unique(self.img)[1:]
        rand_idx = np.random.choice(obj_ids, size=1)
        
        copy_img = self.img.copy()
        copy_img = (copy_img == rand_idx[:, None, None])
        self.img = np.array(copy_img[0], dtype=np.float32)
  

    def copy(self):
        """
        :return: Copy of self.
        """
        return self.__class__(self.img.copy())

    def crop(self, top_left, bottom_right, resize=None):
        """
        Crop the mask to a bounding box given by top left and bottom right pixels.
        :param top_left: tuple, top left pixel.
        :param bottom_right: tuple, bottom right pixel
        :param resize: If specified, resize the cropped mask to this size
        """
        self.img = self.img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        if resize is not None:
            self.resize(resize)

    def cropped(self, *args, **kwargs):
        """
        :return: Cropped copy of the mask.
        """
        i = self.copy()
        i.crop(*args, **kwargs)
        return i
    
    def normalise(self):
        """
        Normalise the mask by converting to float [0,1]
        """
        self.img = self.img / 255.0
    
    def resize(self, shape):
        """
        Resize mask to shape.
        :param shape: New shape.
        """
        if self.img.shape == shape:
            return
        self.img = cv2.resize(self.img, shape, interpolation=cv2.INTER_NEAREST)

    def resized(self, *args, **kwargs):
        """
        :return: Resized copy of the image.
        """
        i = self.copy()
        i.resize(*args, **kwargs)
        return i

    def rotate(self, angle, center=None):
        """
        Rotate the image.
        :param angle: Angle (in radians) to rotate by.
        :param center: Center pixel to rotate if specified, otherwise image center is used.
        """
        if center is not None:
            center = (center[1], center[0])
        self.img = rotate(self.img, angle/np.pi*180, center=center, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def rotated(self, *args, **kwargs):
        """
        :return: Rotated copy of image.
        """
        i = self.copy()
        i.rotate(*args, **kwargs)
        return i

    def show(self, ax=None, **kwargs):
        """
        Plot the image
        :param ax: Existing matplotlib axis (optional)
        :param kwargs: kwargs to imshow
        """
        if ax:
            ax.imshow(self.img, **kwargs)
        else:
            plt.imshow(self.img, **kwargs)
            plt.show()

    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        sr = int(self.img.shape[0] * (1 - factor)) // 2
        sc = int(self.img.shape[1] * (1 - factor)) // 2
        orig_shape = self.img.shape
        self.img = self.img[sr:self.img.shape[0] - sr, sc: self.img.shape[1] - sc].copy()
        self.img = resize(self.img, orig_shape, mode='symmetric', preserve_range=True).astype(self.img.dtype)

    def zoomed(self, *args, **kwargs):
        """
        :return: Zoomed copy of the image.
        """
        i = self.copy()
        i.zoom(*args, **kwargs)
        return i


class WidthImage(Image):
    """
    A width image is one that describes the desired gripper width at each pixel.
    """
    def zoom(self, factor):
        """
        "Zoom" the image by cropping and resizing.  Also scales the width accordingly.
        :param factor: Factor to zoom by. e.g. 0.5 will keep the center 50% of the image.
        """
        super().zoom(factor)
        self.img = self.img/factor

    def normalise(self):
        """
        Normalise by mapping [0, 150] -> [0, 1]
        """
        self.img = np.clip(self.img, 0, 150.0)/150.0

```

### 4.transform.py

```
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import functional as F, InterpolationMode, transforms as T

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class Normalize(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = image.float()
        image = F.normalize(image,
                            [0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
        
        
        return image, target


class RandomRotate(nn.Module):
    def __init__(
        self,
        low=0,
        high=180,
        image_size = (1024, 1024),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()

        self.low = low
        self.high = high
        self.image_size =image_size
        self.interpolation = interpolation

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        
        self.angle = np.random.randint(low=self.low, high=self.high)
        center = (self.image_size[0]//2, self.image_size[1]//2)
          
        image = F.rotate(image, self.angle, center=center, interpolation=self.interpolation)

        if target is not None:
            if "masks" in target:
                target["masks"] = F.rotate(target["masks"], self.angle, center=center, interpolation=InterpolationMode.NEAREST)
                
            if "grasps" in target:
                radian = self.angle * np.pi / 180
                
                R = np.array([
                    [np.cos(-radian), -1*np.sin(-radian)],
                    [np.sin(-radian), np.cos(-radian)],
                ])
                
                
                for idx in range(len(target["grasps"])):
                    x, y, a, w, h = target["grasps"][idx]
                    
                    point = np.array([x, y]) - np.array(center)
                    x, y = (np.dot(R, point.T)).T + np.array(center)
                    
                    target["grasps"][idx] = torch.tensor([x, y, a-radian, w, h])
       
                
        return image, target
 
    
class RandomZoom(nn.Module):
    def __init__(
        self,
        low=0.8,
        high=1.0,
        image_size = (1024, 1024),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()

        self.low = low
        self.high = high
        
        self.image_size = image_size
        self.interpolation = interpolation

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        _, orig_height, orig_width = F.get_dimensions(image)
        
        self.factor = np.random.randint(low=self.low*10, high=self.high*10)/10
        
        center = np.array((self.image_size[0]//2, self.image_size[1]//2)).astype(int)
        
        sr = int(orig_height * (1 - self.factor)) // 2
        sc = int(orig_width * (1 - self.factor)) // 2
 
        

        image = image[:, sr:orig_height - sr, sc: orig_width - sc]
 
        
        image = F.resize(image, [orig_height, orig_width], interpolation=self.interpolation)
        

        if target is not None:
            if "masks" in target:
                
                target["masks"] = target["masks"][:, sr:orig_height - sr, sc: orig_width - sc]
                target["masks"] = F.resize(target["masks"], [orig_height, orig_width], interpolation=InterpolationMode.NEAREST)
            
            
            if "grasps" in target:
                
                T = np.array([
                        [1/self.factor, 0],
                        [0, 1/self.factor]
                    ])
                
                
                for idx in range(len(target["grasps"])):
                    x, y, a, w, h = target["grasps"][idx]
                    
                    point = np.array([x, y]) - center
                    x, y = (np.dot(T, point.T)).T + center

                    target["grasps"][idx] = torch.tensor([x, y, a, w*(1/self.factor), h*(1/self.factor)])
            
            
        return image, target
    
```

## 1.base_grasp_data.py

```py
# This code is from "https://github.com/WangShaoSUN/grasp-transformer/tree/main"
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torch.nn.functional as F

import cv2
import random


class BaseGraspDataset(torch.utils.data.Dataset):
    """
    An abstract dataset for training GG-CNNs in a common format.
    """
    def __init__(self, output_size=1024, crop_size=224, include_depth=False, include_mask=False, include_rgb=True,
                 include_prompt=False,
                 random_rotate=False, random_zoom=False, input_only=False, grasp_map_split=False, multi_masks=False, 
                 seen=False, train=True):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        """
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.include_depth = include_depth
        self.include_mask = include_mask
        self.include_rgb = include_rgb
        self.include_prompt = include_prompt

        
        self.grasp_map_split = grasp_map_split
        self.multi_masks = multi_masks
        self.seen = seen
        self.crop_size = crop_size
        self.train = train
        

        self.grasp_files = []

        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_depth(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()
    
    def get_mask(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_prompt(self, idx):
        raise NotImplementedError()


    def __getitem__(self, idx, vis=False):
        if self.random_rotate:
            rotations = [0, np.pi/2, 2*np.pi/2, 3*np.pi/2]
            rot = random.choice(rotations)
        else:
            rot = 0.0

        if self.random_zoom:
            zoom_factor = np.random.uniform(0.5, 1.0)
        else:
            zoom_factor = 1.0

        # Load the depth image
        if self.include_depth:
            depth_img = self.get_depth(idx, rot, zoom_factor)

        # Load the mask
        if self.include_mask:
            mask = self.get_mask(idx, rot, zoom_factor)
            mask = torch.as_tensor(mask).unsqueeze(0)
        
        # Load the RGB image
        if self.include_rgb:
            rgb_img = self.get_rgb(idx, rot, zoom_factor)

        # Load the grasps
        bbs = self.get_gtbb(idx, rot, zoom_factor)

        if self.include_prompt:
            prompt = self.get_prompt(idx)
            
        ocid = True
        if ocid:
            pos_img, ang_img, width_img = bbs.draw((self.crop_size, self.crop_size))
            pos_img = cv2.resize(pos_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            ang_img = cv2.resize(ang_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
            width_img = cv2.resize(width_img, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)

        else:
            pos_img, ang_img, width_img = bbs.draw((self.output_size, self.output_size))
        # print(np.unique(width_img))
        width_img = np.clip(width_img, 0.0, self.output_size / 2) / (self.output_size / 2)

        if self.include_depth and self.include_rgb:
            x = self.numpy_to_torch(
                np.concatenate(
                    (np.expand_dims(depth_img, 0),
                     rgb_img),
                    0
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        cos = self.numpy_to_torch(np.cos(2*ang_img))
        sin = self.numpy_to_torch(np.sin(2*ang_img))
        width = self.numpy_to_torch(width_img)
        

        if self.grasp_map_split:
            
            obj_ids = np.unique(mask)[1:]
            filtered_ids = []
            th = 5000
            for ids in obj_ids:
                i_mask = (mask.clone() == ids)
                m = pos * i_mask
                m = torch.sum(m)
                if m.item() < th:
                    continue
                filtered_ids.append(ids)
            # print(len(filtered_ids))

            if self.multi_masks:
                num_samples = np.random.randint(1, len(filtered_ids)+1)
                rand_idxs = np.random.choice(filtered_ids, size=num_samples)
                
                semantic_mask = torch.zeros_like(mask, dtype=torch.float32)

                for rand_idx in rand_idxs: 
                    semantic_mask += (mask.clone() == rand_idx) * 1.0

                mask = semantic_mask
            else:
                rand_idx = np.random.choice(filtered_ids, size=1)[0]
                mask = (mask.clone() == rand_idx) * 1.0
            

        if self.grasp_map_split:
            pos[mask==0.0] = 0
            cos[mask==0.0] = 1
            sin[mask==0.0] = 0
            width[mask==0.0] = 0
            
            
        grasps = [pos, cos, sin, width]
        
        if vis:
            # raw_image = Image.open(self.rgb_files[idx]).convert("RGB")
            if self.include_prompt:
                return x, mask, grasps, idx, rot, zoom_factor, bbs, prompt
            else: 
                return x, mask, grasps, idx, rot, zoom_factor, bbs
        else:
            if self.include_prompt:
                return x, mask, grasps, idx, rot, zoom_factor, prompt
            else:
                return x, mask, grasps, idx, rot, zoom_factor
        
        # return x, (pos, cos, sin, width), idx, rot, zoom_factor
        
    def __len__(self):
        return len(self.grasp_files)
```

## 2.grasp_anything_data.py

```
import glob
import os
import re

import pickle
import torch
import numpy as np

from .base_grasp_data import BaseGraspDataset

from .utils import grasp_utils as gu
from .utils import image_utils as iu



class GraspAnythingDataset(BaseGraspDataset):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, root, start=0.0, end=1.0,  ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnythingDataset, self).__init__(**kwargs)

        grasp_files = glob.glob(os.path.join(root, 'grasp_label_positive', '*.pt'))


        if self.seen:
            with open(os.path.join('split/grasp-anything/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)
            
            grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, grasp_files))
            split = int(np.floor(0.9 * len(grasp_files)))
            if self.train:
                self.grasp_files = grasp_files[:split]
                
            else:
                self.grasp_files = grasp_files[split:]
        
        
        else:
            with open(os.path.join('split/grasp-anything/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            self.grasp_files = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, grasp_files))


        l = len(self.grasp_files)

        self.grasp_files.sort()
   
        # self.grasp_files = self.grasp_files[int(l*start):int(l*end)]
        
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(root))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        gtbbs = gu.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        # gtbbs.resize(self.crop_size, self.output_size)
   
        return gtbbs


    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_file = re.sub(r"_\d+\.pt", ".jpg", self.grasp_files[idx])
        rgb_file = rgb_file.replace("grasp_label_positive", "image")
        rgb_img = iu.Image.from_file(rgb_file)
        # rgb_img = image.Image.mask_out_image(rgb_img, mask_img)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        rgb_img.img = rgb_img.img[...,::-1]
        
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
    
    
    def get_mask(self, idx, rot=0, zoom=1.0):
        mask_file = self.grasp_files[idx].replace("grasp_label_positive", "mask").replace(".pt", ".npy")
        mask_image = iu.Mask.from_npy_file(mask_file)
        mask_image.rotate(rot)
        mask_image.zoom(zoom)
        mask_image.resize((self.output_size, self.output_size))
        return mask_image.img

```

## 3.jacquard_data.py

```
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random

import numpy as np
from PIL import Image
import pickle


from .base_grasp_data import BaseGraspDataset

from .utils import grasp_utils as gu
from .utils import image_utils as iu


class JacquardDataset(BaseGraspDataset):
    """
    Dataset wrapper for the Jacquard dataset.
    """
    def __init__(self, root, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: Jacquard Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(JacquardDataset, self).__init__(**kwargs)

        import glob
        # graspf = glob.glob(os.path.join(file_path, '*', '*_grasps.txt'))
        #graspf = glob.glob('/SSDc/jongwon_kim/Datasets/Jacquard_Dataset' + '/*/*/' + '*_grasps.txt')
        graspf = glob.glob(os.path.join(root, '*', '*_grasps.txt'))
        graspf.sort()
        l = len(graspf)
        print("len jaccquard:", l)


        if self.seen:
            with open(os.path.join('split/jacquard/seen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            graspf = list(filter(lambda x: x.split('.')[0].split('/')[-1].split("_")[0] + "_" + x.split('.')[0].split('/')[-1].split("_")[1] in idxs, graspf))
            # graspf = list(filter(lambda x: x.split('/')[-1].split('.')[0] in idxs, graspf))
            
            split = int(np.floor(0.9 * len(graspf)))
            if self.train:
                graspf = graspf[:split]
                
            else:
                graspf = graspf[split:]
        
        else:
            with open(os.path.join('split/jacquard/unseen.obj'), 'rb') as f:
                idxs = pickle.load(f)

            graspf = list(filter(lambda x: x.split('.')[0].split('/')[-1].split("_")[0] + "_" + x.split('.')[0].split('/')[-1].split("_")[1] in idxs, graspf))


        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(root))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        fl = len(graspf)
        # print("len filtered jaccquard:", fl)


        depthf = [f.replace('grasps.txt', 'perfect_depth.tiff') for f in graspf]
        rgbf = [f.replace('perfect_depth.tiff', 'RGB.png') for f in depthf]
        maskf = [f.replace('perfect_depth.tiff', 'mask.png') for f in depthf]


        self.grasp_files = graspf
        self.depth_files = depthf
        self.rgb_files = rgbf
        self.mask_files = maskf

        # when want to use length
        # self.grasp_files = graspf[int(l*start):int(l*end)]
        # self.depth_files = depthf[int(l*start):int(l*end)]
        # self.rgb_files = rgbf[int(l*start):int(l*end)]
        # self.mask_files = maskf[int(l*start):int(l*end)]


        if self.seen:
            pass
        else:
            pass
        
        
    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = gu.GraspRectangles.load_from_jacquard_file(self.grasp_files[idx], scale=self.output_size / 1024.0)
        c = self.output_size//2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = iu.DepthImage.from_tiff(self.depth_files[idx])
        depth_img.rotate(rot)
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = iu.Image.from_file(self.rgb_files[idx])
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

    def get_mask(self, idx, rot=0, zoom=1.0): 
        mask_image = iu.Mask.from_file(self.mask_files[idx])
        mask_image.rotate(rot)
        mask_image.zoom(zoom)
        mask_image.resize((self.output_size, self.output_size))
        mask_image.normalise()
        return mask_image.img

    def get_jname(self, idx):
        return '_'.join(self.grasp_files[idx].split(os.sep)[-1].split('_')[:-1])


```

## 4.load_data.py

```
from .jacquard_data import JacquardDataset
from .ocid_grasp_data import OCIDGraspDataset
from .grasp_anything_data import GraspAnythingDataset
from .grasp_anywhere_data import GraspAnywhereDataset


def load_data(args):
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, 
                                        random_rotate=True, random_zoom=True,
                                        start=0.0, end=0.9, seen=args.seen)
        test_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False)

    # TODO: change args when evaluate graspsam
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.01)
        test_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.99, end=1.0)

    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=args.seen)
        test_dataset = GraspAnythingDataset(root=root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False)
    
    elif args.dataset_name == "grasp_anywhere":
        root="/ailab_mat/dataset/Grasp-Anything/"
        add_d_root="/SSDc/Grasp-Anything++/"
        train_dataset = GraspAnywhereDataset(root=root, include_mask=True, include_prompt=True, start=0.0, end=0.05, add_d_root=add_d_root)
        test_dataset = GraspAnywhereDataset(root=root, include_mask=True, include_prompt=True, start=0.99, end=1.0, add_d_root=add_d_root)    


    else:
        raise("dataset name should be in [jacquard, vmrd, ocid-grasp, cornell, grasp-anything, graspnet]")

    return train_dataset, test_dataset
```

## 5.load_dataset.py

```
import torch
from torch.utils.data.distributed import DistributedSampler

from data.jacquard_data import JacquardDataset
from data.ocid_grasp_data import OCIDGraspDataset
from data.grasp_anything_data import GraspAnythingDataset


def get_dataset(args):
    
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.0, end=0.9)
        test_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.9, end=1.0)
    
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.9)
        test_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.9, end=1.0)
    
    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.0, end=0.9)
        test_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.9, end=1.0)
        
    print("train_dataset size : {}".format(train_dataset.__len__()))
    print("test_dataset size : {}".format(test_dataset.__len__()))
    
    
    return train_dataset, test_dataset


def get_train_loader(args):
    if args.dataset_name == "jacquard":
        root="/SSDc/jongwon_kim/Datasets/Jacquard_Dataset/"
        train_dataset = JacquardDataset(root=root, crop_size=1024, include_mask=True, start=0.0, end=0.9)
    
    elif args.dataset_name == "ocid":
        root="/ailab_mat/dataset/OCID_grasp/"
        train_dataset = OCIDGraspDataset(root=root, include_mask=True, grasp_map_split=True, start=0.0, end=0.9)
    
    elif args.dataset_name == "grasp_anything":
        root="/ailab_mat/dataset/Grasp-Anything/"
        train_dataset = GraspAnythingDataset(root=root, include_mask=True, start=0.0, end=0.9)
        
    print("train_dataset size : {}".format(train_dataset.__len__()))
    
    
    if args.ddp:
        train_sampler = DistributedSampler(train_dataset)
        
        num_workers_ = 1
        if(args.batch_size>1):
            num_workers_ = 2
        if(args.batch_size>4):
            num_workers_ = 4
        if(args.batch_size>8):
            num_workers_ = 8
            
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                        num_workers=num_workers_, pin_memory=True, sampler=train_sampler)

    else:
        train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=True, 
                        num_workers=4, pin_memory=False)
        
    return train_loader

```

# 2.model

## efficient_sam:

### 1.build_efficient_sam.py

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam
from .efficient_sam import build_efficient_sam_w_ad

def build_efficient_sam_vitt(checkpoint=None):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint,
    ).eval()


def build_efficient_sam_vits(checkpoint=None):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint,
    ).eval()

def build_efficient_sam_vitt_w_ad(checkpoint=None):
    return build_efficient_sam_w_ad(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint=checkpoint,
    ).eval()

def build_efficient_sam_vits_w_ad(checkpoint=None):
    return build_efficient_sam_w_ad(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint=checkpoint,
    ).eval()

eff_sam_model_registry = {
    "eff_vit_t": build_efficient_sam_vitt,
    "eff_vit_s": build_efficient_sam_vits,

    "eff_vit_t_w_ad": build_efficient_sam_vitt_w_ad,
    "eff_vit_s_w_ad": build_efficient_sam_vits_w_ad,
}
```

### 2.efficient_sam_decoder.py

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBlock


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.invalid_points = nn.Embedding(1, embed_dim)
        self.point_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_top_left_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_bottom_right_embeddings = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        invalid_label_ids = torch.eq(labels, -1)[:,:,None]
        point_label_ids = torch.eq(labels, 1)[:,:,None]
        topleft_label_ids = torch.eq(labels, 2)[:,:,None]
        bottomright_label_ids = torch.eq(labels, 3)[:,:,None]
        point_embedding = point_embedding + self.invalid_points.weight[:,None,:] * invalid_label_ids
        point_embedding = point_embedding + self.point_embeddings.weight[:,None,:] * point_label_ids
        point_embedding = point_embedding + self.bbox_top_left_embeddings.weight[:,None,:] * topleft_label_ids
        point_embedding = point_embedding + self.bbox_bottom_right_embeddings.weight[:,None,:] * bottomright_label_ids
        return point_embedding

    def forward(
        self,
        coords,
        labels,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points: A tensor of shape [B, 2]
          labels: An integer tensor of shape [B] where each element is 1,2 or 3.

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
        """
        return self._embed_points(coords, labels)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats))
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones([h, w], device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        upscaling_layer_dims: List[int],
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1
        else:
            self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim

        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings (the batch dimension is broadcastable).
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """

        (
            batch_size,
            max_num_queries,
            sparse_embed_dim_1,
            sparse_embed_dim_2,
        ) = sparse_prompt_embeddings.shape

        (
            _,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        ) = image_embeddings.shape

        # Tile the image embedding for all queries.
        image_embeddings_tiled = torch.tile(
            image_embeddings[:, None, :, :, :], [1, max_num_queries, 1, 1, 1]
        ).view(
            batch_size * max_num_queries,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        )
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_embed_dim_1, sparse_embed_dim_2
        )
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings_tiled,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )
        if multimask_output and self.num_multimask_outputs > 1:
            return masks[:, 1:, :], iou_pred[:, 1:]
        else:
            return masks[:, :1, :], iou_pred[:, :1]

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape
        hs, src = self.transformer(image_embeddings, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)

        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)
        hyper_in_list: List[torch.Tensor] = []
        for i, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

```

### 3.efficient_sam_encoder.py

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.segment_anything.modeling.reins import Reins, LoRAReins
from model.segment_anything.modeling.rein_utils import*

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        embed_dim,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias,
        qk_scale=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        window_size=0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
        )

        self.window_size = window_size

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@torch.jit.export
def get_abs_pos(
    abs_pos: torch.Tensor, has_cls_token: bool, hw: List[int]
) -> torch.Tensor:
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h = hw[0]
    w = hw[1]
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


# Image encoder for efficient SAM.
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        patch_embed_dim: int,
        normalization_type: str,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        neck_dims: List[int],
        act_layer: Type[nn.Module],
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            patch_embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            act_layer (nn.Module): Activation layer.
        """
        super().__init__()

        self.patch_size = patch_size

        self.img_size = img_size
        self.image_embedding_size = img_size // ((patch_size if patch_size > 0 else 1))
        self.transformer_output_dim = ([patch_embed_dim] + neck_dims)[-1]
        self.pretrain_use_cls_token = True
        pretrain_img_size = 224
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, patch_embed_dim)
        # Initialize absolute positional embedding with pretrain image size.
        num_patches = (pretrain_img_size // patch_size) * (
            pretrain_img_size // patch_size
        )
        num_positions = num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, patch_embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            vit_block = Block(patch_embed_dim, num_heads, mlp_ratio, True, window_size=0)
            self.blocks.append(vit_block)
        self.neck = nn.Sequential(
            nn.Conv2d(
                patch_embed_dim,
                neck_dims[0],
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
            nn.Conv2d(
                neck_dims[0],
                neck_dims[0],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(neck_dims[0]),
        )

    # TODO: check interm_embeddings
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"

        interm_embeddings=[]
        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        
        interm_embeddings=[]
        for blk in self.blocks:
            x = blk(x)
            interm_embeddings.append(x.view(x.shape[0], 64, 64, -1))
                
        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        # print(interm_embeddings[0].shape)
        # print(len(interm_embeddings))
        # exit()
        return x, interm_embeddings


# TODO: change rein params
class ImageEncoderViTReinAdapter(ImageEncoderViT):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins = Reins(
            num_layers = 12,
            embed_dims = 192,
            # embed_dims = 384,
            patch_size = 16,
        )

    def forward(self, x):
        assert (
            x.shape[2] == self.img_size and x.shape[3] == self.img_size
        ), "input image size must match self.img_size"
        interm_embeddings=[]

        B, C, H, W = x.shape

        x = self.patch_embed(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        x = x + get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, [x.shape[1], x.shape[2]]
        )
        num_patches = x.shape[1]
        assert x.shape[2] == num_patches
        x = x.reshape(x.shape[0], num_patches * num_patches, x.shape[3])
        
        interm_embeddings=[]
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            interm_embeddings.append(x.view(x.shape[0], 64, 64, -1))
            
            # x: [16, 4096, 192]
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=False,
            )

        x = x.reshape(x.shape[0], num_patches, num_patches, x.shape[2])
        x = self.neck(x.permute(0, 3, 1, 2))
        # print(interm_embeddings[0].shape)
        # print(len(interm_embeddings))
        # exit()
        return x, interm_embeddings



    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])
        print("set rein as training mode")

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
```

### 4.efficient_sam.py

```
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, List, Tuple, Type

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from .efficient_sam_decoder import MaskDecoder, PromptEncoder
from .efficient_sam_encoder import ImageEncoderViT
from .efficient_sam_encoder import ImageEncoderViTReinAdapter
from .two_way_transformer import TwoWayAttentionBlock, TwoWayTransformer

class EfficientSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        decoder_max_num_input_points: int,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.decoder_max_num_input_points = decoder_max_num_input_points
        self.mask_decoder = mask_decoder
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False
        )

    @torch.jit.export
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks given image embeddings and prompts. This only runs the decoder.

        Arguments:
          image_embeddings: A tensor of shape [B, C, H, W] or [B*max_num_queries, C, H, W]
          batched_points: A tensor of shape [B, max_num_queries, num_pts, 2]
          batched_point_labels: A tensor of shape [B, max_num_queries, num_pts]
        Returns:
          A tuple of two tensors:
            low_res_mask: A tensor of shape [B, max_num_queries, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """

        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        return output_masks, iou_predictions

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> torch.Tensor:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
        Returns:
          List of image embeddings each of of shape [B, C(i), H(i), W(i)].
          The last embedding corresponds to the final layer.
        """
        batched_images = self.preprocess(batched_images)
        return self.image_encoder(batched_images)

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        scale_to_original_image_size: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_images: A tensor of shape [B, 3, H, W]
          batched_points: A tensor of shape [B, num_queries, max_num_pts, 2]
          batched_point_labels: A tensor of shape [B, num_queries, max_num_pts]

        Returns:
          A list tuples of two tensors where the ith element is by considering the first i+1 points.
            low_res_mask: A tensor of shape [B, 256, 256] of predicted masks
            iou_predictions: A tensor of shape [B, max_num_queries] of estimated IOU scores
        """
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings = self.get_image_embeddings(batched_images)
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h if scale_to_original_image_size else -1,
            output_w=input_w if scale_to_original_image_size else -1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam(encoder_patch_embed_dim, encoder_num_heads, checkpoint=None):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViT(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"])
    return sam


def build_efficient_sam_w_ad(encoder_patch_embed_dim, encoder_num_heads, checkpoint=None):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    decoder_max_num_input_points = 6
    decoder_transformer_depth = 2
    decoder_transformer_mlp_dim = 2048
    decoder_num_heads = 8
    decoder_upscaling_layer_dims = [64, 32]
    num_multimask_outputs = 3
    iou_head_depth = 3
    iou_head_hidden_dim = 256
    activation = "gelu"
    normalization_type = "layer_norm"
    normalize_before_activation = False

    assert activation == "relu" or activation == "gelu"
    if activation == "relu":
        activation_fn = nn.ReLU
    else:
        activation_fn = nn.GELU

    image_encoder = ImageEncoderViTReinAdapter(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type=normalization_type,
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=activation_fn,
    )

    image_embedding_size = image_encoder.image_embedding_size
    encoder_transformer_output_dim = image_encoder.transformer_output_dim

    sam = EfficientSam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=encoder_transformer_output_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
        ),
        decoder_max_num_input_points=decoder_max_num_input_points,
        mask_decoder=MaskDecoder(
            transformer_dim=encoder_transformer_output_dim,
            transformer=TwoWayTransformer(
                depth=decoder_transformer_depth,
                embedding_dim=encoder_transformer_output_dim,
                num_heads=decoder_num_heads,
                mlp_dim=decoder_transformer_mlp_dim,
                activation=activation_fn,
                normalize_before_activation=normalize_before_activation,
            ),
            num_multimask_outputs=num_multimask_outputs,
            activation=activation_fn,
            normalization_type=normalization_type,
            normalize_before_activation=normalize_before_activation,
            iou_head_depth=iou_head_depth - 1,
            iou_head_hidden_dim=iou_head_hidden_dim,
            upscaling_layer_dims=decoder_upscaling_layer_dims,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        sam.load_state_dict(state_dict["model"], strict=False)
    return sam
```

### 5.mlp.py

```
from typing import Type

from torch import nn


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLPBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(n, k), act())
            for n, k in zip([input_dim] + h, [hidden_dim] * num_layers)
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)

```

### 6.two_way_transformer.py

```
import math
from typing import Tuple, Type
import torch
from torch import nn, Tensor
from .mlp import MLPBlock




class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            curr_layer = TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                normalize_before_activation=normalize_before_activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            self.layers.append(curr_layer)

        self.final_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """

        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for idx, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module],
        normalize_before_activation: bool,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = AttentionForTwoWayAttentionBlock(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(
            embedding_dim,
            mlp_dim,
            embedding_dim,
            1,
            activation,
        )

        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = AttentionForTwoWayAttentionBlock(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if not self.skip_first_layer_pe:
            queries = queries + query_pe
        attn_out = self.self_attn(q=queries, k=queries, v=queries)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class AttentionForTwoWayAttentionBlock(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."
        self.c_per_head = self.internal_dim / num_heads
        self.inv_sqrt_c_per_head = 1.0 / math.sqrt(self.c_per_head)

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # The fan_out is incorrect, but matches pytorch's initialization
        # for which qkv is a single 3*embedding_dim x embedding_dim matrix
        fan_in = self.embedding_dim
        fan_out = 3 * self.internal_dim
        # Xavier uniform with our custom fan_out
        bnd = math.sqrt(6 / (fan_in + fan_out))
        nn.init.uniform_(self.q_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.k_proj.weight, -bnd, bnd)
        nn.init.uniform_(self.v_proj.weight, -bnd, bnd)
        # out_proj.weight is left with default initialization, like pytorch attention
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn * self.inv_sqrt_c_per_head
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out

```

## segment_anything

## 1.build_grasp_sam.py

```
import torch
import torch.nn as nn
from collections import OrderedDict

from model.segment_anything import sam_model_registry
from model.efficient_sam.build_efficient_sam import eff_sam_model_registry

from model.segment_anything.modeling import MaskDecoder
from model.segment_anything.modeling import TwoWayTransformer

from model.efficient_sam.efficient_sam_decoder import MaskDecoder as EffMaskDecoder
from model.efficient_sam.two_way_transformer import TwoWayTransformer as EffTwoWayTransformer


def load_sam_encoder(sam_encoder_type, checkpoint_dict, adapter):
    checkpoint_path = checkpoint_dict[sam_encoder_type]
    
    if "eff" in sam_encoder_type:
        print("efficient sam encoder is loading")
        sam = eff_sam_model_registry[sam_encoder_type](checkpoint=checkpoint_path)
    
    else:
        print("default sam encoder is loading")
        sam = sam_model_registry[sam_encoder_type](checkpoint=checkpoint_path)
      
    image_encoder  = sam.image_encoder
    prompt_encoder = sam.prompt_encoder
    
    return image_encoder, prompt_encoder


def load_sam_decoder(sam_encoder_type, checkpoint_dict):
    checkpoint_path = checkpoint_dict[sam_encoder_type]
    
    if "eff" in sam_encoder_type:
        print("efficient sam decoder is loading")
        print(sam_encoder_type)
        transformer = EffTwoWayTransformer(
                                    depth=2,
                                    embedding_dim=256,
                                    mlp_dim=2048,
                                    num_heads=8,
                                    activation=nn.GELU,
                                    normalize_before_activation= False,
                                    )
        
        decoder = EffMaskDecoder(transformer_dim=256,
                    transformer=transformer,
                    num_multimask_outputs=3,
                    activation=nn.GELU,
                    iou_head_depth= 2,# eff sam's decoder depth is 2
                    iou_head_hidden_dim= 256,
                    normalization_type = "layer_norm",
                    normalize_before_activation=False,
                    upscaling_layer_dims=[64, 32]
                    )
        

        state_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
   
        for k, v in state_dict["model"].items():
            # print(k)
            if 'mask_decoder' in k:
                # print(k[13:])
                new_state_dict[k[13:]] = v
        
        decoder.load_state_dict(new_state_dict)

    else:
        print("default sam decoder is loading")
        
        transformer = TwoWayTransformer(
                                    depth=2,
                                    embedding_dim=256,
                                    mlp_dim=2048,
                                    num_heads=8,
                                    activation=nn.GELU,
                                    )
        
        decoder = MaskDecoder(transformer_dim=256,
                    transformer=transformer,
                    num_multimask_outputs=3,
                    activation=nn.GELU,
                    iou_head_depth= 3,
                    iou_head_hidden_dim= 256)
        
        
        if "vit_t" in  sam_encoder_type:
    
            state_dict = torch.load(checkpoint_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'mask_decoder' in k:
                    new_state_dict[k[13:]] = v
                    
            decoder.load_state_dict(new_state_dict)
            
        else:
            state_dict = torch.load(checkpoint_path)
            decoder.load_state_dict(state_dict)
            
    return decoder
    
    
def build_grasp_sam(sam_encoder_type="vit_t", 
                    freeze_encoder=True, freeze_decoder=True, freeze_prompt_encoder=True,
                    adapter=False):
    
    assert sam_encoder_type in ["vit_b", "vit_l", "vit_h", "vit_t", "vit_t_w_ad", 
                                "eff_vit_t", "eff_vit_s", "eff_vit_t_w_ad", "eff_vit_s_w_ad"]
    checkpoint_dict = {
                        'vit_t': "pretrained_checkpoint/mobile_sam.pt",
                        "vit_b":"pretrained_checkpoint/sam_vit_b_01ec64.pth",
                        "vit_l":"pretrained_checkpoint/sam_vit_l_0b3195.pth",
                        'vit_h':"pretrained_checkpoint/sam_vit_h_4b8939.pth",

                        'vit_t_w_ad':"pretrained_checkpoint/mobile_sam.pt",
                        
                        "eff_vit_t":"pretrained_checkpoints/efficient_sam/efficient_sam_vitt.pt",
                        "eff_vit_s":"pretrained_checkpoints/efficient_sam/efficient_sam_vits.pt",

                        "eff_vit_t_w_ad":"pretrained_checkpoints/efficient_sam/efficient_sam_vitt.pt",
                        "eff_vit_s_w_ad":"pretrained_checkpoints/efficient_sam/efficient_sam_vits.pt",

                        }
    
    image_encoder, prompt_encoder = load_sam_encoder(sam_encoder_type, checkpoint_dict, adapter)
    mask_decoder                  = load_sam_decoder(sam_encoder_type, checkpoint_dict)
    
    if freeze_encoder:
        for param in image_encoder.parameters():
            param.requires_grad = False
                
    if freeze_prompt_encoder:
        for param in prompt_encoder.parameters():
            param.requires_grad = False
            
    if freeze_decoder:
        for param in mask_decoder.parameters():
            param.requires_grad = False

    return image_encoder, prompt_encoder, mask_decoder


```

## 2.planar_grasp_sam.py

```
import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple

from model.utils import LayerNorm2d, MLP, masks_sample_points, masks_to_boxes, masks_noise, dice_loss


from model.build_grasp_sam import build_grasp_sam

class SamEncoder(nn.Module):
    def __init__(self, image_encoder, prompt_encoder, sam_encoder_type):
        super(SamEncoder, self).__init__()
        
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.sam_encoder_type = sam_encoder_type

    def forward(self, batched_input):
        
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings, interm_embeddings = self.image_encoder(input_images)
        
        batched_output = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            
            
            if "eff" in self.sam_encoder_type:
                sparse_embeddings = self.prompt_encoder(
                    coords=image_record["point_coords"],
                    labels=image_record["point_labels"]
                )
                
                batched_output.append(
                    {
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": self.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":None,
                    }
                )
                
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                batched_output.append(
                    {
                        "encoder_embedding": curr_embedding.unsqueeze(0),
                        "image_pe": self.prompt_encoder.get_dense_pe(),
                        "sparse_embeddings":sparse_embeddings,
                        "dense_embeddings":dense_embeddings,
                    }
                )

        return batched_output, interm_embeddings

class ConvGraspHeader(nn.Module):
    def __init__(self, num_layers=3):
        super(ConvGraspHeader, self).__init__()

        self.layer = []
        for i in range(0, num_layers):
            self.layer.append(nn.Sequential(nn.Conv2d(4, 4, kernel_size=1),
                              nn.BatchNorm2d(4),
                              nn.ReLU()))
        self.early_conv = nn.Sequential(*self.layer)

        self.point_predictor = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.width_predictor = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        
        self.cos_predictor   = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.sin_predictor   = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        
        self.fusion          = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                

    def forward(self, x):       # x: [1, 4, 256, 256]
        
        if len (self.layer) > 0:
            x = self.early_conv(x)
        # pos_out   = F.sigmoid(self.point_predictor(x[:,0]))
        # width_out = F.sigmoid(self.width_predictor(x[:,1]))
        pos_out   = F.relu(self.point_predictor(x[:,0]))
        width_out = F.relu(self.width_predictor(x[:,1]))
    
        cos_out   = self.cos_predictor(x[:,2])
        sin_out   = self.sin_predictor(x[:,3])

        return pos_out, cos_out, sin_out, width_out 

class SamDecoder(nn.Module):
    def __init__(self, mask_decoder, sam_encoder_type, grasp_header_type, num_layers):
        super(SamDecoder, self).__init__()

        self.mask_decoder = mask_decoder
        self.sam_encoder_type = sam_encoder_type
        
        self.transformer = self.mask_decoder.transformer
        self.iou_token = self.mask_decoder.iou_token
        self.mask_tokens = self.mask_decoder.mask_tokens
        self.num_mask_tokens = self.mask_decoder.num_mask_tokens
        
        
        if "eff" in sam_encoder_type:
            self.iou_prediction_head = self.mask_decoder.iou_prediction_head
            
            
            def output_upscaling(upscaled_embedding):
                for upscaling_layer in self.mask_decoder.final_output_upscaling_layers:
                    upscaled_embedding = upscaling_layer(upscaled_embedding)
                return upscaled_embedding

            self.output_upscaling = output_upscaling
              
            
            self.output_hypernetworks_mlps = self.mask_decoder.output_hypernetworks_mlps
            
        else:
            self.iou_prediction_head = self.mask_decoder.iou_prediction_head
            self.output_upscaling = self.mask_decoder.output_upscaling
            self.output_hypernetworks_mlps = self.mask_decoder.output_hypernetworks_mlps
        
        
        self.grasp_header_type = grasp_header_type
        
        self.num_grasp_queries = 4 
        transformer_dim=256
        
        vit_dim_dict = {"vit_b":768,"vit_l":1024,"vit_h":1280,"vit_t":160, "vit_t_w_ad":160,
                        "eff_vit_t":192, "eff_vit_s":384, "eff_vit_t_w_ad":192, "eff_vit_s_w_ad":384}
        
        vit_dim = vit_dim_dict[sam_encoder_type]
        
                
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        
        self.num_grasp_tokens = 4
        self.grasp_token = nn.Embedding(self.num_grasp_tokens, transformer_dim)
        self.grasp_mlp   = nn.ModuleList([
                            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                            for i in range(self.num_grasp_tokens)])
        
        self.num_hq_tokens = self.num_mask_tokens + 1
        self.num_total_tokens = self.num_hq_tokens + self.num_grasp_tokens


        
        if 'vit_t' in sam_encoder_type:
            mid_compress_dim = transformer_dim//2
            out_compress_dim = transformer_dim//8
        
        else:
            mid_compress_dim = transformer_dim
            out_compress_dim = transformer_dim//8
        
        
        self.compress_vit_feat = nn.Sequential(
                                nn.ConvTranspose2d(vit_dim, mid_compress_dim, kernel_size=2, stride=2),
                                LayerNorm2d(mid_compress_dim),
                                nn.GELU(), 
                                nn.ConvTranspose2d(mid_compress_dim, out_compress_dim, kernel_size=2, stride=2))
        
        self.compress_vit_feat_g = nn.Sequential(
                                nn.ConvTranspose2d(vit_dim, mid_compress_dim, kernel_size=2, stride=2),
                                LayerNorm2d(mid_compress_dim),
                                nn.GELU(), 
                                nn.ConvTranspose2d(mid_compress_dim, out_compress_dim, kernel_size=2, stride=2))
        
        
        
        self.embedding_encoder_mask = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim//4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim//4, transformer_dim//8, kernel_size=2, stride=2))


        self.embedding_maskfeature = nn.Sequential(
                                nn.Conv2d(transformer_dim//8, transformer_dim // 4, 3, 1, 1), 
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim//4, transformer_dim//8, 3, 1, 1))

        
        self.embedding_encoder_grasp = nn.Sequential(
                                nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
                                LayerNorm2d(transformer_dim // 4),
                                nn.GELU(),
                                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2))
        

        self.embedding_graspfeature = nn.Sequential(
                                nn.Conv2d(transformer_dim//8, transformer_dim//4, 3, 1, 1), 
                                LayerNorm2d(transformer_dim//4),
                                nn.GELU(),
                                nn.Conv2d(transformer_dim//4, transformer_dim//8, 3, 1, 1))

        
        
        
        self.grasp_header = ConvGraspHeader(num_layers=num_layers)
        
    def predict_grasps(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        hq_feature,
        grasp_feature
    ):

        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight, self.grasp_token.weight], dim=0) #[7, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)                                       #[1, 7, 256]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)                                                              #[1, 11, 256]

    
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)                                                          #[1, 256, 64, 64]
        
        if "eff" in self.sam_encoder_type:
            pass
        else:
            src = src + dense_prompt_embeddings
        
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)            # [B, 21, 256] /[1, 4096, 256]
        iou_token_out = hs[:, 0, :]                                 # [1, 256]
        mask_tokens_out = hs[:, 1 : (1 + self.num_total_tokens), :]

        
        src = src.transpose(1, 2).view(b, c, h, w)   
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_feature
        upscaled_embedding_grasp = self.embedding_graspfeature(upscaled_embedding_sam) + grasp_feature
         
        hyper_in_list = []
        for i in range(self.num_total_tokens):
            if i < 4:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            elif i == 4:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
            elif i > 4:
                hyper_in_list.append(self.grasp_mlp[i-5](mask_tokens_out[:, i, :]))
        
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding_sam.shape
        
        
        masks_sam = (hyper_in[:,:4] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_ours = (hyper_in[:,4] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam,masks_ours],dim=1)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

        # grasp = (hyper_in[:, -1] @ upscaled_embedding_grasp.view(b, c, h * w)).view(b, -1, h, w)           #[1, 1, 256, 256]
        grasp_maps = (hyper_in[:,5:] @ upscaled_embedding_grasp.view(b, c, h * w)).view(b, -1, h, w)
   
        grasp = self.grasp_header(grasp_maps)

        return masks, iou_pred, grasp
    

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        interm_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the ViT image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted hq masks
        """
        # print(len(interm_embeddings))
        # print(interm_embeddings[0].shape)
        # exit()
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.embedding_encoder_mask(image_embeddings) + self.compress_vit_feat(vit_features)
        grasp_features= self.embedding_encoder_grasp(image_embeddings) + self.compress_vit_feat_g(vit_features) 

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        grasps_pos = []
        grasps_cos = []
        grasps_sin = []
        grasps_width = []
        
        for i_batch in range(batch_len):
            mask, iou_pred, grasp_pred = self.predict_grasps(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0),
                grasp_feature = grasp_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
            grasps_pos.append(grasp_pred[0])
            grasps_cos.append(grasp_pred[1])
            grasps_sin.append(grasp_pred[2])
            grasps_width.append(grasp_pred[3])
            
            
        grasps_poses = torch.cat(grasps_pos,0)
        grasps_coses = torch.cat(grasps_cos,0)
        grasps_sines = torch.cat(grasps_sin,0)
        grasps_widthes = torch.cat(grasps_width,0)
        
        grasps = [grasps_poses, grasps_coses, grasps_sines, grasps_widthes]
    
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)

        if multimask_output:
            mask_slice = slice(1,self.num_hq_tokens-1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds,dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)),max_iou_idx].unsqueeze(1)       
                 
        else:
            mask_slice = slice(0, 1)
            masks_sam = masks[:,mask_slice]            

        
        masks_hq = masks[:,slice(self.num_hq_tokens-1, self.num_hq_tokens), :, :]
        
 
        return grasps, masks_hq

class PlanarGraspSAM(nn.Module):
    def __init__(self, sam_encoder_type, vis=False, num_layers=0):
        super(PlanarGraspSAM, self).__init__()
        
        self.image_encoder, self.prompt_encoder, self.mask_decoder = build_grasp_sam(sam_encoder_type, adapter=False)
        
        self.encoder = SamEncoder(self.image_encoder, self.prompt_encoder, sam_encoder_type)
        
        self.decoder = SamDecoder(self.mask_decoder, sam_encoder_type, grasp_header_type="conv", num_layers=num_layers)
        
        self.sam_encoder_type = sam_encoder_type
        self.vis = vis # for inference
        
    def total_forward(self, imgs, targets, input_type="10point"):
        
        if input_type =='default':
            input_keys = ['box','point']
            k = 10
            
        elif input_type == '1point':
            input_keys = ['point']
            k = 1
            
        elif input_type == '3point':
            input_keys = ['point']
            k = 3
            
        elif input_type == '5point':
            input_keys = ['point']
            k = 5
        
        elif input_type == '10point':
            input_keys = ['point']
            k = 10
            
        elif input_type == "box":
            input_keys = ['box']
            k = 1
        
        
        masks = targets["masks"]

    
        labels_box = masks_to_boxes(masks*255)
        labels_points = masks_sample_points(masks*255, k=k)

    
        batched_input = []
        for b_i in range(len(imgs)):
            
            dict_input = dict()
            dict_input['image'] = imgs[b_i]
            input_type = random.choice(input_keys)
            if input_type == 'box':
                dict_input['boxes'] = labels_box[b_i:b_i+1]
                
                if "eff" in self.sam_encoder_type:
                    x1, y1, x2, y2 = labels_box[b_i:b_i+1][0]
                    dict_input['point_coords'] = torch.tensor([[x1,y1],[x2,y2]], device=labels_box.device)[None,:]
                    dict_input['point_labels'] = torch.tensor([2,3], device=labels_box.device)[None,:]

                
            elif input_type == 'point':
                point_coords = labels_points[b_i:b_i+1]
                
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]

            else:
                raise NotImplementedError
            
            dict_input['original_size'] = imgs[b_i].shape[:2]
            batched_input.append(dict_input)
     
        batched_output, interm_embeddings = self.encoder(batched_input)
        
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        
        results = self.decoder(
                    image_embeddings  = encoder_embedding, 
                    image_pe          = image_pe,
                    sparse_prompt_embeddings = sparse_embeddings,
                    dense_prompt_embeddings  = dense_embeddings,
                    multimask_output=False,
                    interm_embeddings = interm_embeddings,
                    )
        
        if self.vis:

            if input_type=="point":
                prompt_input = batched_input[0]['point_coords']
            elif input_type=="box":
                prompt_input = batched_input[0]['boxes']

            return results, prompt_input
        
            
        else:
            return results
    

    def weighted_mse(self, preds, targets, masks, weight=0.01):
        
        preds = preds.view(-1, 256**2)
        targets = targets.view(-1, 256**2)
        masks = masks.view(-1, 256**2)
        
        loss = 0
        for pd, gt, mk in zip(preds, targets, masks):
            fore_pred = pd[mk>0].clone()
            fore_gt = gt[mk>0]
            fore = F.mse_loss(fore_pred, fore_gt)
        
            back_pred = pd[mk==0].clone() 
            back_gt = gt[mk==0]
            back =  F.mse_loss(back_pred, back_gt)
            
            batch_loss = fore + (back * weight)
            loss += batch_loss
        
        mean_loss = loss / masks.shape[0]
            
        return mean_loss
        
        
    def compute_loss(self, grasp_pred, mask_pred, target, g_s, m_s, p, c, s, w):
        g_s = g_s
        m_s = m_s
    
        grasps_gt = target["grasps"]
        masks_gt = target["masks"]
        masks_gt = F.interpolate(masks_gt, size=(256, 256))
 
        pos_pred, cos_pred, sin_pred, width_pred = grasp_pred
        
        pos, cos, sin, width = grasps_gt
        
        pos   = F.interpolate(pos, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        cos   = F.interpolate(cos, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        sin   = F.interpolate(sin, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)
        width = F.interpolate(width, size=(256, 256), mode="bilinear", align_corners=True).squeeze(1)    
    
        p_loss = self.weighted_mse(pos_pred, pos, masks_gt)
        cos_loss = self.weighted_mse(cos_pred, cos, masks_gt)
        sin_loss = self.weighted_mse(sin_pred, sin, masks_gt)
        width_loss = self.weighted_mse(width_pred, width, masks_gt)
    
        grasp_loss = p*p_loss + c*cos_loss + s*sin_loss + w*width_loss
    
        d_loss = dice_loss(mask_pred, masks_gt)
        bce_loss = F.binary_cross_entropy_with_logits(mask_pred, masks_gt)
        mask_loss = d_loss + bce_loss

        total_loss = g_s*grasp_loss + m_s*mask_loss


        return {
            "loss": total_loss,
            "mask_loss" : mask_loss,
            "g_loss" : grasp_loss,
            "g_losses":{
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss,
                },
            "pred":{
                "pos" : pos_pred,
                "cos" : cos_pred,
                "sin" : sin_pred,
                "width" : width_pred

            }
        }
    

```

## 3.utils.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    
    
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    
    targets = targets.flatten(1)
 
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
   
    return loss.mean()


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
    
    
def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks>128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks>128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

    
    
def masks_sample_points(masks, k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    # k = 10
    samples = []
    for b_i in range(len(masks)):
        
        select_mask = (masks[b_i]>128)
        x_idx = torch.masked_select(x,select_mask)
        y_idx = torch.masked_select(y,select_mask)
        
        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:,None],samples_y[:,None]),dim=1)
        
        samples.append(samples_xy)
        
    samples = torch.stack(samples)
    
    return samples


def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector,  8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector
```

# 3.utils

## 1.misc.py

```
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import random 
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

import colorsys
import torch.nn.functional as F

import cv2

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message




def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and 
        # args.rank = int(os.environ["RANK"])
        # args.world_size = int(os.environ['WORLD_SIZE'])
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])

        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks>128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks>128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_noise(boxes, box_noise_scale=0):
    
    known_bbox_expand = box_xyxy_to_cxcywh(boxes)
    
    diff = torch.zeros_like(known_bbox_expand)
    diff[:, :2] = known_bbox_expand[:, 2:] / 2
    diff[:, 2:] = known_bbox_expand[:, 2:]
    known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),diff).cuda() * box_noise_scale
    boxes = box_cxcywh_to_xyxy(known_bbox_expand)
    boxes = boxes.clamp(min=0.0, max=1024)

    return boxes

def masks_sample_points(masks,k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, 2), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    # k = 10
    samples = []
    for b_i in range(len(masks)):
        select_mask = (masks[b_i]>128)
        x_idx = torch.masked_select(x,select_mask)
        y_idx = torch.masked_select(y,select_mask)
        
        perm = torch.randperm(x_idx.size(0))
        idx = perm[:k]
        samples_x = x_idx[idx]
        samples_y = y_idx[idx]
        samples_xy = torch.cat((samples_x[:,None],samples_y[:,None]),dim=1)
        samples.append(samples_xy)

    samples = torch.stack(samples)
    return samples


# Add noise to mask input
# From Mask Transfiner https://github.com/SysCV/transfiner
def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        w = input_masks.shape[-1]
        h = input_masks.shape[-2]
        mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
        mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    gt_masks_vector = masks / 255
    mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
    inc_masks = get_incoherent_mask(gt_masks_vector,  8)
    gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
    gt_masks_vector = gt_masks_vector * 255

    return gt_masks_vector


def mask_iou(pred_label,label):
    '''
    calculate mask iou for pred_label and gt_label
    '''

    pred_label = (pred_label>0)[0].int()
    label = (label>128)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union



# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
    device = gt.device
    dt = (dt>0)[0].cpu().byte().numpy()
    gt = (gt>128)[0].cpu().byte().numpy()

    gt_boundary = mask_to_boundary(gt, dilation_ratio)
    dt_boundary = mask_to_boundary(dt, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    boundary_iou = intersection / union
    return torch.tensor(boundary_iou).float().to(device)
```

## 2.utils.py

```
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches-1) + ']'
    
class TrainProgress():
    
    def __init__(self, dataloader, epoch):
        self.losses = AverageMeter('Loss', ':.4f')
        self.p_loss = AverageMeter('pos_loss', ':.4f')
        self.c_loss = AverageMeter('cos_loss', ':.4f')
        self.s_loss = AverageMeter('sin_loss', ':.4f')
        self.w_loss = AverageMeter('width_loss', ':.4f')
        
        self.m_loss = AverageMeter('mask_loss', ':.4f')

        self.progress = ProgressMeter(
            len(dataloader),
            [self.losses, self.p_loss, self.c_loss, self.s_loss, self.w_loss, self.m_loss],
            prefix="Epoch: [{}]".format(epoch))
        

    def progress_update(self, lossd, batch_size):
        
        losses = lossd["loss"].item()
        
        p_loss = lossd["g_losses"]["p_loss"].item()   
        cos_loss = lossd["g_losses"]["cos_loss"].item()
        sin_loss = lossd["g_losses"]["sin_loss"].item()
        width_loss = lossd["g_losses"]["width_loss"].item()
        
        mask_loss = lossd["mask_loss"].item()

        self.losses.update(losses, batch_size)
        self.p_loss.update(p_loss, batch_size)
        self.c_loss.update(cos_loss, batch_size)
        self.s_loss.update(sin_loss, batch_size)
        self.w_loss.update(width_loss, batch_size)
        self.m_loss.update(mask_loss, batch_size)
      
```

# train.py

```
import os
import time
import argparse
import warnings

warnings.filterwarnings(action='ignore')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn.functional as F

from data.jacquard_data import JacquardDataset
from data.grasp_anything_data import GraspAnythingDataset


from model.planar_grasp_sam import PlanarGraspSAM
from utils.utils import TrainProgress

from skimage.filters import gaussian
from data.utils.grasp_utils import *

def calculate_iou_match(grasp_q, grasp_angle, ground_truth_bbs, no_grasps=1, grasp_width=None):
    """
    Calculate grasp success using the IoU (Jacquard) metric (e.g. in https://arxiv.org/abs/1301.3592)
    A success is counted if grasp rectangle has a 25% IoU with a ground truth, and is withing 30 degrees.
    :param grasp_q: Q outputs of GG-CNN (Nx300x300x3)
    :param grasp_angle: Angle outputs of GG-CNN
    :param ground_truth_bbs: Corresponding ground-truth BoundingBoxes
    :param no_grasps: Maximum number of grasps to consider per image.
    :param grasp_width: (optional) Width output from GG-CNN
    :return: success
    """

    if not isinstance(ground_truth_bbs, GraspRectangles):
        gt_bbs = GraspRectangles.load_from_array(ground_truth_bbs)
    else:
        gt_bbs = ground_truth_bbs
    gs = detect_grasps(grasp_q, grasp_angle, width_img=grasp_width, no_grasps=no_grasps)
    for g in gs:
        if g.max_iou(gt_bbs) > 0.25:
            return True
    else:
        return False


def post_process_output(q_img, cos_img, sin_img, width_img):
    """
    Post-process the raw output of the GG-CNN, convert to numpy arrays, apply filtering.
    :param q_img: Q output of GG-CNN (as torch Tensors)
    :param cos_img: cos output of GG-CNN
    :param sin_img: sin output of GG-CNN
    :param width_img: Width output of GG-CNN
    :return: Filtered Q output, Filtered Angle output, Filtered Width output
    """
    if len(q_img.shape) == 3:
        q_img = F.interpolate(q_img.unsqueeze(0), size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img.unsqueeze(0), size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img.unsqueeze(0), size=(1024, 1024))[0]
        width_img = F.interpolate(width_img.unsqueeze(0), size=(1024, 1024))[0]
        
    elif len(q_img.shape) == 4:
        q_img = F.interpolate(q_img, size=(1024, 1024))[0]
        cos_img = F.interpolate(cos_img, size=(1024, 1024))[0]
        sin_img = F.interpolate(sin_img, size=(1024, 1024))[0]
        width_img = F.interpolate(width_img, size=(1024, 1024))[0]

    width_scale = 512
    q_img = q_img.data.detach().cpu().numpy().squeeze()
    ang_img = (torch.atan2(sin_img, cos_img) / 2.0).data.detach().cpu().numpy().squeeze()
    width_img = width_img.data.detach().cpu().numpy().squeeze() * width_scale

    q_img = gaussian(q_img, 2.0, preserve_range=True)
    ang_img = gaussian(ang_img, 2.0, preserve_range=True)
    width_img = gaussian(width_img, 1.0, preserve_range=True)

    return q_img, ang_img, width_img

def cal_grasp_ious(test_dataset, model, args):
    test_loader = torch.utils.data.DataLoader(test_dataset, 1, pin_memory=False, 
                                               num_workers=4, shuffle=True)
    ld = len(test_loader)
    results = {"correct": 0, "failed": 0, 
               "g_loss":0, 
               "g_losses":{
                "p_loss": 0,
                "cos_loss": 0,
                "sin_loss": 0,
                "width_loss": 0,
                },}
    
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, masks, grasps, didx, rot, zoom_factor = data
            images = images.to(args.device)    
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]
            
        
            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps    
                
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)    
            
            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)


            loss = lossd["g_loss"]
            results['g_loss'] += loss.item() / ld
            for ln, l in lossd['g_losses'].items():
                if ln not in results['g_losses']:
                    results['g_losses'][ln] = 0
                results['g_losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            success = calculate_iou_match(q_out, ang_out, 
                                          test_dataset.get_gtbb(didx.item(), rot.item(), zoom_factor.item()), 
                                          no_grasps=1, 
                                          grasp_width=w_out)
            
            if success:
                results["correct"] += 1
            else:
                results["failed"] += 1
            
            success_rate = 100 * results["correct"] / (results["correct"] + results["failed"])
    return results["g_loss"], success_rate


def setup_model(model_type, sam_encoder_type, prompt_mode):
    if model_type == "bs_grasp_sam":
        model = PlanarGraspSAM(sam_encoder_type=sam_encoder_type, num_layers=0)

    else:
        raise("please input correct model type")

    return model


def main(args):
    is_save = args.save
    
    if is_save:
        if args.seen:
            args.exp_name = "seen" + "_" + args.sam_encoder_type + "_" + args.prompt
            
        else:
            args.exp_name = "total" + "_" + args.sam_encoder_type + "_" + args.prompt
            
        exp_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        save_path = os.path.join(args.save_dir, args.exp_name, args.dataset_name, '{}'.format(exp_time))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    
        f = open(os.path.join(save_path, "info.txt"), 'w')
        f.write(str(args))
        f.close()
        

    args.device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
        
    if args.dataset_name == "jacquard":
        train_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                        random_rotate=False, random_zoom=False,
                                        start=0.0, end=0.9, seen=args.seen, train=True)
        test_dataset = JacquardDataset(root=args.root, crop_size=1024, include_mask=True, 
                                       random_rotate=False, random_zoom=False,   
                                       start=0.9, end=1.0, seen=False, train=False)
    
    elif args.dataset_name == "grasp_anything":
        train_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                             random_rotate=False, random_zoom=False,
                                             start=0.0, end=0.9, seen=args.seen, train=True)
        test_dataset = GraspAnythingDataset(root=args.root, include_mask=True, 
                                            random_rotate=False, random_zoom=False,
                                            start=0.9, end=1.0, seen=False, train=False)
            
    print("train_dataset size : {}".format(train_dataset.__len__()))
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=False, num_workers=4, shuffle=True)
    
    
    model = setup_model(model_type="bs_grasp_sam", sam_encoder_type=args.sam_encoder_type, prompt_mode=args.prompt)

    if args.resume_ckp:
        state_dict = torch.load(args.resume_ckp, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)


    model.to(args.device)
    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, 1e-7)
    
    start_epoch = 0
    
    if args.resume_ckp:
        start_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])

    print("-"*80)
    

    best_success_rate = 0.0
    for epoch in range(start_epoch, args.epochs):
        
        t_progress = TrainProgress(train_loader, epoch)
        for i, data in enumerate(train_loader):
            images, masks, grasps, _, _, _ = data
        
            images = images.to(args.device)
            masks = masks.to(args.device)
            grasps = [g.to(args.device) for g in grasps]

            targets = {}
            targets["masks"] = masks
            targets["grasps"] = grasps          
            
            optimizer.zero_grad()
            
            grasp_pred, mask_pred = model.total_forward(imgs=images, targets=targets)

            lossd = model.compute_loss(grasp_pred, mask_pred, targets, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            loss = lossd["loss"]
            loss.backward()

            optimizer.step()
            
            t_progress.progress_update(lossd, args.batch_size)
            if i % args.print_freq == 0:
                t_progress.progress.display(i)
                
        scheduler.step()
        print("-"*80)


        if args.validate:
            g_loss, success_rate = cal_grasp_ious(test_dataset, model, args)
            print("epoch : {} / g_loss : {:.4f} / success_rate : {:.4f}".format(epoch, g_loss, success_rate))

            if args.save: 
                if success_rate > best_success_rate or epoch == 0 or (epoch % 10) == 0:
                    best_success_rate = success_rate
                    save_dict = {
                        "epoch" : epoch,
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "scheduler" : scheduler.state_dict(),
                    }
                    
                    torch.save(save_dict, os.path.join(save_path, "epoch_%02d_sr_%0.2f.pth" % (epoch, success_rate)))
                    print("save best model in {}".format(save_path))
        else:
            if args.save:
                save_dict = {
                        "epoch" : epoch,
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "scheduler" : scheduler.state_dict(),
                    }
                
                torch.save(save_dict, os.path.join(save_path, "epoch{}.pth".format(str(epoch))))
                print("save model in {}".format(save_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-encoder-type", type=str, default="eff_vit_t_w_ad")
    parser.add_argument("--gpu-num", type=int, default=2, help="gpu id number")

    parser.add_argument("--dataset-name", type=str, default="jacquard", help="dataset name")
    parser.add_argument("--root", type=str, help="dataset root")
    parser.add_argument("--seen", action='store_true')
    parser.add_argument("--prompt", type=str, default='default')
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=50)
    
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--save-dir", type=str, default="final_result")
    parser.add_argument("--resume-ckp", type=str, default=None)

    parser.add_argument("--validate", action='store_true')
    parser.add_argument("--split", default=0.9, type=float)

    args = parser.parse_args()

    main(args)

```

