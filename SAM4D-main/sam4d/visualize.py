import copy

import cv2
import numpy as np
from easydict import EasyDict as edict
import supervision as sv
import open3d as o3d


def sam4d_viz(image, pcd, mask_logits_dict=None, instance_ids=None, point_prompts=None, box_prompts=None, title=None):
    image = copy.deepcopy(image)
    pcd = copy.deepcopy(pcd)
    if mask_logits_dict is not None:
        img_masks = (mask_logits_dict['img'][:, 0] > 0.0).cpu().numpy()
        pts_masks = (mask_logits_dict['pts'][:, 0] > 0.0).squeeze(-1).cpu().numpy()
    else:
        img_masks = None
        pts_masks = None

    if point_prompts is not None:
        assert isinstance(point_prompts, dict)
        img_point_prompts = point_prompts.get('img', None)
        pts_point_prompts = point_prompts.get('pts', None)
    else:
        img_point_prompts = None
        pts_point_prompts = None
    if box_prompts is not None:
        assert isinstance(box_prompts, dict)
        img_box_prompts = box_prompts.get('img', None)
        pts_box_prompts = box_prompts.get('pts', None)
    else:
        img_box_prompts = None
        pts_box_prompts = None

    viz_img = img_draw_masks_and_prompts(image, img_masks, instance_ids, img_point_prompts, img_box_prompts)
    viz_pts = pts_draw_masks_and_prompts(pcd, pts_masks, instance_ids, pts_point_prompts, pts_box_prompts)
    ret = plot_images_together_cv2([viz_img, viz_pts], show=False)

    if title is not None:
        h, w = ret.shape[:2]
        ret[:70, :] = 0
        tw, th = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
        org_x = (w - tw) // 2
        org_y = (70 + th) // 2
        cv2.putText(ret, title, (org_x, org_y), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 255, 255), 2, cv2.LINE_AA)
    return ret


def img_draw_masks_and_prompts(image, masks, instance_ids, point_prompts=None, box_prompts=None):
    '''
    image: the image to draw on
    masks: nd array (n, h, w)
    instance_ids: []
    point_prompts: None or {point_coords: array (n, m, 2) or (p, 2), point_labels: array (n, m) or (p, )}
    box_prompts: None or {point_coords: array (n, m, 2) or (p, 2), point_labels: array (n, m) or (p, )}
    '''

    def get_bounding_box(mask):
        if mask.sum() == 0:
            return (0, 0, 0, 0)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        return (col_min, row_min, col_max, row_max)

    if masks is None:
        return image

    # draw prompts first
    if point_prompts is not None:
        positive_points = point_prompts['point_coords'][point_prompts['point_labels'] == 1]
        negative_points = point_prompts['point_coords'][point_prompts['point_labels'] == 0]
        for point in positive_points:
            cv2_scatter(image, point[0], point[1], color=(0, 255, 0), marker_size=200, edge_color=(255, 255, 255), line_width=4)
        for point in negative_points:
            cv2_scatter(image, point[0], point[1], color=(255, 0, 0), marker_size=200, edge_color=(255, 255, 255), line_width=4)

    assert isinstance(masks, np.ndarray) and masks.ndim == 3
    if masks.dtype != np.bool_:
        masks = masks == 1
    detections = sv.Detections(
        xyxy=np.array([get_bounding_box(mask) for mask in masks]),  # if box is None else box.reshape(-1, 4),
        mask=masks,
        class_id=np.array(instance_ids, dtype=np.int32),
    )

    # custom label to show both id and class name
    labels = [f"{instance_id}" for instance_id in instance_ids]

    annotated_frame = image.copy()
    if box_prompts is not None:
        box_annotator = sv.BoxAnnotator(thickness=2)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    label_annotator = sv.LabelAnnotator(text_scale=0.25, text_thickness=1)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    return annotated_frame


def pts_draw_masks_and_prompts(pcd, masks, instance_ids, point_prompts=None, box_prompts=None):
    '''
    pcd: the point cloud to draw on, array (n_pts, 3)
    masks: nd array (n, n_pts)
    instance_ids: []
    point_prompts: None or {point_coords: array (n, m, 3) or (p, 3), point_labels: array (n, m) or (p, )}
    box_prompts: None or {point_coords: array (n, m, 3) or (p, 3), point_labels: array (n, m) or (p, )}
    '''
    viz_cfg = edict()
    viz_cfg.lookat = [30, 0, 0]
    viz_cfg.front = [-10.0, 0, 3.0]
    viz_cfg.up = [0, 0, 1]
    viz_cfg.zoom = 0.2
    viz_cfg.width = 1920
    viz_cfg.height = 1280

    pc_plotter = Open3dPCPlotter(viz_cfg=viz_cfg, axis_size=2.0)
    if masks is None:
        pc_plotter.add_points(pcd, color=(0.4, 0.4, 0.4))  # BG
    else:
        assert isinstance(masks, np.ndarray) and masks.ndim == 2
        assert pcd.shape[0] == masks.shape[1]

        mask_annotator = sv.MaskAnnotator()
        color_palette = [(x.r / 255., x.g / 255., x.b / 255.) for x in mask_annotator.color.colors]  # keep same with image mask

        colors = np.zeros_like(pcd[:, :3])
        colors[:, :] = np.array((0.4, 0.4, 0.4))
        colors[masks[0] == -1] = np.array((0.75, 0.75, 0.75))  # ignore
        for i, (mask, ins_id) in enumerate(zip(masks, instance_ids)):
            colors[mask == 1] = np.array(color_palette[ins_id % len(color_palette)])

        pc_plotter.add_points(pcd, colors=colors)

        if point_prompts is not None:
            assert point_prompts['point_coords'].shape[-1] == 3
            prpt_coords = point_prompts['point_coords'].reshape(-1, 3)
            prpt_labels = point_prompts['point_labels'].reshape(-1)
            for i, l in enumerate(prpt_labels):
                tmp_clr = (0., 0., 1.) if l == 0 else (0., 1., 0.)
                pc_plotter.add_spheres(prpt_coords[i:i + 1], color=tmp_clr, size=0.3)

    screenshot = './tmp.png'
    pc_plotter.visualize(screenshot=screenshot)
    out_img = cv2.imread(screenshot)
    return out_img


class Open3dPCPlotter(object):
    def __init__(self, viz_cfg=None, axis_size=5.0):
        self.geometries = dict()
        self.geometries['axis'] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)

        self.viz_cfg = viz_cfg
        self.geometries['pcds'] = []
        self.geometries['spheres'] = []

    def add_points(self, points, colors=None, color=None):
        '''
        Args:
            points: (N, 3)
            colors: None or (N, 3)
            color: None or a triplet of float ranging from 0 to 1, eg (1, 1, 1) for white.
        '''
        assert not (colors is not None and color is not None), 'colors and color cannot be set at the same time.'
        xyz = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif color is not None:
            pcd.paint_uniform_color(color)
        self.geometries['pcds'].append(pcd)

    def add_spheres(self, points, color=(0., 0., 0.), size=1):
        if isinstance(points, np.ndarray):
            points = points[:, :3]
        for point in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
            sphere.paint_uniform_color(color)
            sphere.translate(tuple(point))
            self.geometries['spheres'].append(sphere)

    def visualize(self, screenshot=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.viz_cfg.width, height=self.viz_cfg.height, visible=screenshot is None)
        geometry_list = self.get_geometry_list()
        for geometry in geometry_list:
            vis.add_geometry(geometry)
        self.set_view(vis)

        if screenshot is not None:
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(screenshot, do_render=True)
        else:
            vis.run()
            vis.destroy_window()

    def get_geometry_list(self):
        geometry_list = []
        for name, geometry in self.geometries.items():
            if isinstance(geometry, list):
                [geometry_list.append(x) for x in geometry]
            else:
                geometry_list.append(geometry)
        return geometry_list

    def set_view(self, vis):
        view_control = vis.get_view_control()
        view_control.set_lookat(self.viz_cfg.lookat)
        view_control.set_front(self.viz_cfg.front)
        view_control.set_up(self.viz_cfg.up)
        view_control.set_zoom(self.viz_cfg.zoom)


def plot_images_together_cv2(images, show=False, save_path=None):
    num_images = len(images)

    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    total_width = cols * images[0].shape[1]
    total_height = rows * images[0].shape[0]

    canvas = np.zeros((total_height, total_width, images[0].shape[2]), dtype=np.uint8)

    for idx, img in enumerate(images):
        row_start = (idx // cols) * images[0].shape[0]
        col_start = (idx % cols) * images[0].shape[1]

        canvas[row_start:row_start + images[0].shape[0], col_start:col_start + images[0].shape[1]] = img

        # Add horizontal separator line
        cv2.line(canvas, (col_start + images[0].shape[1] - 1, row_start),
                 (col_start + images[0].shape[1] - 1, row_start + images[0].shape[0]), color=(255, 255, 255), thickness=1)

        # Add vertical separator line
        cv2.line(canvas, (col_start, row_start + images[0].shape[0] - 1),
                 (col_start + images[0].shape[1], row_start + images[0].shape[0] - 1), color=(255, 255, 255), thickness=1)

    # Remove extra separators on the last row and column
    canvas[:, -1, :] = (255, 255, 255)
    canvas[-1, :, :] = (255, 255, 255)

    if show:
        cv2.imshow('Combined Images', canvas)
        cv2.waitKey(0)

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"Saved combined image to {save_path}")

    return canvas


def cv2_scatter(img, x, y, color=(0, 255, 0), marker_size=20,
                edge_color=(255, 255, 255), line_width=1):
    """
    OpenCV implementation of a star-shaped scatter plot function

    Parameters:
    img       : Background image to draw on (in BGR format)
    x         : Array of x-coordinates
    y         : Array of y-coordinates
    color     : Fill color (BGR tuple)
    marker_size : Marker size (controls the size of the star)
    edge_color : Edge line color (BGR tuple)
    line_width : Edge line width (pixels)
    """
    # Parameter preprocessing
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    scale_factor = 2.0  # Size scaling factor

    # Calculate star parameters
    outer_radius = np.sqrt(marker_size) * scale_factor
    inner_radius = outer_radius * 0.382  # Golden ratio

    # Iterate over all points to draw
    for xi, yi in zip(x, y):
        center = (int(round(xi)), int(round(yi)))

        # Generate star vertices
        star = []
        for i in range(10):
            angle_deg = 18 + 36 * i  # Ensure vertex points upward
            angle_rad = np.deg2rad(angle_deg)
            r = outer_radius if i % 2 == 0 else inner_radius
            px = center[0] + r * np.cos(angle_rad)
            py = center[1] + r * np.sin(angle_rad)
            star.append([px, py])

        star_pts = np.array(star, dtype=np.int32).reshape((-1, 1, 2))

        # Fill color
        cv2.fillPoly(img, [star_pts], color=color)

        # Draw edge lines
        if line_width > 0:
            cv2.polylines(img, [star_pts], isClosed=True, color=edge_color, thickness=line_width, lineType=cv2.LINE_AA)
