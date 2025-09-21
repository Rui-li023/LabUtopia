import cv2
import numpy as np

def string_to_color(category, num_colors=256):
    hash_value = hash(category) % num_colors
    color = [(hash_value * 37) % 256, (hash_value * 59) % 256, (hash_value * 73) % 256]
    return tuple(color)

def create_color_map(id_to_labels):
    color_map = {}
    for id_str, label in id_to_labels.items():
        color_map[id_str] = string_to_color(label)
    return color_map

class CameraUtils:
    def __init__(self):
        pass
    
    def process_single_type(self, camera, image_type):
        if image_type == "rgb":
            rgb_img = camera.get_rgb()
            img_for_record = np.transpose(rgb_img, (2, 0, 1))
            return img_for_record, img_for_record
        elif image_type == "pointcloud":
            rgb_img = camera.get_rgb()
            img_for_display = np.transpose(rgb_img, (2, 0, 1))
            pointcloud = camera.get_pointcloud()
            if pointcloud is not None:
                pointcloud_for_record = pointcloud.astype(np.float32)
                return pointcloud_for_record, img_for_display
            else:
                print("Warning: Point cloud data not available.")
                return None, None
        elif image_type == "depth":
            depth = camera.get_depth()
            if depth is not None:
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_for_record = depth_normalized.astype(np.uint8)
                depth_for_record = depth_for_record[ :, :, np.newaxis]
                depth_for_record = np.transpose(depth_for_record, (2, 0, 1)) # C * W * H
                depth_for_display = cv2.applyColorMap(depth_for_record[0], cv2.COLORMAP_JET)
                return depth_for_record, depth_for_display
            else:
                print("Warning: Depth data not available.")
                return None, None
        elif image_type == "segmentation":
            frame_dict = camera.get_current_frame()
            instance_id_seg = frame_dict.get('instance_segmentation')
            if instance_id_seg is not None:
                seg = instance_id_seg['data']
                if seg is not None:
                    seg_for_record = seg.astype(np.uint8)
                    id_to_labels = instance_id_seg.get('info', {}).get('idToLabels', {})
                    color_map = create_color_map(id_to_labels)
                    seg_for_display = np.zeros((seg_for_record.shape[0], seg_for_record.shape[1], 3), dtype=np.uint8) # H * W * C
                    for id_value in np.unique(seg_for_record):
                        if str(id_value) in color_map:
                            seg_for_display[seg_for_record == id_value] = color_map[str(id_value)]
                    seg_for_record = seg_for_record[:, :, np.newaxis]
                    seg_for_record = np.transpose(seg_for_record, (2, 0, 1)) # C * W * H
                    seg_for_display = np.transpose(seg_for_display, (2, 0, 1)) # C * W * H
                return seg_for_record, seg_for_display
            else:
                print("Warning: Segmentation data not available.")
                return None, None
        elif image_type == "point":
            frame_dict = camera.get_current_frame()
            rgb_img = camera.get_rgb()
            img_for_display = rgb_img[..., ::-1].copy()  # Make a copy to modify
            instance_id_seg = frame_dict.get('instance_segmentation')
            if instance_id_seg is not None:
                seg = instance_id_seg['data']
                if seg is not None:
                    seg_for_process = seg.astype(np.uint8)
                    id_to_labels = instance_id_seg.get('info', {}).get('idToLabels', {})
                    for id_value in np.unique(seg_for_process):
                        if str(id_value) in id_to_labels and id_value != 0 and id_value != 1:
                            mask = (seg_for_process == id_value)
                            y, x = np.where(mask)
                            if len(y) > 0 and len(x) > 0:
                                center_y, center_x = int(np.mean(y)), int(np.mean(x))
                                cv2.circle(img_for_display, (center_x, center_y), 6, (255, 100, 100), -1)
                    img_for_display_rgb = img_for_display[..., ::-1]  # BGR to RGB
                    img_for_record = np.transpose(img_for_display_rgb, (2, 0, 1))
                    return img_for_record, img_for_display
                else:
                    print("Warning: Segmentation data not available.")
                    return None, None
            else:
                print("Warning: Instance segmentation not available.")
                return None, None
        else:
            return None, None

def process_camera_image(camera, image_type):
    """
    Process camera image with support for combined types (e.g., 'rgb+pointcloud')
    
    Args:
        camera: Camera instance
        image_type: String indicating the type(s) of image data to process
                   Can be single type or combined types with '+' (e.g., 'rgb+pointcloud')
    
    Returns:
        tuple: (record_data, display_data) where each can be single item or dict
    """
    record, display = CameraUtils().process_single_type(camera, "rgb")
    if '+' in image_type:
        types = image_type.split('+')
        record_dict = {}
        for t in types:
            record, _ = CameraUtils().process_single_type(camera, t)
            if record is not None:
                record_dict[t] = record
        return record_dict, display
    else:
        return CameraUtils().process_single_type(camera, image_type)