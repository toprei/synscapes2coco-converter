import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
from PIL import Image

# R + G * 256 + B * 256^2.
# sub_mask , image_id, category_id, annotation_id, is_crowd == False
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask = np.asanyarray(sub_mask)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):

            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if poly.geom_type is not 'MultiPolygon':
            polygons.append(poly)
            try:
                segmentation = np.array(poly.exterior.coords).ravel().tolist()
                #segmentation.clip(0)
                segmentations.append(segmentation)
            except:
                print(poly.geom_type, len(poly.geoms), poly)
        else:
            for i in range(0, len(poly.geoms)):
                segmentation = np.array(poly[i].exterior.coords).ravel().tolist()
                #segmentation.clip(0)
                segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    if not multi_poly.is_empty:
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area
        # size = [width - x + 1, height - y + 1]
        # segmentations["size"].append(size)
    else:
        bbox = (0, 0, 0, 0)
        area = 0
        # size = [0, 0]
        # segmentations["size"].append(size)

    # round segmentation int numbers and remove empty lists
    rounded_segmentations = []
    for num_list in segmentations:
        if len(num_list) != 0:
            rounded_segmentations.append([round(x) for x in num_list])

    annotation = {
        'id': annotation_id,
        'image_id': image_id,
        'segmentation': rounded_segmentations,
        'category_id': category_id,
        'iscrowd': is_crowd,
        'area': round(area),
        'bbox': [round(num) for num in bbox]
    }
    return annotation


def create_sub_masks(mask_image, rgb_list):
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    counter = 0
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]
            # If the pixel is in the rgb list
            if pixel in rgb_list:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                    # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    counter = counter + 1
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))
                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    # print(counter)
    return sub_masks

