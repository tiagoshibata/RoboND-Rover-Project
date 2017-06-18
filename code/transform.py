import numpy as np


def image_to_rover(binary_img):
    """Convert from image to rover coordinates."""
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


def rotate(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    return xpix_rotated, ypix_rotated


def translate(xpix_rot, ypix_rot, xpos, ypos, scale):
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    return xpix_translated, ypix_translated


def to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    """Apply rotation, translation and clipping."""
    xpix_rot, ypix_rot = rotate(xpix, ypix, yaw)
    xpix_tran, ypix_tran = translate(xpix_rot, ypix_rot, xpos, ypos, scale)
    return [np.clip(np.int_(x), 0, world_size - 1) for x in (xpix_tran, ypix_tran)]


def to_polar_coords(x_pixel, y_pixel):
    """Convert (x_pixel, y_pixel) to polar coordinates in rover space."""
    # Distance to each pixel
    dist = np.sqrt(x_pixel * x_pixel + y_pixel * y_pixel)
    # Angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles
