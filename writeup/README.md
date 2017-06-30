# Sample search and return writeup

## Setup

All the tests and datasets were created using the "Fastest" graphic setting. The code was tested using the [provided Anaconda environment](https://github.com/udacity/RoboND-Python-StarterKit).

## Notebook analysis

Quick test were done on the jupyter notebook before jumping into the simulator. Since the simulator provides more information and a better overview on the rover's performance, I didn't bother optimizing the jupyter notebook.

The first step was implementing detection of navigable terrain. The following steps were taken:

* Threshold the camera image, allowing only bright (RGB value greater than 160, 160, 160) pixels to pass.
* Apply a perspective transform.
* Convert the data to rover-centric coordinates (considering the rover at the origin, facing the positive X axis).
* Convert the data to world coordinates and add to the map.

I noticed that the thresholding operation would often see too much navigable terrain when the robot was accelerating or too few when breaking due to changes in pitch. Furthermore, depending on the lightning conditions, some of the sky could be detected as navigable terrain, and navigable terrain far away from the robot could be detected as obstacles.

![Sky in navigable terrain](sky-in-navigable.png)

I decided to keep only the pixels closest to the robot after the perspective transform, by setting other pixels to 0 after the perspective transform. Initially, I used only the bottom two thirds of the image:

```python
warped = perspect_transform(thresh, source, destination)
warped[0:img.shape[0] // 3, :] = 0
warped[:, :img.shape[1] // 5] = 0
warped[:, -img.shape[1] // 5:] = 0
```

A test with the provided dataset [can be found here](https://www.youtube.com/watch?v=6upkIXDBIvI&list=PLxN-KgeGj8fiu5coK2FuH_vvGjpgjRN1t&index=4) and with a custom, [bigger dataset here](https://www.youtube.com/watch?v=xZgmuXtt2ec&list=PLxN-KgeGj8fiu5coK2FuH_vvGjpgjRN1t&index=5).

All non-navigable terrain was considered as obstacles. A 200x200 auxiliary map was created to save the confidence that a given tile was navigable or obstacle. Every time navigable terrain was seen, some score would be added to that pixel, and obstacles would reduce the score. Pixels with positive scores were considered navigable, and negative scores obstacles. Using scores allowed for the robot to correct wrong detections made from far away.

The robot would much more often mistake navigable terrain as obstacles than the other way around; the weights were chosen to compensate for this (4 for navigable and 1 for obstacles).

The mapping code used on the notebook was:

```python
thresh = color_thresh(img)
output_image[0:img.shape[0], img.shape[1]:, 2] = 255 * thresh
obstacle = perspect_transform(1 - thresh, source, destination)

# Warped image
warped = perspect_transform(thresh, source, destination)
warped[0:2 * img.shape[0] // 3, :] = 0
warped[:, :img.shape[1] // 5] = 0
warped[:, -img.shape[1] // 5:] = 0

obstacle[0:2 * img.shape[0] // 3, :] = 0
obstacle[:, :img.shape[1] // 3] = 0
obstacle[:, -img.shape[1] // 3:] = 0

# Lower right corner
output_image[data.worldmap.shape[0]:, img.shape[1]:, 2] = 255 * warped
output_image[data.worldmap.shape[0]:, img.shape[1]:, 0] = 255 * obstacle

rover_navigable = rover_coords(warped)
world_navigable_x, world_navigable_y = pix_to_world(*rover_navigable, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
data.map[world_navigable_y, world_navigable_x] += 4

rover_obstacle = rover_coords(obstacle)
world_obstacle_x, world_obstacle_y = pix_to_world(*rover_obstacle, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
data.map[world_obstacle_y, world_obstacle_x] -= 1

data.worldmap[:, :, 2] = np.zeros((200, 200), dtype=int)
data.worldmap[:, :, 2][data.map > 0] = 255
data.worldmap[:, :, 0] = np.zeros((200, 200), dtype=int)
data.worldmap[:, :, 0][data.map < 0] = 255
```

A small program was written to choose threshold values for the rocks. The program loads screenshots containing rocks and shows the detections for a fixed threshold value. [It can be found here](https://github.com/tiagoshibata/RoboND-Rover-Project/blob/master/code/rock_visualizer.py). Appropriate values were chosen and put in the jupyter notebook. I decided to ignore rocks too far away, since the location gets imprecise due to changes in pitch when accelerating. The full `process_image` code is:

```python
def process_image(img):
    if not data.count:
        data.map[:, :] = 0
        data.worldmap[:, :, :] = 0

    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    # Original image in upper left corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

    thresh = color_thresh(img)
    obstacle = perspect_transform(1 - thresh, source, destination)

    low = (140, 140, 140)
    high = (5, 70, 70)
    under_thresh = (img[:, :, 0] < low[0]) & (img[:, :, 1] < low[1]) & (img[:, :, 2] < low[2])
    above_thresh = (img[:, :, 0] > high[0]) & (img[:, :, 1] > high[1]) & (img[:, :, 2] > high[2])
    rocks = np.ones_like(img[:, :, 0])
    rocks[under_thresh] = 0
    rocks[above_thresh] = 0

    output_image[0:img.shape[0], img.shape[1]:, 2] = 255 * thresh
    output_image[0:img.shape[0], img.shape[1]:, 1] = 255 * rocks

    # Warped image
    warped = perspect_transform(thresh, source, destination)
    warped[0:2 * img.shape[0] // 3, :] = 0
    warped[:, :img.shape[1] // 5] = 0
    warped[:, -img.shape[1] // 5:] = 0

    obstacle[0:2 * img.shape[0] // 3, :] = 0
    obstacle[:, :img.shape[1] // 3] = 0
    obstacle[:, -img.shape[1] // 3:] = 0

    rocks = perspect_transform(rocks, source, destination)
    rocks[0:img.shape[0] // 2, :] = 0

    # Lower right corner
    output_image[data.worldmap.shape[0]:, img.shape[1]:, 0] = 255 * obstacle
    output_image[data.worldmap.shape[0]:, img.shape[1]:, 1] = 255 * rocks
    output_image[data.worldmap.shape[0]:, img.shape[1]:, 2] = 255 * warped

    rover_navigable = rover_coords(warped)
    world_navigable_x, world_navigable_y = pix_to_world(*rover_navigable, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
    data.map[world_navigable_y, world_navigable_x] += 4

    rover_obstacle = rover_coords(obstacle)
    world_obstacle_x, world_obstacle_y = pix_to_world(*rover_obstacle, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)
    data.map[world_obstacle_y, world_obstacle_x] -= 1

    rover_rocks = rover_coords(rocks)
    world_rocks_x, world_rocks_y = pix_to_world(*rover_rocks, data.xpos[data.count], data.ypos[data.count], data.yaw[data.count], data.worldmap.shape[0], 10)

    data.worldmap[:, :, 2] = np.zeros((200, 200), dtype=int)
    data.worldmap[:, :, 2][data.map > 0] = 255
    data.worldmap[:, :, 0] = np.zeros((200, 200), dtype=int)
    data.worldmap[:, :, 0][data.map < 0] = 255
    data.worldmap[world_rocks_y, world_rocks_x, :] = 255

    # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    # Flip map overlay so y-axis points upward and add to output_image
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)

    # Then putting some text over the image
    # cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20),
    #            cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()

    return output_image
```

[A video with mapping and rock detection can be found here.](https://www.youtube.com/watch?v=-GnjOVLRNz4&index=6&list=PLxN-KgeGj8fiu5coK2FuH_vvGjpgjRN1t) The generated video shows mapping and rock detection:

![Final notebook](final_notebook.png)

## Simulator

Several enhancements were done when testing in the simulator. The final project maps 
