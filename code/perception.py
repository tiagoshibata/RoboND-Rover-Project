from vision import Vision
import transform

vision = Vision()


def perception_step(Rover):
    obstacle, rock, navigable = vision.process(Rover.img)
    Rover.vision_image[:, :, 0] = 255 * obstacle
    Rover.vision_image[:, :, 1] = 255 * rock
    Rover.vision_image[:, :, 2] = 255 * navigable

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    rover_navigable = transform.image_to_rover(navigable)

    rover_rock = transform.image_to_rover(rock)
    world_rock_x, world_rock_y = transform.to_world(*rover_rock, *Rover.pos, Rover.yaw, 200, 10)
    navigation_map = Rover.navigation_map
    navigation_map.add_rock(world_rock_x, world_rock_y)

    if len(rover_rock[0].nonzero()[0]):
        _, Rover.rock_angle = transform.to_polar_coords(*rover_rock)
    else:
        Rover.rock_angle = None

    if (Rover.pitch > 356 or Rover.pitch < 4) and (Rover.roll > 356 or Rover.roll < 4):
        rover_x, rover_y = round(Rover.pos[0]), round(Rover.pos[1])
        navigation_map.map[rover_y, rover_x] = 1e8
        world_navigable_x, world_navigable_y = transform.to_world(*rover_navigable, *Rover.pos, Rover.yaw, 200, 10)
        navigation_map.add_navigable(world_navigable_x, world_navigable_y, 200 - rover_navigable[0])

        rover_obstacle = transform.image_to_rover(obstacle)
        world_obstacle_x, world_obstacle_y = transform.to_world(*rover_obstacle, *Rover.pos, Rover.yaw, 200, 10)
        navigation_map.add_obstacle(world_obstacle_x, world_obstacle_y, 200 - rover_obstacle[0])

        navigation_map.update()
        Rover.worldmap[:, :, 0] = navigation_map.undiscovered_paths()
        # Rover.worldmap[:, :, 0] = navigation_map.obstacle
        Rover.worldmap[:, :, 1] = navigation_map.rock
        Rover.worldmap[:, :, 2] = navigation_map.navigable

    dist, angles = transform.to_polar_coords(*rover_navigable)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
    return Rover
