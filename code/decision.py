import a_star
from math import atan2, pi
import numpy as np

collected = 0
target_yaw = None
frames_to_path_trace = 0
stuck = 0
aligning = False


def angle_diff(a, b):
    return (a - b + 180 + 360) % 360 - 180


def path_to_undiscovered(rover):
    rover_x, rover_y = round(rover.pos[0]), round(rover.pos[1])
    rover_yx = (rover_y, rover_x)
    navigation_map = rover.navigation_map

    def goal_score(goal):
        # Sort with closest goals last
        dy, dx = goal[0] - rover.pos[1], goal[1] - rover.pos[0]
        print('Dist {}, angle {}'.format(a_star.distance(rover_yx, goal) ** 2, abs(angle_diff(rover.yaw, 180 / pi * atan2(dy, dx)))))
        return -(a_star.distance(rover_yx, goal) ** 2) - abs(angle_diff(rover.yaw, 180 / pi * atan2(dy, dx)))

    y_goal, x_goal = navigation_map.undiscovered_paths().nonzero()
    goals = list(sorted(
        zip(y_goal, x_goal),
        key=goal_score,
    ))
    if len(goals) and goals[-1] == rover_yx:
        goals.pop()
    while len(goals):
        tentative = goals[-1]
        path = a_star.run(rover_yx, goals.pop(), navigation_map.navigable)
        if path:
            return path
        print('PATH {}x{} failed'.format(tentative[0], tentative[1]))


def rotate_to_angle(Rover, a):
    Rover.throttle = 0
    if Rover.vel > 0.2:
        Rover.brake = 1
    else:
        Rover.brake = 0
        Rover.steer = angle_diff(a, Rover.yaw) > 0 and 15 or -15


def follow_path(Rover, path):
    global aligning
    global target_yaw
    target = path[:5][-1]  # Read up to 4 steps in advance
    dy, dx = target[0] - Rover.pos[1], target[1] - Rover.pos[0]
    target_yaw = 180 / pi * atan2(dy, dx)

    Rover.throttle_set = 0.6
    forward(Rover)
    diff = angle_diff(target_yaw, Rover.yaw)
    print('yaw = {}, target = {}, diff = {}'.format(Rover.yaw, target_yaw, diff))
    if abs(diff) > 50:
        print('Rover not following path - forcing alignment')
        aligning = True
        rotate_to_angle(Rover, target_yaw)
    else:
        print('follow_wall')
        follow_wall(Rover, target_yaw)


def forward(Rover):
    Rover.throttle = min(Rover.max_vel - Rover.vel, Rover.throttle_set)
    Rover.brake = 0


def follow_wall(Rover, target_yaw):
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if Rover.mode == 'forward':
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:
                # If mode is forward, navigable terrain looks good
                # and velocity is below max, then throttle
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    forward(Rover)
                else:  # Else coast
                    Rover.throttle = 0
                    Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                print('direction: {} {}'.format(np.mean(Rover.nav_angles) / 4, -angle_diff(target_yaw, Rover.yaw) / 30))
                Rover.steer = np.clip((np.mean(Rover.nav_angles) / 4 + angle_diff(target_yaw, Rover.yaw) / 10) * 180 / np.pi, -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "stop" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            else:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = Rover.steer > 0 and 15 or -15
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    forward(Rover)
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something
    # even if no modifications have been made to the code
    else:
        forward(Rover)
        Rover.steer = 0

    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True


def decision_step(Rover):
    global aligning
    global target_yaw
    global frames_to_path_trace
    global stuck

    frames_to_path_trace -= 1

    going_home = Rover.samples_found >= Rover.samples_to_find - 1
    if going_home:
        if a_star.distance(Rover.pos, Rover.start_pos) < 10:
            print("Done")
            Rover.throttle = Rover.steer = 0
            Rover.brake = 1
            return Rover

    if (stuck >= 20 or Rover.throttle) and abs(Rover.vel) < 0.2:
        stuck += 1
        if stuck > 100:
            stuck = 0
        elif stuck > 40:
            print('Rover stuck - throttle')
            Rover.throttle = -1
            Rover.brake = 0
            Rover.steer = 0
            return Rover
        elif stuck > 20:
            print('Rover stuck')
            Rover.throttle = 0
            if Rover.vel > 0.2:
                Rover.brake = 1
            else:
                Rover.brake = 0
                Rover.steer = Rover.steer > 0 and 15 or -15
            return Rover
    else:
        stuck = 0

    if Rover.near_sample:
        Rover.throttle = Rover.steer = 0
        Rover.brake = 1
        if Rover.vel == 0 and not Rover.picking_up:
            Rover.send_pickup = True
        return Rover
    elif Rover.rock_angle is not None:
        print("Chasing rock")
        forward(Rover)
        Rover.steer = np.clip(np.mean(Rover.rock_angle) * 180 / np.pi, -15, 15)
        return Rover

    if aligning:
        if abs(angle_diff(target_yaw, Rover.yaw)) < 10:
            aligning = False
        else:
            print("Aligning to target yaw")
            rotate_to_angle(Rover, target_yaw)
            return Rover
    if frames_to_path_trace <= 0:
        path = []
        if not going_home:
            path = path_to_undiscovered(Rover)
        if not path:
            start = (round(Rover.start_pos[1]), round(Rover.start_pos[0]))
            rover_yx = (round(Rover.pos[1]), round(Rover.pos[0]))
            path = a_star.run(rover_yx, start, Rover.navigation_map.real_navigable)
        if path:
            follow_path(Rover, path)
            frames_to_path_trace = len(path)
        else:
            frames_to_path_trace = 10
    else:
        follow_wall(Rover, target_yaw is not None and target_yaw or Rover.yaw)
    return Rover
