def transform2coor(start, goal, planepath='construct_map/coordinate.txt'):
    with open(planepath, "r") as file:
        lines = file.readlines()
        plane_coor = (lines[1: 5])
        plane_coor = [list(eval(x.strip())) for x in plane_coor]
    plane_left_bottom = (plane_coor[0][0], plane_coor[0][-1])
    plane_right_top = (plane_coor[2][0], plane_coor[2][-1])
    start_coor = []
    goal_coor = []

    for array in start:
        start_coor.append(array2coor(array, plane_right_top, plane_left_bottom))

    for array in goal:
        goal_coor.append(array2coor(array, plane_right_top, plane_left_bottom))

    return start_coor, goal_coor


def array2coor(arrays, plane_right_top, plane_left_bottom):
    y = plane_right_top[1] - (arrays[0] + 1) * 0.5
    x = (arrays[1] + 1) * 0.5 + plane_left_bottom[0]

    x = round(x, 3)
    y = round(y, 3)
    return [x, y]
