from DHC.configs import PROJECT_ROOT


def get_plane_coordinate(plane_path=str(PROJECT_ROOT / "construct_map/coordinate.txt")):
    with open(plane_path, "r") as file:
        lines = file.readlines()
        plane_coordinate = (lines[1: 5])
        plane_coordinate = [list(eval(x.strip())) for x in plane_coordinate]
    plane_left_bottom, plane_right_top = (plane_coordinate[0][0], plane_coordinate[0][-1]), (
        plane_coordinate[2][0], plane_coordinate[2][-1])
    return plane_left_bottom, plane_right_top


def array2coordinate(arrays, plane_right_top, plane_left_bottom):
    x = round((arrays[1] + 0.5) * 0.5 + plane_left_bottom[0], 3)
    y = round(plane_right_top[1] - (arrays[0] + 0.5) * 0.5, 3)
    return x, y


def is_in_plane(x, y, plane_left_bottom, plane_right_top):
    x_in = plane_left_bottom[0] <= x <= plane_right_top[0]
    y_in = plane_left_bottom[1] <= y <= plane_right_top[1]
    return x_in and y_in


def grid2coord(position):
    plane_left_bottom, plane_right_top = get_plane_coordinate()
    x, y = array2coordinate(position, plane_right_top, plane_left_bottom)
    x = min(max(x, plane_left_bottom[0]), plane_right_top[0])
    y = min(max(y, plane_left_bottom[1]), plane_right_top[1])
    is_in = is_in_plane(x, y, plane_left_bottom, plane_right_top)
    return x, y, is_in


# 计算向量之间夹角
def cal_cos_between_2_vector(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def cal_distance_between_2_point(first_x, first_y, second_x, second_y):
    return (first_x - second_x) ** 2 + (first_y - second_y) ** 2


def is_next_position_toward_goal(cur_x, cur_y, next_x, next_y, goal_x, goal_y):
    first = (next_x - cur_x, next_y - cur_y)
    second = (goal_x - cur_x, goal_y - cur_y)
    return cal_cos_between_2_vector(*first, *second) > 0


def is_next_position_toward_goal_by_distance(cur_x, cur_y, next_x, next_y, goal_x, goal_y):
    origin_distance = cal_distance_between_2_point(cur_x, cur_y, goal_x, goal_y)
    next_distance = cal_distance_between_2_point(next_x, next_y, goal_x, goal_y)
    return next_distance <= origin_distance


if __name__ == '__main__':
    y = is_next_position_toward_goal_by_distance(1, 0, -3, -4, -2, 9)
    print(y)
