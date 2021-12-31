import os
from PIL import Image
import cv2
import math
import numpy as np
from numpy.lib.function_base import extract
from tqdm import tqdm
from random import randint
from statistics import mean
import shutil
import pytesseract


def txt_to_list(txt_path):
    ls = open(txt_path, "r").read()
    return ls


def list2txt(in_list, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as f:
        for item in in_list:
            f.write("%s\n" % item)


def get_lines(lines_in):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines_in]


def merge_lines_pipeline_2(lines, angle_to_merge):
    super_lines_final = []
    super_lines = []
    min_distance_to_merge = 10
    min_angle_to_merge = angle_to_merge

    for line in (lines):
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line2 in group:
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))
                        group.append(line)

                        create_new_group = False
                        group_updated = True
                        break

            if group_updated:
                break

        if (create_new_group):
            new_group = []
            new_group.append(line)

            for idx, line2 in enumerate(lines):
                # check the distance between lines
                if get_distance(line2, line) < min_distance_to_merge:
                    # check the angle between lines       
                    orientation_i = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                    orientation_j = math.atan2((line2[0][1]-line2[1][1]),(line2[0][0]-line2[1][0]))

                    if int(abs(abs(math.degrees(orientation_i)) - abs(math.degrees(orientation_j)))) < min_angle_to_merge: 
                        #print("angles", orientation_i, orientation_j)
                        #print(int(abs(orientation_i - orientation_j)))

                        new_group.append(line2)

                        # remove line from lines list
                        #lines[idx] = False
            # append new group
            super_lines.append(new_group)


    for group in super_lines:
        super_lines_final.append(merge_lines_segments1(group))

    return super_lines_final


def merge_lines_segments1(lines):
    if(len(lines) == 1):
        return lines[0]

    line_i = lines[0]

    # orientation
    orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))

    points = []
    for line in lines:
        points.append(line[0])
        points.append(line[1])

    if (abs(math.degrees(orientation_i)) > 89) and (abs(math.degrees(orientation_i)) < 91):
        #sort by y
        points = sorted(points, key=lambda point: point[1])

    elif (abs(math.degrees(orientation_i)) > 179) and (abs(math.degrees(orientation_i)) < 181):
        #sort by x
        points = sorted(points, key=lambda point: point[0])

    return [points[0], points[len(points)-1]]


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude


def DistancePointLine(px, py, x1, y1, x2, y2):
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine


def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])


    return min(dist1,dist2,dist3,dist4)


def is_contour_good(c):
    # initialize the shape name and approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    # a rectangle
    if len(approx) <=4:
        return True
    else:
        False


def process_lines(image_src, save_path, plot_flag):
    src = cv2.imread(image_src)
    if len(src.shape) != 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src
    
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    vertical = np.copy(bw)
    horizontal = np.copy(bw)
    morph = vertical * 0

    rows, columns = src.shape[:2]

    # size to lookout on vertical axis
    verticalsize = 50
    horizontalsize = 50
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    detect_vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, vertical_kernel, iterations = 2)
    cnts_vertical = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts_vertical) == 2:
        cnts_vertical = cnts_vertical[0]
    else: 
        cnts_vertical = cnts_vertical[1]


    detect_horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontal_kernel, iterations = 2)
    cnts_horizontal = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts_horizontal) == 2:
        cnts_horizontal = cnts_horizontal[0]
    else: 
        cnts_horizontal = cnts_horizontal[1]
    
    cnts_total = cnts_vertical + cnts_horizontal

    for cnt in cnts_total:
        if is_contour_good(cnt):
            cv2.drawContours(morph, [cnt], -1, (255), thickness=8)
    
    border_pixels = int(0.01 * rows) # 1% of height of image
    morph[0:rows, 0:border_pixels] = 0 # left border
    morph[0:rows, columns-border_pixels:columns] = 0 # right border
    morph[0:border_pixels, 0:columns] = 0 # top border
    morph[rows-border_pixels:rows, 0:columns] = 0 # bottom border

    lines = cv2.HoughLinesP(morph, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=100)   # maxLineGap - max allowed gap between line segments to treat them as a single line


    # merge lines

    # prepare
    _lines = []
    for _line in get_lines(lines):
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])


    vertical_line_length_list = []
    horizontal_line_length_list = []
    # sort
    _lines_x = []
    _lines_y = []
    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 89) and (abs(math.degrees(orientation_i)) < 91):
            line_length = abs(line_i[1][1] - line_i[0][1])
            vertical_line_length_list.append(line_length)
        
        elif (abs(math.degrees(orientation_i)) > 179) and (abs(math.degrees(orientation_i)) < 181):
            line_length = abs(line_i[1][0] - line_i[0][0])
            horizontal_line_length_list.append(line_length)


    internal_threshold_vertical = mean(vertical_line_length_list)
    internal_threshold_horizontal = mean(horizontal_line_length_list)

    for line_i in _lines:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 89) and (abs(math.degrees(orientation_i)) < 91):
            line_length = abs(line_i[1][1] - line_i[0][1])            
            if line_length >= internal_threshold_vertical:
                _lines_y.append(line_i)
        
        elif (abs(math.degrees(orientation_i)) > 179) and (abs(math.degrees(orientation_i)) < 181):
            line_length = abs(line_i[1][0] - line_i[0][0])
            if line_length >= internal_threshold_horizontal:
                _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x, angle_to_merge = 180)
    merged_lines_y = merge_lines_pipeline_2(_lines_y, angle_to_merge = 90)

    merged_lines_all = []
    merged_lines_all.extend(merged_lines_x)
    merged_lines_all.extend(merged_lines_y)
    # print("process groups lines", len(_lines), len(merged_lines_all))
    img_merged_lines = cv2.imread(image_src)
    bw = img_merged_lines * 0
    
    all_lines_dict = {'vertical' : {}, 'horizontal': {}}
    for line in merged_lines_all:
        orientation_line = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
        angle_of_line = abs(math.degrees(orientation_line))
        if (angle_of_line > 89 and angle_of_line < 91):
            # vertical line
            line_length = abs(line[1][1] - line[0][1])
            if line_length >= 400:
                cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
                if str(line_length) in all_lines_dict['vertical'].keys():
                    rand_no = randint(1000, 9999)
                    key_vertical = str(line_length) + '.' + str(rand_no)
                else:
                    key_vertical = str(line_length)
                all_lines_dict['vertical'][key_vertical] = [int(line[0][0]), int(line[0][1]), int(line[1][0]), int(line[1][1])]
        
        elif(angle_of_line > 179 and angle_of_line < 181):
            # horizontal line
            line_length = abs(line[1][0] - line[0][0])
            if line_length >= 400:
                cv2.line(img_merged_lines, (line[0][0], line[0][1]), (line[1][0],line[1][1]), (0,0,255), 6)
                if str(line_length) in all_lines_dict['horizontal'].keys():
                    rand_no = randint(1000, 9999)
                    key_horizontal = str(line_length) + '.' + str(rand_no)
                else:
                    key_horizontal = str(line_length)
                all_lines_dict['horizontal'][key_horizontal] = [int(line[0][0]), int(line[0][1]), int(line[1][0]), int(line[1][1])]
    print('plot_flag',plot_flag)
    print('save_path',save_path)
    print('img_merged_lines',img_merged_lines)
    if plot_flag:
        cv2.imwrite(save_path, img_merged_lines)
    # json_path = save_path.replace('.jpg', '.json')
    # with open(json_path, 'w+') as sjf:
    #     json.dump(all_lines_dict, sjf)
    return merged_lines_all, all_lines_dict


# cropping functions


def get_iou(bb1, bb2, overlap_threshold=0.7):

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return False

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    min_area = min(bb1_area, bb2_area)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    if intersection_area >= (overlap_threshold * min_area):
        return True
    else:
        return False


def count_common_of_2_lists(list1, list2):
    a_set = set(list1)
    b_set = set(list2)
    common = a_set & b_set
    if common:
        return len(common) - 1, common, True
    else:
        return 0, None, False


def count_common_from_2_lines(line1, line2):
    list1 = list(range(line1[1], line1[3]+1))
    list2 = list(range(line2[1], line2[3]+1))
    len_line = min(len(list2), len(list1))
    a_set = set(list1)
    b_set = set(list2)
    common = a_set & b_set
    if common:
        if (len(common) - 1) >= (0.8 * len_line):
            return True
        else:
            return False
    else:
        return False


def count_common_from_2_horizontal_lines(line1, line2):
    list1 = list(range(line1[0], line1[2]+1))
    list2 = list(range(line2[0], line2[2]+1))
    len_line = min(len(list2), len(list1))
    a_set = set(list1)
    b_set = set(list2)
    common = a_set & b_set
    if common:
        if (len(common) - 1) >= (0.95 * len_line):
            return True
        else:
            return False
    else:
        return False


def sorted_list_to_dict(sorted_list):
    sorted_dict = {}
    for ele in sorted_list:
        sorted_dict[ele[0]] = ele[1]
    return sorted_dict


def join_nearby_vertical_lines(sorted_vertical):
    # sorted vertical - dictionary of lines from left to right
    new_sorted_vertical = {}
    sorted_vertical_backup = sorted_vertical
    keys_list = list(sorted_vertical.keys())
    delete_current_flag = False
    interchange_dict_flag = False
    for ix, v_len in enumerate(keys_list):
        if interchange_dict_flag:
            sorted_vertical = sorted_vertical_backup
            interchange_dict_flag = False
        line_coords = sorted_vertical[v_len]
        if ix == 0:
            new_sorted_vertical[v_len] = line_coords
        else:
            # check if two lines are within 100 px and are 70% overlapping too then remove the small line
            prev_line_key = keys_list[ix-1]
            if prev_line_key not in list(new_sorted_vertical.keys()):
                sorted_vertical = new_sorted_vertical
                prev_line_key = list(new_sorted_vertical.keys())[-1]
                interchange_dict_flag = True
            if abs(sorted_vertical[prev_line_key][0] - line_coords[0]) < 100:
                len_two_lines_in_consideration = (sorted_vertical[prev_line_key][3] - sorted_vertical[prev_line_key][1], line_coords[3] - line_coords[1])
                line_length_small = min(len_two_lines_in_consideration)
                small_line_index = len_two_lines_in_consideration.index(min(len_two_lines_in_consideration))    
                l1_temp = list(range(sorted_vertical[prev_line_key][1], sorted_vertical[prev_line_key][3]+1)) # list of y coords of line 1
                l2_temp = list(range(line_coords[1], line_coords[3]+1)) # list of y coords of line 2
                if small_line_index == 0:
                    small_lines_coords = l1_temp
                else:
                    small_lines_coords = l2_temp
                common_count_ret = count_common_of_2_lists(l1_temp, l2_temp)
                if common_count_ret[2] == True:
                    common_count = common_count_ret[0]
                    common_elements = common_count_ret[1]
                else:
                    common_count = 0
                if common_count > (0.7 * line_length_small):
                    # if set(small_lines_coords).issubset(set(common_elements)):
                    if count_common_of_2_lists(small_lines_coords, common_elements)[0] > (0.8 * len(small_lines_coords)):
                        if sorted_vertical[prev_line_key][3] - sorted_vertical[prev_line_key][1] > line_coords[3] - line_coords[1]:
                            new_line_coords = sorted_vertical[prev_line_key]
                            new_line_length = sorted_vertical[prev_line_key][3] - sorted_vertical[prev_line_key][1]
                            try:
                                del new_sorted_vertical[prev_line_key]
                            except:
                                delete_current_flag = True
                        else:
                            new_line_coords = line_coords
                            new_line_length = line_coords[3] - line_coords[1]
                            try:
                                del new_sorted_vertical[prev_line_key]
                            except:
                                delete_current_flag = True
                        
                        new_sorted_vertical[str(new_line_length)] = new_line_coords
                        if delete_current_flag:
                            del new_sorted_vertical[v_len]
                            delete_current_flag = False
                    else:
                        new_sorted_vertical[v_len] = line_coords
                else:
                    new_sorted_vertical[v_len] = line_coords
            else:
                new_sorted_vertical[v_len] = line_coords
    return new_sorted_vertical


def crop_cutout_by_two_horizontal_lines(line1, line2):
    y1 = line1[1]
    y2 = line2[3]

    if line1[0] >= line2[0]:
        x1 = line1[0]
    else:
        x1 = line2[0]
    
    if line1[2] <= line2[2]:
        x2 = line1[2]
    else:
        x2 = line2[2]

    return x1, y1, x2+1, y2+1


def crop_column_by_two_vertical_lines(line1, line2):
    
    x1 = line1[0]
    x2 = line2[2]
    
    if line1[1] >= line2[1]:
        y1 = line1[1]
    else:
        y1 = line2[1]

    if line1[3] <= line2[3]:
        y2 = line1[3]
    else:
        y2 = line2[3]
    
    return x1, y1, x2+1, y2+1


def sort_vertical_if_image_plane_wrong(sorted_vertical_dict):
    new_sorted_vertical_dict = {}
    skip_line_flag = False
    cnt_change = 0
    keys_list = list(sorted_vertical_dict.keys())
    for v_len in keys_list:
        try:
            if skip_line_flag == True:
                skip_line_flag = False
                cnt_change += 1
                continue
            line1_coords = sorted_vertical_dict[v_len]
            line2_coords = sorted_vertical_dict[keys_list[keys_list.index(v_len) + 1]]
            if abs(line1_coords[0] - line2_coords[0]) <= 150:
                if line2_coords[1] < line1_coords[1]:
                    new_sorted_vertical_dict[keys_list[keys_list.index(v_len) + 1]] = line2_coords
                    new_sorted_vertical_dict[v_len] = line1_coords
                    skip_line_flag = True
                else:
                    new_sorted_vertical_dict[v_len] = line1_coords
            else:
                new_sorted_vertical_dict[v_len] = line1_coords
        except:
            new_sorted_vertical_dict[v_len] = line1_coords
    return new_sorted_vertical_dict, cnt_change


def crop_sub_part_of_column(prev_line, line_coords, vertical_lines_coords, column_count, sub_line_flag, cropped_rectangular_regions_list, column_number_backup, cutouts_res_path, sub_part_count, v_len, ix, img, extension):
    if prev_line[1] > line_coords[3]:
        x = 0
        y = 2
    else:
        x = 2
        y = 0

    if abs(prev_line[x] - line_coords[y]) < 50 and int(float(v_len)) > 500: # add sub part of the same column if found
        # line_length_temp = min(prev_line[3] - prev_line[1], line_coords[3] - line_coords[1])
        l1_temp = list(range(prev_line[1], prev_line[3]+1)) # list of y coords of line 1
        l2_temp = list(range(line_coords[1], line_coords[3]+1)) # list of y coords of line 2
        common_count_ret = count_common_of_2_lists(l1_temp, l2_temp) # len of common coords on y axis of both lines
        if common_count_ret[2] == True:
            common_count = common_count_ret[0]
        else:
            common_count = 0
        if common_count == 0:   # there should be nothing common in y axis if the line is of same column but broken from somewhere
            temp_len = len(vertical_lines_coords)
            try:
                last_line_min_gap = 600
                # check 1: the lines are column distance apart AND check 2: the line is parallel to a previous line with majority y axis being common AND check3: the previous lines should not be from previous to previous columns
                if abs(vertical_lines_coords[ix-2][0] - line_coords[0]) >= last_line_min_gap and count_common_from_2_lines(vertical_lines_coords[ix-2], line_coords) and vertical_lines_coords[ix-2][5] == column_count - 1:
                    line1 = vertical_lines_coords[ix-2]
                    sub_line_flag = True
                elif abs(vertical_lines_coords[ix-3][0] - line_coords[0]) >= last_line_min_gap and count_common_from_2_lines(vertical_lines_coords[ix-3], line_coords) and vertical_lines_coords[ix-3][5] == column_count - 1:
                    line1 = vertical_lines_coords[ix-3]
                    sub_line_flag = True
                elif abs(vertical_lines_coords[ix-4][0] - line_coords[0]) >= last_line_min_gap and count_common_from_2_lines(vertical_lines_coords[ix-4], line_coords) and vertical_lines_coords[ix-4][5] == column_count - 1:
                    line1 = vertical_lines_coords[ix-4]
                    sub_line_flag = True
                elif abs(vertical_lines_coords[ix-5][0] - line_coords[0]) >= last_line_min_gap and count_common_from_2_lines(vertical_lines_coords[ix-5], line_coords) and vertical_lines_coords[ix-5][5] == column_count - 1:
                    line1 = vertical_lines_coords[ix-5]
                    sub_line_flag = True
                else:
                    sub_line_flag = False
                    print('###################')
            except:
                print('###################')
            if sub_line_flag:
                sub_line_flag = False
                x1, y1, x2, y2 = crop_column_by_two_vertical_lines(line1, line_coords)
                iou = False
                for rect in cropped_rectangular_regions_list:
                    iou = get_iou(rect, [x1, y1, x2, y2])
                    if iou:
                        break
                if not iou:
                    column = img[y1:y2, x1:x2]
                    if column_number_backup == column_count:
                        sub_part_count += 1
                    else:
                        sub_part_count = 2
                    vertical_lines_coords.append(line_coords)
                    column_number_backup = column_count
                    column_name = 'column_' + str(column_count) + '_' + str(sub_part_count) + extension
                    column_path = os.path.join(cutouts_res_path, column_name)
                    cv2.imwrite(column_path, column)
                    line_coords.append(False)
                    line_coords.append(column_count)
                    cropped_rectangular_regions_list.append([x1, y1, x2, y2])

    return prev_line, line_coords, vertical_lines_coords, column_count, sub_line_flag, cropped_rectangular_regions_list, column_number_backup, cutouts_res_path, sub_part_count, v_len, ix


def crop_column_main(prev_line, line_coords, cropped_rectangular_regions_list, column_count, first_line_added, img, vertical_lines_coords, cutouts_res_path, extension, crop_another_line_flag_list) :
    image_height, image_width = img.shape[:2]
    if abs(line_coords[0] - image_width) <= 250:
        line_coords = [image_width, 0, image_width, image_height]
    line_length_temp = min(prev_line[3] - prev_line[1], line_coords[3] - line_coords[1]) # length of shorter line from 2 in consideration
    l1_temp = list(range(prev_line[1], prev_line[3]+1)) # list of y coords of line 1
    l2_temp = list(range(line_coords[1], line_coords[3]+1)) # list of y coords of line 2
    common_count_ret = count_common_of_2_lists(l1_temp, l2_temp) # len of common coords on y axis of both lines
    if common_count_ret[2] == True:
        common_count = common_count_ret[0]
    else:
        common_count = 0
    if common_count > (0.3 * line_length_temp): # checking if two lines have at least 30% y axis common
        x1, y1, x2, y2 = crop_column_by_two_vertical_lines(prev_line, line_coords)
        # print(x1, y1, x2, y2)
        # print('*****************')
        iou = False
        for rect in cropped_rectangular_regions_list:
            iou = get_iou(rect, [x1, y1, x2, y2])
            if iou:
                break
        if not iou:
            vertical_lines_coords.append(line_coords)
            first_line_added = True
            column = img[y1:y2, x1:x2]
            column_count += 1
            column_name = 'column_' + str(column_count) + extension
            if len(crop_another_line_flag_list) != 0:
                column_number_backup_another, sub_part_count_another = crop_another_line_flag_list
                column_count -= 1
                if column_number_backup_another == column_count:
                    sub_part_count_another += 1
                else:
                    sub_part_count_another = 2
                column_name = 'column_' + str(column_count) + '_' + str(sub_part_count_another) + extension
                column_number_backup_another = column_count
            else:
                column_number_backup_another = 0
                sub_part_count_another = 0
            column_path = os.path.join(cutouts_res_path, column_name)
            cv2.imwrite(column_path, column)
            line_coords.append(True)
            line_coords.append(column_count)
            cropped_rectangular_regions_list.append([x1, y1, x2, y2])
        else:
            column_number_backup_another = 0
            sub_part_count_another = 2
    else:
        column_number_backup_another = 0
        sub_part_count_another = 2
    return prev_line, line_coords, cropped_rectangular_regions_list, column_count, first_line_added, vertical_lines_coords, column_number_backup_another, sub_part_count_another


def cropping_algorithm_main(img, vertical_dict, horizontal_dict, cutouts_res_path, extension):

    # sort them by their occurences from left to right (vertical) and top to bottom (horizontal)
    sorted_vertical_list = sorted(vertical_dict.items(), key=lambda x: x[1][0])
    sorted_horizontal_list = sorted(horizontal_dict.items(), key=lambda x: x[1][1])

    sorted_vertical = sorted_list_to_dict(sorted_vertical_list)
    sorted_horizontal = sorted_list_to_dict(sorted_horizontal_list)

    sorted_vertical = join_nearby_vertical_lines(join_nearby_vertical_lines(sorted_vertical))
    sorted_vertical, iteration = sort_vertical_if_image_plane_wrong(sorted_vertical)
    for i in range(iteration+1):
        sorted_vertical, _ = sort_vertical_if_image_plane_wrong(sorted_vertical)

    image_height, image_width = img.shape[:2]

    # add first line in the vertical line list
    vertical_line_1 = [0, 0, 0, image_height, True, 0]
    # the 5th element : bool : denotes that whether the line is a different column or no
    # if True: this line is a new column, if False: this is another vertical line in the same column

    # the 6th element shows the the vertical line / column number
    vertical_lines_coords = [vertical_line_1]
    first_line_added = False
    column_count = 0
    sub_part_count = 2
    sub_part_count_another = 2
    column_number_backup = 0
    column_number_backup_another = 0
    cropped_rectangular_regions_list = []
    sub_line_flag = False

    # add last line manually in the vertical dict
    sorted_vertical[str(image_height)] = [image_width, 0, image_width, image_height, True, -1]
    for v_len in sorted_vertical.keys():
        line_coords = sorted_vertical[v_len]
        if first_line_added:
            line_gap = 600
        else:
            line_gap = 700
        
        ix = len(vertical_lines_coords)
        
        try:
            if vertical_lines_coords[ix-1][4] == True:
                prev_line = vertical_lines_coords[ix-1]
            elif vertical_lines_coords[ix-2][4] == True:
                prev_line = vertical_lines_coords[ix-2]
            elif vertical_lines_coords[ix-3][4] == True:
                prev_line = vertical_lines_coords[ix-3]
            elif vertical_lines_coords[ix-4][4] == True:
                prev_line = vertical_lines_coords[ix-4]
            else:
                prev_line = vertical_lines_coords[0]
        except:
            prev_line = vertical_lines_coords[0]
        
        # check block for if the prev column line had multiple lines, then the current column would also have multiple subparts
        col_name = prev_line[5]
        crop_another_line = []
        temp_cnt = 0
        vertical_lines_coords_backup = [i for i in vertical_lines_coords if i != prev_line]
        for line in vertical_lines_coords_backup:
            if line[5] == col_name:
                crop_another_line.append(line)
        # close
        
        if abs(prev_line[0] - line_coords[0]) > line_gap and int(float(v_len)) > 500: # check if lines are 600-700 px apart and line_length is > 500
            prev_line, line_coords, cropped_rectangular_regions_list, column_count, \
            first_line_added, vertical_lines_coords, column_number_backup_another, sub_part_count_another = crop_column_main(prev_line, \
            line_coords, cropped_rectangular_regions_list, column_count, first_line_added, img, \
            vertical_lines_coords, cutouts_res_path, extension, crop_another_line_flag_list = [])

        else:
            prev_line, line_coords, vertical_lines_coords, column_count, sub_line_flag, cropped_rectangular_regions_list, \
            column_number_backup, cutouts_res_path, sub_part_count, v_len, ix = crop_sub_part_of_column(prev_line, line_coords, \
            vertical_lines_coords, column_count, sub_line_flag, cropped_rectangular_regions_list, column_number_backup, \
            cutouts_res_path, sub_part_count, v_len, ix, img, extension)

        if len(crop_another_line) != 0:
            for prev_line in crop_another_line:
                if abs(prev_line[0] - line_coords[0]) > line_gap and int(float(v_len)) > 500:
                    prev_line, line_coords, cropped_rectangular_regions_list, column_count, first_line_added, \
                    vertical_lines_coords, column_number_backup_another, sub_part_count_another = crop_column_main(prev_line, \
                    line_coords, cropped_rectangular_regions_list, column_count, first_line_added, img, vertical_lines_coords, \
                    cutouts_res_path, extension, crop_another_line_flag_list=[column_number_backup_another, sub_part_count_another])


    # print(sorted_horizontal)
    cutout_count = 0
    sorted_horizontal_temp = sorted_horizontal
    for i in list(sorted_horizontal_temp.keys()):
        if int(float(i)) < 500:
            del sorted_horizontal[i]

    horizontal_keys_list = list(sorted_horizontal.keys())
    for h_line in horizontal_keys_list:
        line_coords = sorted_horizontal[h_line]
        for h_line2 in horizontal_keys_list:
            line_coords2 = sorted_horizontal[h_line2]
            # check if lines are minimum 500 px apart
            if line_coords2[1] - line_coords[1] > 500:
                # check if length of both lines is similar (90% same length)
                small_line_len = min(int(float(h_line)), int(float(h_line2)))
                big_line_len = max(int(float(h_line)), int(float(h_line2)))
                if small_line_len >= int(0.9 * big_line_len):
                    # check if the coordinates are overlapping on x axis
                    if count_common_from_2_horizontal_lines(line_coords, line_coords2):
                        x1, y1, x2, y2 = crop_cutout_by_two_horizontal_lines(line_coords, line_coords2)
                        iou = False
                        for rect in cropped_rectangular_regions_list:
                            iou = get_iou(rect, [x1, y1, x2, y2], overlap_threshold=0.4)
                            if iou:
                                break
                        if not iou:
                            cutout_img = img[y1:y2, x1:x2]
                            cutout_count += 1
                            cutout_name = 'cutout_' + str(cutout_count) + extension
                            cutout_save_path = os.path.join(cutouts_res_path, cutout_name)
                            cv2.imwrite(cutout_save_path, cutout_img)
                            cropped_rectangular_regions_list.append([x1, y1, x2, y2])
    return

# running arguments
# data_path = r"D:\garvit\image_processing\1_newspaper_columns\jp2_images\long"
# results_path = r"C:\Users\garvi\OneDrive\Desktop\pytess_check\cutouts"
# plot_results_path = r"C:\Users\garvi\OneDrive\Desktop\pytess_check\plot"
# exception_images = r"C:\Users\garvi\OneDrive\Desktop\pytess_check\exception"
# txt_results_path = r"C:\Users\garvi\OneDrive\Desktop\pytess_check\txts"
# tesseract_installed_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
plot_flag = True
save_ext = 'jp2' # type jpg or jp2

# running arguments
data_path = r"/Users/aki/Downloads/OCR/Rough/sample"
results_path = r"/Users/aki/Downloads/OCR/Rough/cutouts"
plot_results_path = r"/Users/aki/Downloads/OCR/Rough/plots"
exception_images = r"/Users/aki/Downloads/OCR/Rough/exception"
txt_results_path = r"/Users/aki/Downloads/OCR/Rough/txts"
# tesseract_installed_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# do not change after this!!

# pytesseract.pytesseract.tesseract_cmd = tesseract_installed_path
for img_name in tqdm(sorted(os.listdir(data_path))):
    if save_ext == 'jpg':
        extension = '.jpg'
    elif save_ext == 'jp2':
        extension = '.jp2 '
    print('extension',extension)
    print('img_name',img_name)
    if img_name == ".DS_Store":                 #aki - extra added for mac
        continue
    try:
        img_path = os.path.join(data_path, img_name)
        file_name = img_name.split(".")[0]
        plot_save_path = os.path.join(plot_results_path, file_name + extension)
        print('plot_save_path',plot_save_path)
        merged_lines, all_lines_dict = process_lines(img_path, plot_save_path, plot_flag)
        vertical_dict = all_lines_dict['vertical']
        horizontal_dict = all_lines_dict['horizontal']
        img = cv2.imread(img_path)
        cutouts_save_path = os.path.join(results_path, file_name)
        os.makedirs(cutouts_save_path, exist_ok=True)
        cropping_algorithm_main(img, vertical_dict, horizontal_dict, cutouts_save_path, extension)

        # pytess code:
        for splits in sorted(os.listdir(cutouts_save_path)):
            name, extension = os.path.splitext(cutouts_save_path+"/"+splits)
            if extension == ".jp2 ":                             #aki - extra added
                print('splits',splits)
                split_path = os.path.join(cutouts_save_path, splits)
                split_img = np.array(Image.open(split_path))
                text = pytesseract.image_to_string(split_img)
                if extension == '.jp2 ':
                    replace_word = '.jp2'
                else:
                    replace_word = extension
                split_txt_save = split_path.replace(replace_word, '.txt')
                with open(split_txt_save,'w') as f:
                    f.write(str(text))
        whole_txt = []
        for txt_split in sorted(os.listdir(cutouts_save_path)):
            name, extension = os.path.splitext(cutouts_save_path+"/"+txt_split)
            if extension == ".txt ":
                txt_split_path = os.path.join(cutouts_save_path, txt_split)
                txt_list = txt_to_list(txt_split_path)
                whole_txt.append(txt_list)
        list2txt(whole_txt, os.path.join(txt_results_path, file_name + '.txt'))

    except Exception as e:                          # add this exception code above so that it doesnt end in exception case
        print(e)
        if os.path.exists(exception_images+"/"+img_name):           #aki - if exception file exist remove it.
            os.remove(exception_images+"/"+img_name)
        shutil.move(img_path, exception_images)