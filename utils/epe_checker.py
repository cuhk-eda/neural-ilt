import numpy as np
import cv2 as cv
import os
import copy


def show_img(img, title, wait_key=0, output_path=None):
    cv.imshow(title, img)
    cv.waitKey(wait_key)
    if output_path is not None:
        cv.imwrite(output_path, img)
    return


def find_all_contours(gray_img, contour_approx=cv.CHAIN_APPROX_SIMPLE):
    cnts, hier = cv.findContours(gray_img, cv.RETR_TREE, contour_approx)
    return cnts, hier


def check_poly(poly):
    r'''
    Remove the reduant pts in concave vertexs of a ploygon
    '''
    legal_poly = []
    for i, pt in enumerate(poly):
        cur_pt = pt[0]
        next_pt = None
        if i == len(poly)-1:
            next_pt = poly[0][0]
        else:
            next_pt = poly[i+1][0]
        if cur_pt[0] != next_pt[0] and cur_pt[1] != next_pt[1]:
            fixed_cur_pt = copy.deepcopy(cur_pt)
            last_pt = poly[i-1][0]
            if last_pt[0] == cur_pt[0] and last_pt[1] != cur_pt[1]:
                fixed_cur_pt[1] = next_pt[1]
            elif last_pt[1] == cur_pt[1] and last_pt[0] != cur_pt[0]:
                fixed_cur_pt[0] = next_pt[0]
            else:
                raise NotImplementedError(
                    f"Error: Illegal vertex {last_pt}->{cur_pt}->{next_pt} detected!")
            legal_poly.append(fixed_cur_pt)
        else:
            legal_poly.append(cur_pt)
    legal_poly = np.array(legal_poly)
    for i, pt in enumerate(legal_poly):
        if pt[0] != legal_poly[i-1][0] and pt[1] != legal_poly[i-1][1]:
            raise NotImplementedError("Error: Illegal ploygon detected!")
    return legal_poly


def get_epe_checkpoints(layout, check_stepsize=40):
    r'''
    Rules for EPE checkpoint insertions (DOI： 10.1109/ICCAD.2013.6691131):
    (i)   Every polygon will be processed edge by edge 
    (ii)  EPE is always measured perpendicular to the edge i.e. 
          Manhattan distance 
    (iii) The number of points per edge is the minimum number 
          needed to insure that the distance between any two points or any 
          point and the start or end of the edge is <= check_stepsize nm (check_stepsize pixels) 
    (iv)  Points are to be distributed uniformly and symmetrically across 
          the edge subject to the minimum stipulation above. 
    '''
    polys, _ = find_all_contours(layout)
    checkpoints = {'v_pts': [], 'h_pts': [], 'rv_pts': [],
                   'lv_pts': [], 'th_pts': [], 'bh_pts': [], 'all_pts': []}
    for poly in polys:
        legal_poly = check_poly(poly)
        for i, pt in enumerate(legal_poly):
            next_pt = None
            cp_key = None
            if i == len(poly)-1:
                next_pt = legal_poly[0]
            else:
                next_pt = legal_poly[i+1]
            if abs(pt[0] - next_pt[0]) <= 1 and abs(pt[1] - next_pt[1]) <= 1:
                continue
            if pt[0] == next_pt[0]:  # vertical
                check_pt = None
                if pt[1] < next_pt[1]:
                    cp_key = 'lv_pts'
                else:
                    cp_key = 'rv_pts'
                if abs(pt[1]-next_pt[1]) <= 2*check_stepsize:
                    new_y = int((pt[1] + next_pt[1])/2)
                    check_pt = [pt[0], new_y]
                    checkpoints[cp_key].append(check_pt)
                    checkpoints['v_pts'].append(check_pt)
                    checkpoints['all_pts'].append(check_pt)
                else:
                    start = None
                    end = None
                    if pt[1] < next_pt[1]:
                        start = pt[1] + check_stepsize
                        end = next_pt[1] - check_stepsize
                    else:
                        start = next_pt[1] + check_stepsize
                        end = pt[1] - check_stepsize
                    while start < end:
                        check_pt = [pt[0], start]
                        checkpoints[cp_key].append(check_pt)
                        checkpoints['v_pts'].append(check_pt)
                        checkpoints['all_pts'].append(check_pt)
                        start = start + check_stepsize
                    check_pt = [pt[0], end]
                    checkpoints[cp_key].append(check_pt)
                    checkpoints['v_pts'].append(check_pt)
                    checkpoints['all_pts'].append(check_pt)
            elif pt[1] == next_pt[1]:  # horizontal
                if pt[0] < next_pt[0]:
                    cp_key = 'bh_pts'
                else:
                    cp_key = 'th_pts'
                check_pt = None
                if abs(pt[0]-next_pt[0]) <= 2*check_stepsize:
                    new_x = int((pt[0] + next_pt[0])/2)
                    check_pt = [new_x, pt[1]]
                    checkpoints[cp_key].append(check_pt)
                    checkpoints['h_pts'].append(check_pt)
                    checkpoints['all_pts'].append(check_pt)
                else:
                    start = None
                    end = None
                    if pt[0] < next_pt[0]:
                        start = pt[0] + check_stepsize
                        end = next_pt[0] - check_stepsize
                    else:
                        start = next_pt[0] + check_stepsize
                        end = pt[0] - check_stepsize
                    while start < end:
                        check_pt = [start, pt[1]]
                        checkpoints[cp_key].append(check_pt)
                        checkpoints['h_pts'].append(check_pt)
                        checkpoints['all_pts'].append(check_pt)
                        start = start + check_stepsize
                    check_pt = [end, pt[1]]
                    checkpoints[cp_key].append(check_pt)
                    checkpoints['h_pts'].append(check_pt)
                    checkpoints['all_pts'].append(check_pt)
            else:
                raise NotImplementedError("Error: Illegal ploygon detected!")

    return checkpoints


def calc_epe_violations(wafer, checkpoints, epe_threshold=15):
    r'''
    Simple rules are used to determine the locations to measure edge 
    placement error (EPE). The constraint on EPE is <= th_epe (nm), 
    i.e., EPE Violations = Number of sites where measured EPE >= th_epe+1 (nm)
    '''
    epe_violation_cnt = 0
    violations = []
    for check_pt in checkpoints['rv_pts']:
        if wafer[check_pt[1]][check_pt[0]+epe_threshold+1] != 0 or wafer[check_pt[1]][check_pt[0]-epe_threshold-1] == 0:
            epe_violation_cnt += 1
            violations.append(check_pt)
    for check_pt in checkpoints['lv_pts']:
        if wafer[check_pt[1]][check_pt[0]+epe_threshold+1] == 0 or wafer[check_pt[1]][check_pt[0]-epe_threshold-1] != 0:
            epe_violation_cnt += 1
            violations.append(check_pt)
    for check_pt in checkpoints['th_pts']:
        if wafer[check_pt[1]+epe_threshold+1][check_pt[0]] == 0 or wafer[check_pt[1]-epe_threshold-1][check_pt[0]] != 0:
            epe_violation_cnt += 1
            violations.append(check_pt)
    for check_pt in checkpoints['bh_pts']:
        if wafer[check_pt[1]+epe_threshold+1][check_pt[0]] != 0 or wafer[check_pt[1]-epe_threshold-1][check_pt[0]] == 0:
            epe_violation_cnt += 1
            violations.append(check_pt)
    return epe_violation_cnt, violations


def report_epe_violations(wafer, checkpoints, epe_threshold=15):
    r'''
    Simple rules are used to determine the locations to measure edge 
    placement error (EPE). The constraint on EPE is <= th_epe (nm), 
    i.e., EPE Violations = Number of sites where measured EPE >= th_epe+1 (nm)
    '''
    epe_violation_cnt = 0
    for check_pt in checkpoints['rv_pts']:
        if wafer[check_pt[1]][check_pt[0]+epe_threshold+1] != 0 or wafer[check_pt[1]][check_pt[0]-epe_threshold-1] == 0:
            epe_violation_cnt += 1
    for check_pt in checkpoints['lv_pts']:
        if wafer[check_pt[1]][check_pt[0]+epe_threshold+1] == 0 or wafer[check_pt[1]][check_pt[0]-epe_threshold-1] != 0:
            epe_violation_cnt += 1
    for check_pt in checkpoints['th_pts']:
        if wafer[check_pt[1]+epe_threshold+1][check_pt[0]] == 0 or wafer[check_pt[1]-epe_threshold-1][check_pt[0]] != 0:
            epe_violation_cnt += 1
    for check_pt in checkpoints['bh_pts']:
        if wafer[check_pt[1]+epe_threshold+1][check_pt[0]] != 0 or wafer[check_pt[1]-epe_threshold-1][check_pt[0]] == 0:
            epe_violation_cnt += 1
    return epe_violation_cnt


def epe_eval(root, design, epe_threshold=15, check_stepsize=40, draw=False):
    design_file_name = design + '_0_mask.png'
    wafer_file_name = design + '_0_mask_res.png'
    layout_path = os.path.join(root, 'dataset/ibm_opc_test', design_file_name)
    wafer_path = os.path.join(root, 'output/refine_litho_out', wafer_file_name)
    layout = cv.imread(layout_path)
    wafer = cv.imread(wafer_path)
    layout = cv.cvtColor(layout, cv.COLOR_BGR2GRAY)
    wafer = cv.cvtColor(wafer, cv.COLOR_BGR2GRAY)

    check_pts = get_epe_checkpoints(layout, check_stepsize)
    epe_violation_cnt, violations = calc_epe_violations(
        wafer, checkpoints=check_pts)
    L2_error = cv.bitwise_xor(layout, wafer)
    L2_error = L2_error / L2_error.max()

    print(
        f"{design}: #EPEV = {epe_violation_cnt} out of {len(check_pts['all_pts'])} checkpoints, L2 = {L2_error.sum()}")
    if draw:
        wafer_rgb = cv.cvtColor(wafer, cv.COLOR_GRAY2RGB)
        polys, _ = find_all_contours(layout)
        for x in range(len(polys)):
            cv.drawContours(wafer_rgb, polys, x,
                            color=(255, 0, 255), thickness=2)
        for vio in violations:
            cv.drawMarker(wafer_rgb, (vio[0], vio[1]), (0, 0, 255),
                          markerType=cv.MARKER_TILTED_CROSS, markerSize=40, thickness=3)
        for cp in check_pts['all_pts']:
            cv.circle(wafer_rgb, (cp[0], cp[1]), 5, (255, 0, 0), -1)
        show_img(wafer_rgb, 'epe_rest')
    return epe_violation_cnt, violations
