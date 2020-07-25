'''
IOU Tracker

Assuming we have N detections on each frame F, we can track an object (and say they are same) only
if the IOU between j prediction on frame N and i prediction on frame N-1,
is big enough (i.e higher than .90) since that means the bounding boxes are overlapping on
subsequent frames which is almost the case since there's not a lot of movement
between frames specially on videos with high FPS.
There might be cases however when there is occlusion or multiple objects
are overlapping on the same frame.  This is only helpful to a certain point since
once the object we previously located is out of frame, we would lose that reference (ID)
and if it ever appears again, it will be consider as a new object with a different ID.
This is where using image vector or image embeddings or even a classifier might help
but in the particular example of a soccer match is difficult but that's another topic.

For example, assume we have the following frames with its detections:

frame_0 = [(570, 270, 627, 343), (482, 314, 539, 391)]
frame_1 = [(481, 310, 548, 391), (744, 331, 805, 414), (570, 271, 622, 350)]
frame_2 = [(481, 310, 548, 391), (570, 271, 622, 350), (240, 102, 300, 566)]
frame_3 = [(482, 308, 542, 387), (741, 329, 796, 417)]
frame_4 = [(566, 267, 633, 351), (475, 304, 543, 391), (714, 329, 794, 422), (350, 183, 404, 262)]

Start with a Tracker List of Dictionaries which has the following keys:
ID - (number) - ID of the object
BBOXES - (list) - A list of (frame, bbox) associated to that ID

The first frame, you will assign all existing bboxes to an entry on the Tracker.
For the next N frames, you will compare each detection on the frame N to each entry on the Tracker.
For each comparison, you'll calculate the IOU and only consider those who have an
IOU higher than some threshold (i.e. more than .70)
You then pick the highest and add it to the appropriate entry.
Those who did not have a match will be added as a new entry.

FULL EXPECTED OUTPUT

    TRACKER = [
        {
            'id': 0,
            'bboxes': [
                [0, (570, 270, 627, 343)],
                [1, (570, 271, 622, 350)],
                [2, (570, 271, 622, 350)]
            ]
        },
        {
            'id': 1,
            'bboxes': [
                [0, (482, 314, 539, 391)],
                [1, (481, 310, 548, 391)],
                [2, (481, 310, 548, 391)],
                [3, (482, 308, 542, 387)],
                [4, (475, 304, 543, 391)]
            ]
        },
        {
            'id': 2,
            'bboxes': [
                [1, (744, 331, 805, 414)],
            ]
        },
        {
            'id': 3,
            'bboxes': [
                [2, (240, 102, 300, 566)]
            ]
        },
        {
            'id': 4,
            'bboxes': [
                [3, (741, 329, 796, 417)]
                [4, (714, 329, 794, 422)]
            ]
        },
        {
            'id': 5,
            'bboxes': [
                [4, (566, 267, 633, 351)]
            ]
        },
        {
            'id': 6,
            'bboxes': [
                [4, (350, 183, 404, 262)]
            ]
        }
    ]

'''

import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def draw_points(image, player=False):
    ''' Get N points (x, y) from image '''

    def mouse_handler(event, x_pos, y_pos, _, data):
        ''' Draw points over image '''

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data['im'], (x_pos, y_pos), 3, (0, 0, 255), 5, 16)
            cv2.imshow('Image', data['im'])
            if player:
                data['points'].append([x_pos, y_pos, 1])
            else:
                data['points'].append([x_pos, y_pos])
        elif event == cv2.EVENT_RBUTTONDOWN:
            data['valid'] = False

    data = {}
    data['im'] = image.copy()
    data['points'] = []
    data['valid'] = True

    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', mouse_handler, data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not data['valid']:
        return None

    points = np.vstack(data['points']).astype(float)
    return points

def detect_teams_hsv(frame, x0, y0, width, height):

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    crop = frame[int(y0):int(y0+height), int(x0):int(x0+width), :]
    crop_hsv = frame_hsv[int(y0):int(y0+height), int(x0):int(x0+width), :]

    ## RED
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(crop_hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 120, 70])
    upper_red_2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(crop_hsv, lower_red_2, upper_red_2)
    mask_red_res = cv2.bitwise_or(mask_red1, mask_red2)
    red_res = cv2.bitwise_and(crop, crop, mask=mask_red_res)
    red_res = cv2.cvtColor(red_res, cv2.COLOR_HSV2BGR)
    red_res = cv2.cvtColor(red_res, cv2.COLOR_BGR2GRAY)
    red_count = cv2.countNonZero(red_res)

    ## WHITE
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)
    mask_white = cv2.inRange(crop_hsv, lower_white, upper_white)
    white_res = cv2.bitwise_and(crop, crop, mask=mask_white)
    white_res = cv2.cvtColor(white_res, cv2.COLOR_HSV2BGR)
    white_res = cv2.cvtColor(white_res, cv2.COLOR_BGR2GRAY)
    white_count = cv2.countNonZero(white_res)

    if red_count >= 250:
        bbox_color = (60, 71, 222)
    else:
        if white_count >= 250:
            bbox_color = (230, 230, 237)
        else:
            bbox_color = (41, 224, 227)

    return bbox_color


def plot_graph(tracks):

    fig, ax = plt.subplots(figsize=(10.4, 6.8))

    x_avgs = []
    y_avgs = []
    for t in tracks:
        print(t['id'], len(t['bbox']))
        if t['id'] == 2:
            for bbox in t['bbox']:
                x0, y0, x1, y1 = bbox[-1]
                x_avg = (x0 + x1) / 2
                y_avg = y1 - 10
                x_avgs.append(x_avg)
                y_avgs.append(y_avg)

    print('x_avgs', len(x_avgs))
    print('y_avgs', len(y_avgs))

    # Smooth
    ## Usually the more window_length you use, the more 'smooth' it is. (window_length is the second parameter)
    x_avgs_smooth_1 = savgol_filter(x_avgs, 7, 3)
    y_avgs_smooth_1 = savgol_filter(y_avgs, 7, 3)

    plt.plot(x_avgs, y_avgs, color='#32a852', label='Normal')
    plt.plot(x_avgs_smooth_1, y_avgs_smooth_1, color='#b942f5', label='Smooth #1')
    plt.legend(loc='upper right')
    plt.show()

def visualize_pitch(tracks):

    input_video_filename = '/home/alex/Downloads/TolucaVSPumas_sample.mp4'

    input_video = cv2.VideoCapture(input_video_filename)
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    ## Smooth data
    data = []
    for t in tracks:
        ids = []
        x_avgs = []
        y_avgs = []
        frame_numbers = []
        scores = []
        class_ids = []
        print(t['id'], len(t['bbox']))
        for bbox in t['bbox']:

            ids.append(t['id'])
            x_avgs.append(bbox[-1][0][0])
            y_avgs.append(bbox[-1][1][0])
            frame_numbers.append(bbox[0])
            scores.append(bbox[1])
            class_ids.append(bbox[2])

        data.append({
            'ids': ids,
            'x_avgs': x_avgs,
            'y_avgs': y_avgs,
            'frame_numbers': frame_numbers,
            'scores': scores,
            'class_ids': class_ids
        })

        for d in data:
            # print(set(d['ids']), len(d['x_avgs']), len(d['y_avgs']))
            assert len(d['x_avgs']) == len(d['y_avgs'])
            if len(d['x_avgs']) > 400:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 199, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 199, 3)
            elif len(d['x_avgs']) > 200:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 99, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 99, 3)
            elif len(d['x_avgs']) > 100:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 49, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 49, 3)
            elif len(d['x_avgs']) > 50:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 23, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 23, 3)
            elif len(d['x_avgs']) > 25:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 11, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 11, 3)
            elif len(d['x_avgs']) > 12:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 5, 3)
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 5, 3)
            else:
                d['x_avgs_smooth'] = d['x_avgs']
                d['y_avgs_smooth'] = d['y_avgs']
            
        ## END Smooth data

    bbox_colors = [(235, 64, 52), (235, 208, 52), (52, 235, 98)]
    i = 0
    while input_video.isOpened():

        if i % 60 == 0:
            prog = round(i / num_frames, 2) * 100
            print('{}%'.format(prog))

        ret, frame = input_video.read()

        if ret:

            pitch = cv2.imread('/home/alex/Downloads/pitch.png')
            for d in data:
                for fid, frame_number, x_avg, y_avg, score, class_id in zip(d['ids'], d['frame_numbers'], d['x_avgs_smooth'], d['y_avgs_smooth'], d['scores'], d['class_ids']):
                    if fid in [30, 31, 32, 33, 47, 50]:
                        continue
                    if frame_number == i:
                        bbox_color = bbox_colors[class_id]
                        # cv2.putText(pitch, str(fid), (int(x_avg + 5), int(y_avg + 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        cv2.circle(pitch, (int(x_avg), int(y_avg)), 8, bbox_color, -1)

            cv2.imwrite('/home/alex/Downloads/pitch/result/{}.jpg'.format(i), pitch)
            i += 1

        else:
            break

    input_video.release()

def visualize_video(tracks, smooth=False):

    input_video_filename = '/home/alex/Downloads/TolucaVSPumas_sample.mp4'

    input_video = cv2.VideoCapture(input_video_filename)
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    if smooth:
        output_video_filename = '/home/alex/Downloads/iou_tracker_smooth.mp4'
    else:
        output_video_filename = '/home/alex/Downloads/iou_tracker.mp4'

    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = input_video.get(cv2.CAP_PROP_FPS)
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = cv2.VideoWriter(filename=output_video_filename, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=float(frames_per_second), frameSize=(width, height), isColor=True)

    if smooth:
        ## Smooth data
        data = []
        for t in tracks:
            bboxs = []
            ids = []
            x_avgs = []
            y_avgs = []
            frame_numbers = []
            for bbox in t['bbox']:
                x0, y0, x1, y1 = bbox[-1].tolist()
                x_avg = (x0 + x1) / 2
                y_avg = y1 - 10

                ids.append(t['id'])
                x_avgs.append(x_avg)
                y_avgs.append(y_avg)
                frame_numbers.append(bbox[0])
                bboxs.append(bbox)

            data.append({
                'ids': ids,
                'x_avgs': x_avgs,
                'y_avgs': y_avgs,
                'frame_numbers': frame_numbers,
                'bboxs': bboxs
            })

        for d in data:
            # print(set(d['ids']), len(d['x_avgs']), len(d['y_avgs']))
            if len(d['x_avgs']) > 200:
                d['x_avgs_smooth'] = savgol_filter(d['x_avgs'], 119, 3)
            else:
                d['x_avgs_smooth'] = d['x_avgs']
            
            if len(d['y_avgs']) > 200:
                d['y_avgs_smooth'] = savgol_filter(d['y_avgs'], 119, 3)
            else:
                d['y_avgs_smooth'] = d['y_avgs']
        ## END Smooth data

    bbox_colors = [(235, 64, 52), (235, 208, 52), (52, 235, 98)]
    i = 0
    while input_video.isOpened():

        if i % 60 == 0:
            prog = round(i / num_frames, 2) * 100
            print('{}%'.format(prog))

        ret, frame = input_video.read()

        if ret:
  
            if smooth:
                for d in data:
                    for fid, frame_number, x_avg, y_avg, bbox in zip(d['ids'], d['frame_numbers'], d['x_avgs_smooth'], d['y_avgs_smooth'], d['bboxs']):
                        if frame_number == i:

                            x0, y0, x1, y1 = bbox[-1].tolist()
                            width = x1 - x0
                            height = y1 - y0

                            bbox_color = bbox_colors[bbox[2]]

                            # cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), bbox_color, 5)
                            # cv2.putText(frame, str(fid), (int(x_avg), int(y_avg)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.circle(frame, (int(x_avg), int(y_avg)), 10, bbox_color, -1)

            else:
                for t in tracks:
                    if t['id'] in [30, 31, 32, 33, 47, 50]:
                        continue
                    for bbox in t['bbox']:
                        ## bbox[0] represents the frame number
                        if bbox[0] == i:
                            x0, y0, x1, y1 = bbox[-1].tolist()
                            width = x1 - x0
                            height = y1 - y0
                            x_avg = (x0 + x1) / 2
                            y_avg = y1 - 10

                            bbox_color = bbox_colors[bbox[2]]

                            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), bbox_color, 5)
                            cv2.putText(frame, str(t['id']), (int(x_avg), int(y_avg)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            # cv2.circle(frame, (int(x_avg), int(y_avg)), 8, (243, 27, 88), -1)
                            # cv2.putText(frame, '({},{},{},{})'.format(int(x0), int(y0), int(x1), int(y1)), (int(x_avg + 20), int(y_avg + 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, str(i), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
            output_video.write(frame)
            i += 1

        else:
            break

    input_video.release()
    output_video.release()

def get_iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def iou_tracker():
    '''
    IOU Tracker implementation
    NOT OPTIMIZED!
    '''

    def get_last_id(trackers):
        if trackers:
            return sorted([t['id'] for t in trackers], reverse=True)[0]
        return 1

    with open('/home/alex/Downloads/team_detections.pickle', 'rb') as fp:
        frames_detections = pickle.load(fp)

    trackers = []

    for frame_detections in frames_detections:

        frame_number = frame_detections[0]
        detections = frame_detections[1]
        scores = frame_detections[2]
        classes = frame_detections[3]

        if detections.size == 0:
            continue

        for detection, score, class_id in zip(detections, scores, classes):

            if class_id == 1:
                continue

            if not trackers:
                trackers.append({
                    'id': get_last_id(trackers),
                    'bbox': [[frame_number, score, class_id, detection]],
                })
            else:
                ious = []
                for tracker in trackers:
                    prev_frame_number = frame_number - 60
                    if tracker['bbox'][-1][0] > prev_frame_number:
                    # if tracker['bbox'][-1][0] == prev_frame_number:
                        if tracker['bbox'][-1][2] == class_id:
                            iou = get_iou(detection, tracker['bbox'][-1][-1])
                            if iou > 0:
                                ious.append({
                                    'id': tracker['id'],
                                    'iou': iou
                                })
                if ious:
                    max_iou = max(ious, key=lambda x: x['iou'])
                    for tracker in trackers:
                        if tracker['id'] == max_iou['id']:
                            ### Ugly fix to avoid having multiple same ID's on a single frame
                            if tracker['bbox'][-1][0] != frame_number:
                                tracker['bbox'].append([frame_number, score, class_id, detection])
                else:
                    trackers.append({
                        'id': get_last_id(trackers) + 1,
                        'bbox': [[frame_number, score, class_id, detection]]
                    })

    return trackers

if __name__ == "__main__":
  
    iou_tracks = iou_tracker()
    with open('/home/alex/Downloads/tracks.pickle', 'wb') as fp:
        pickle.dump(iou_tracks, fp)

    visualize_video(iou_tracks, False)

    with open('/home/alex/Downloads/map_tracks.pickle', 'rb') as fp:
        map_tracks = pickle.load(fp)
    visualize_pitch(map_tracks)
    
    # plot_graph(tracks)
