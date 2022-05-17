import bk.load
import bk.compute


import neuroseries as nts
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np

def CountFramesSlow(path):
    # Count frames in a video by reading all frames one by one
    
    cap = cv2.VideoCapture(path)
    
    count = 0
    while True:
        (grabbed, frames) = cap.read()
        if not grabbed:
            break
        count += 1
    return count

def count_frames_fast(path):
    # Count frames in a video by reading its metadata, might be innacurate if metadata are poorly written

    cap = cv2.VideoCapture(path)
    return(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

def count_frames(path):
    #wrap to count_frames_fast
    return count_frames_fast(path)

def led_intensity(path,x,y,channel = 0):
    intensity = []
    cap = cv2.VideoCapture(path)
    for i in tqdm(range(count_frames_fast(path))):
        grabbed,frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        intensity.append(np.average(frame[y-1:y+1,x-1:x+1,0]))
    cap.release()
    return np.array(intensity)


def sanity_check(path,x,y):
    video = bk.load.video_path(path)
    digitalin = bk.load.digitalin_path(path)


    ttl = bk.load.digitalin(digitalin,0,as_Tsd = True)
    led = bk.load.digitalin(digitalin,15,as_Tsd = True)
    times = bk.compute.TTL_to_times(ttl,start = True)

    
    led_intensity = bk.video.led_intensity(video,x,y)
    led_intensity = nts.Tsd(times[:len(led_intensity)],led_intensity,time_units = 's')
    

    n_ttl = bk.compute.n_TTL(ttl)
    n_frames = count_frames(video)
    
    print(f'Videos Frames : {n_frames}')
    print(f'Number of TTL : {n_ttl}')

    plt.figure()
    plt.plot(led_intensity.as_units('s'))
    plt.plot(led.as_units('s')*100)
    plt.show()