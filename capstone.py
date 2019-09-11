import time
import cv2
import pianoDisplay
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from glob import glob
from itertools import combinations
import os 
from mido import Message, MidiFile, MidiTrack, second2tick

keras.clear_session()
print(keras.tensorflow_backend._get_available_gpus())
testing_dir = '.'
semseg_model_dir = "./unet_piano_5.hdf5"

video_dim = (640, 480)
prediction_dim = (288, 160)

semseg_model = load_model(semseg_model_dir)

white_key_model_dir = "./white_keys2.hdf5"
black_key_model_dir = "./black_keys2.hdf5"
white_key_model = load_model(white_key_model_dir)
black_key_model = load_model(black_key_model_dir)

#cap = cv2.VideoCapture(os.path.join(testing_dir, "V1.wmv"))
cap = cv2.VideoCapture(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_delta = 1./fps

frame_arr = list()
image_arr = list()

frame_index = 0
keyboardDetected = False
best_homography = None
background = None
should_flip = False

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

PRESSED = 0
UNPRESSED = 1
DEFAULT_TEMPO = 500000
pressed_notes = list()

last_time = 0
current_time = 0

while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret:
		if frame_index > 50:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			img = cv2.resize(frame, (288, 160))
			img = np.reshape(img, img.shape + (1,))
			img = img/255

			if not keyboardDetected:
				max_area = 0
				while max_area == 0:
					start = time.time()
					predictions = semseg_model.predict(np.array([img]), verbose=1)
					print(time.time() - start)

					ret, thresh = cv2.threshold(predictions[0], 0.5, 255, cv2.THRESH_BINARY)
					cv2.imwrite(os.path.join(testing_dir, "frame.png"), frame)
					cv2.imwrite(os.path.join(testing_dir, "thresh.png"), thresh)
			  
					contours,hierarchy = cv2.findContours(thresh.astype(np.uint8), 1, 1)

					#find largest shape in image
					for cont in contours:
						if cv2.contourArea(cont) > max_area:
							cnt = cont
							max_area = cv2.contourArea(cont)
					
					if max_area > 0:
						epsilon = 0.02*cv2.arcLength(cnt,True)
						approx = cv2.approxPolyDP(cnt,epsilon,True)
						comb_approx = combinations(approx, 4)

						max_area = 0
						best_comb = None
						for comb in comb_approx:
							hull = cv2.convexHull(np.array(comb))
							if len(hull) == 4:
								hull_area = cv2.contourArea(hull)
								if hull_area > max_area:
									max_area = hull_area
									best_comb = comb
						#print(best_comb)
						resize_ratio = np.divide(video_dim, prediction_dim)
						#print(best_comb)

						best_comb = np.multiply(resize_ratio, best_comb)
						homography_points = [np.array([[624, 0],[0, 0],[0, 80],[624, 80]]), np.array([[0, 0],[0, 80],[624, 80],[624, 0]]),
									 np.array([[0, 80],[624, 80],[624, 0],[0, 0]]), np.array([[624, 80],[624, 0],[0, 0],[0, 80]])]

						min_average = None
						for normalized_points in homography_points:
							homography, mask = cv2.findHomography(np.array(best_comb), normalized_points)
				  
							keyboard = cv2.warpPerspective(frame, homography, (640, 360))
							keyboard = keyboard[0:80, 0:624]
							#analyze homographies for concentration of black pixels
							average_pixels = np.average(keyboard[0:40, 0:624])
							if min_average is None or average_pixels < min_average:
								min_average = average_pixels
								best_homography = homography
								background = keyboard
				
						flip = cv2.flip(background, 1) #check horizontal orientation
						if (np.average(background[0:80,0:20]) > np.average(flip[0:80,0:20])):
							background = flip
							should_flip = True
				  
				keyboardDetected = True
				cv2.imwrite(os.path.join(testing_dir, "back.png"), background)
			else:
			#PER KEY CLASSIFICATION
				start = time.time()
				new_pressed = list()
				new_unpressed = list()
				keyboard = cv2.warpPerspective(frame, best_homography, (640, 360))
				keyboard = keyboard[0:80, 0:624]
				if should_flip:
					keyboard = cv2.flip(keyboard, 1)
				diff = cv2.absdiff(background, keyboard)

			  #    if idx == (background_index + 1):
			   #     cv2.imwrite(os.path.join(testing_dir, "keyboard1.png"), keyboard)
				#    cv2.imwrite(os.path.join(testing_dir, "diff1.png"), diff)
					
				 # if round(current_time,2) == 4.95:
				  #  cv2.imwrite(os.path.join(testing_dir, "keyboard.png"), keyboard)
				   # cv2.imwrite(os.path.join(testing_dir, "diff.png"), diff)

				white_keys_regions = list()
				black_keys_regions = list()
				for i in range(52): #white keys
					start_horz = 0
					end_horz = 0

					if i == 0:
						start_horz = 0
						end_horz = 21
					elif i == 51:
						start_horz = 602
						end_horz = 623
					else:
						start_horz = i*12 - 5
						end_horz = (i+1)*12 + 4

					key_region = diff[0:80, start_horz:(end_horz + 1)]
					key_region = np.reshape(key_region, key_region.shape + (1,))
					key_region = key_region/255
					white_keys_regions.append(key_region)
					  
				for i in range(-1, 7): #black keys (occur in 8 cycles of 5)
					if i == -1: #only one key here
						start_horz = 2
						end_horz = 23

						key_region = diff[0:60, start_horz:(end_horz + 1)]        
						key_region = np.reshape(key_region, key_region.shape + (1,))
						key_region = key_region/255
						black_keys_regions.append(key_region)
					else:
						for j in range(5):
							start_horz = 0
							end_horz = 0
							if j == 0:
								start_horz = i*85 + 22
								end_horz = i*85 + 43
							elif j == 1:
								start_horz = i*85 + 39
								end_horz = i*85 + 60
							elif j == 2:
								start_horz = i*85 + 58
								end_horz = i*85 + 79
							elif j == 3:
								start_horz = i*85 + 72
								end_horz = i*85 + 93
							elif j == 4:
								start_horz = i*85 + 86
								end_horz = i*85 + 107

							key_region = diff[0:60, start_horz:(end_horz + 1)]
							key_region = np.reshape(key_region, key_region.shape + (1,))
							key_region = key_region/255
							black_keys_regions.append(key_region)          
			   
				white_predictions = white_key_model.predict_classes(np.array(white_keys_regions))
				black_predictions = black_key_model.predict_classes(np.array(black_keys_regions))
				  #if idx == (background_index + 1):
				  #  print(white_keys_regions)
				   # print(black_keys_regions)
					#print(white_predictions)
					#print(black_predictions)
				  #print(current_time, white_predictions)
				white_index = 0
				black_index = 0
				black_pos = [1, 3, 6, 8, 10] #where black keys lie in the 12-note cycle
				changed = False
				for note in range(21, 109): #range of piano key MIDI values
					note_pressed = False
					if (note % 12) in black_pos: #check position in octave
						if black_predictions[black_index][0] == PRESSED:
							note_pressed = True
						black_index += 1
					else:
						if white_predictions[white_index][0] == PRESSED:
							note_pressed = True
						white_index += 1

					time_delta = int(second2tick((current_time - last_time), mid.ticks_per_beat, DEFAULT_TEMPO))
					if changed:
						time_delta = 0 #so multiple notes can be pressed and unpressed at once
					if note_pressed:
						if note not in pressed_notes:
							changed = True
							#print("%d pressed"%note)
							new_pressed.append(note)
							pressed_notes.append(note)
							track.append(Message('note_on', note=note, time=time_delta))
					else:
						if note in pressed_notes:
							changed = True
							#print("%d unpressed"%note)
							new_unpressed.append(note)
							pressed_notes.remove(note)
							track.append(Message('note_off', note=note, time=time_delta))
					if changed:
						last_time = current_time
				current_time += frame_delta
				pianoDisplay.highlightPressed(new_pressed)
				pianoDisplay.restoreUnpressed(new_unpressed)

				#print(time.time() - start)  
		frame_index += 1
		#print(frame_index)
		if frame_index > 1200:
			break
	else:
		break
  


for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
    for msg in track:
        print(msg)
mid.save(os.path.join(testing_dir, 'piano.mid'))
time.sleep(.1)
cap.release()
pianoDisplay.end()
