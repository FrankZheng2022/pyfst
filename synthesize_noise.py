def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    nfft = 200 # Length of each window segment 0.2s
    fs = 8000 # Sampling frequencies
    noverlap = 100 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
    
def get_random_time_segment(segment_ms): 
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)

def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time   
    overlap = False

    for previous_start, previous_end in previous_segments:
        if previous_start<segment_end and segment_start<=previous_end:
            overlap = True
    return overlap

def insert_audio_clip(background, audio_clip, previous_segments):
    
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    i = 0
    while is_overlapping(segment_time,previous_segments):
        segment_time = get_random_time_segment(segment_ms)
        if (i>=50):
            raise Exception()
        i+=1

    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position = segment_time[0])
    
    return new_background, segment_time

def insert_label(y, segment_end_ms):
    Ty = y.shape[1]
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    for i in range(segment_end_y, segment_end_y+Number_pos):
        if i < Ty:
            y[0, i] = 1
    return y

#Standardize volume of audo clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

def create_training_example(background, activates, negatives, i, train):
    y = np.zeros((1,Ty))
    background = background-10
    previous_segments = []
    number_of_activates = np.random.randint(0, 4)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]
    for random_activate in random_activates:
        try:
            # Insert the audio clip on the background
            background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
            # Retrieve segment_end from segment_time
            _, segment_end = segment_time
            # Insert labels in "y"
            y = insert_label(y, segment_end)
        except:
            raise Exception()
    # Select 0-2 random negatives audio recordings and insert it into the background
    number_of_negatives = np.random.randint(0, 4)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]
    for random_negative in random_negatives:
        try:
            background, _ = insert_audio_clip(background, random_negative, previous_segments)
        except:
            raise Exception()
    #Standardize the volume of the audio clip 
    background = match_target_amplitude(background, -20.0)
    x = 0
    if (train):
        background.export("./train/train" + str(i) + ".wav", format="wav")
        if (i%10==0):
            print ("Training example {} has been created".format(i))
        x = graph_spectrogram("./train/train" + str(i) + ".wav")
        x = x.reshape(x.shape[1],-1)
    else:
        background.export("./dev/dev" + str(i) + ".wav", format="wav")
        if (i%10==0):
            print ("Testing example {} has been created".format(i))
        x = graph_spectrogram("./dev/dev" + str(i) + ".wav")
        x = x.reshape(x.shape[1],-1)
    return (x, y)

def create_training_set(backgrounds, activates, negatives, batch_size):
        random_indices = np.random.randint(0,len(backgrounds))
        background = backgrounds[random_indices]
        try:
            create_training_example(background, activates, negatives, i, True)
        except:
            i-=1
    for i in range(int(batch_size*0.2)):
        random_indices = np.random.randint(0,len(backgrounds))
        background = backgrounds[random_indices]
        try:
            create_training_example(background, activates, negatives, i, False)
        except:
            i-=1
