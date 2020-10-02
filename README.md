# Gender_recognition_by_voice_MATLAB
In this project, we will train a new 4 layer Neural network model in MATLAB to recognize the gender with an audio sample. The first layer has 20 neurons and it is the input layer(our dataset has 20 features for each training example). The second and third hidden layers have 10 and 5 neurons respectively. The final layer is a single neuron activated by a sigmoid function, while the inner layers are activated by a ReLu function. We were able to obtain an accuracy of about 89% in the training set.
The following acoustic properties of each voice are measured and included in the dataset :
meanfreq: mean frequency (in kHz)
sd: standard deviation of frequency
median: median frequency (in kHz)
Q25: first quantile (in kHz)
Q75: third quantile (in kHz)
QR: interquantile range (in kHz)
skew: skewness 
kurt: kurtosis 
sp.ent: spectral entropy
sfm: spectral flatness
mode: mode frequency
centroid: frequency centroid
peakf: peak frequency 
meanfun: average of fundamental frequency 
minfun: minimum fundamental frequency 
maxfun: maximum fundamental frequency 
meandom: average of dominant frequency
mindom: minimum of dominant frequency 
maxdom: maximum of dominant frequency 
dfrange: range of dominant frequency 
modindx: modulation index
label: male or female
Dataset obtained from www.kaggle.com/primaryobjects/voicegender

