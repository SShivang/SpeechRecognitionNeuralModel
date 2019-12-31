# This code is modeled after the SpeechModelTutorial. It is designed to run outside
# the jupyter environment, as it requires a lot

# This cell imports libraries that you will need
from matplotlib.pyplot import figure, cm
import matplotlib.pyplot as plt
import numpy as np
import logging
import tensorflow as tf
import tensorflow_hub as hub
from DataSequence import DataSequence
import pprint
import os
from stimulus_utils import load_grids_for_stories
from stimulus_utils import load_generic_trfiles
from SemanticModel import SemanticModel
from dsutils import make_word_ds, make_phoneme_ds,make_semantic_model
from util import make_delayed
import tables
from npp import zscore
from ridge import bootstrap_ridge
import statistics
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load semantic model
eng1000 = SemanticModel.load("english1000sm.hf5")

# These are lists of the stories
Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
            'life', 'myfirstdaywiththeyankees', 'naked',
            'odetostepfather', 'souls', 'undertheinfluence']

# Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
Pstories = ['wheretheressmoke']

# This vector represents all the stories
allstories = Rstories + Pstories

# Load TextGrids
grids = load_grids_for_stories(allstories)

# Load TRfiles
trfiles = load_generic_trfiles(allstories)

# Make word and phoneme datasequences
wordseqs = make_word_ds(grids, trfiles) # dictionary of {storyname : word DataSequence}
phonseqs = make_phoneme_ds(grids, trfiles) # dictionary of {storyname : phoneme DataSequence}

# Given a story in the form a string generate ELMO word encodings for the story
# In order to do this the word sequences from wordseqs is read.
def storyToEncoding(story):

    # retrieve the sequence of words for a story
    wordSequence = wordseqs[story]

    # load the tensor flow module
    elmo = hub.Module("https://tfhub.dev/google/elmo/1")

    # supply the words of the story
    tokens_input = [list(wordSequence.data)]

    # this assumes that the entire story can be one context
    tokens_length = [len(list(wordSequence.data))]

    # initialize the tensor flow module using the story and length
    print(tokens_input, tokens_length)
    features = elmo(
    inputs={
        "tokens": tokens_input
        , "sequence_len": tokens_length
    },
    signature="tokens",
    as_dict=True)

    embeddings = []

    # create a session and run the tensor flow module
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        result = sess.run(features)
        embeddings = result["elmo"]
        embeddings = embeddings[0]

    print (embeddings)

    print (embeddings.data.shape)

    # return the embeddings of the story as a vector
    return embeddings


## Downsample the projected stimuli.
# kept the same code as the tutorial
def downSampleProjectStimuli(semanticseqs):
    # Downsample stimuli
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter

    downsampled_semanticseqs = dict() # dictionary to hold downsampled stimuli
    for story in allstories:
        downsampled_semanticseqs[story] = semanticseqs[story].chunksums(interptype, window=window)

    #Combine stimuli
    trim = 5
    Rstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories])
    Pstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Pstories])

    storylens = [len(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories]

    print ("Rstim shape: ", Rstim.shape)
    print ("Pstim shape: ", Pstim.shape)

    # Delay stimuli
    ndelays = 4
    delays = range(1, ndelays+1)

    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)

    return (delRstim, delPstim)

# load the FMRI Responses same as the tutorial
def loadFRMIResponses():
    # Load responses
    resptf = tables.open_file("fmri-responses.hf5")
    zRresp = resptf.root.zRresp.read()
    zPresp = resptf.root.zPresp.read()
    mask = resptf.root.mask.read()

    return (zRresp, zPresp)

# Run the regression model using the downsampled time series
def runRegression(delRstim,delPstim,zRresp,zPresp):

    # Run regression
    alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 1 # Number of cross-validation runs.
    chunklen = 40 #
    nchunks = 20

    wt, corr, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                     alphas, nboots, chunklen, nchunks,
                                                     singcutoff=1e-10, single_alpha=True)

    f = figure()
    ax = f.add_subplot(1,1,1)
    ax.semilogx( np.logspace(1, 3, 10), bscorrs.mean(2).mean(1), 'o-')

    # wt is the regression weights
    print ("wt has shape: ", wt.shape)

    # corr is the correlation between predicted and actual voxel responses in the Prediction dataset
    print ("corr has shape: ", corr.shape)

    # alphas is the selected alpha value for each voxel, here it should be the same across voxels
    print ("alphas has shape: ", alphas.shape)

    # bscorrs is the correlation between predicted and actual voxel responses for each round of cross-validation
    # within the Regression dataset
    print ("bscorrs has shape (num alphas, num voxels, nboots): ", bscorrs.shape)

    # valinds is the indices of the time points in the Regression dataset that were used for each
    # round of cross-validation
    print ("valinds has shape: ", np.array(valinds).shape)


    # ### Testing the regression models by predicting responses
    # The `bootstrap_ridge` function already computed predictions and correlations for the Prediction dataset, but this is important so let's reproduce that step more explicitly.
    #
    # Remember that according to the linear model, the predicted responses for each voxel are a weighted sum of the semantic features. An easy way to compute that is by taking the dot product between the weights and semantic features: $$\hat{R} = S \beta$$

    # In[37]:


    # Predict responses in the Prediction dataset

    # First let's refresh ourselves on the shapes of these matrices
    print ("zPresp has shape: ", zPresp.shape)
    print ("wt has shape: ", wt.shape)
    print ("delPstim has shape: ", delPstim.shape)


    # In[38]:


    # Then let's predict responses by taking the dot product of the weights and stim
    pred = np.dot(delPstim, wt)

    print ("pred has shape: ", pred.shape)


    # #### Visualizing predicted and actual responses
    # Next let's plot some predicted and actual responses side by side.

    f = figure(figsize=(15,5))
    ax = f.add_subplot(1,1,1)

    selvox = 20710 # a decent voxel

    realresp = ax.plot(zPresp[:,selvox], 'k')[0]
    predresp = ax.plot(pred[:,selvox], 'r')[0]

    ax.set_xlim(0, 291)
    ax.set_xlabel("Time (fMRI time points)")

    ax.legend((realresp, predresp), ("Actual response", "Predicted response"));


    # #### Visualizing predicted and actual responses cont'd
    # You might notice above that the predicted and actual responses look pretty different scale-wise, although the patterns of ups and downs are vaguely similar. But we don't really care about the scale -- for fMRI it's relatively arbitrary anyway, so let's rescale them both to have unit standard deviation and re-plot.

    f = figure(figsize=(15,5))
    ax = f.add_subplot(1,1,1)

    selvox = 20710 # a good voxel

    realresp = ax.plot(zPresp[:,selvox], 'k')[0]
    predresp = ax.plot(zscore(pred[:,selvox]), 'r')[0]

    ax.set_xlim(0, 291)
    ax.set_xlabel("Time (fMRI time points)")

    ax.legend((realresp, predresp), ("Actual response", "Predicted response (scaled)"));


    #  Now you see that the actual and scaled predicted responses look very similar. We can quantify this similarity by computing the correlation between the two (correlation is scale-free, so it effectively automatically does the re-scaling that we did here). This voxel has high correlation.

    # Compute correlation between single predicted and actual response
    # (np.corrcoef returns a correlation matrix; pull out the element [0,1] to get
    # correlation between the two vectors)
    voxcorr = np.corrcoef(zPresp[:,selvox], pred[:,selvox])[0,1]
    print ("Correlation between predicted and actual responses for voxel %d: %f" % (selvox, voxcorr))


    # #### Computing correlations for all voxels
    # Next let's compute this correlation for every voxel in the dataset. There are some very efficient ways to do this, but here I've written a for loop so that it's very explicit what's happening. (This should give exactly the same values as the variable `corr`, which was returned by `bootstrap_ridge`.)

    voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
    for vi in range(zPresp.shape[1]):
        voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]
    print (voxcorrs)

    # ### Visualizing correlations across the brain
    # Let's start with a supposition: the correlation should not be high everywhere, even if this is a good model of how the brain represents the semantic content of speech. There are parts of the brain that just don't respond to speech, so the correlation should be low in those areas. There are other parts of the brain that respond to speech, but maybe don't represent semantic information, so the correlation should be low in those areas as well.
    # But let's begin by plotting a histogram of the correlations across the entire brain. This will show generally whether the model is working well or not.


    # Plot histogram of correlations
    f = figure(figsize=(8,8))
    ax = f.add_subplot(1,1,1)
    ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Num. voxels");
    ax.set_title('Histogram of correlations')
    #plt.show()

    return voxcorrs

# vestige of code from the tutorial
def print_voxel_words(voxnum):
    # find_words_like_vec returns 10 words most correlated with the given vector, and the correlations
    voxwords = eng1000.find_words_like_vec(udwt[:,voxnum])
    print ("Best words for voxel %d (correlation %0.3f):" % (voxnum, voxcorrs[voxnum]))
    pprint.pprint(voxwords)

# This function runs both the ELMO word encodings alongside
# the english1000sm word encodings and compares the correlations between them
def main():
    elmoFeatureSeqs = dict()
    semanticseqs = dict()

    # go through the stories and generate a data sequence of the elmo encoding
    # as well as the semantic model
    for story in allstories:
        ds = wordseqs[story]
        elmoFeatureSeqs[story] = DataSequence(storyToEncoding(story), ds.split_inds, ds.data_times, ds.tr_times)
        semanticseqs[story] = make_semantic_model(wordseqs[story], eng1000)

    # perform the regression using elmo word encodings
    (delRstim, delPstim) = downSampleProjectStimuli(elmoFeatureSeqs)
    (zRresp, zPresp) = loadFRMIResponses()
    elmoCorrelations = runRegression(delRstim, delPstim, zRresp, zPresp)

    # the mean of the mean elmoCorrelations
    x = np.mean(elmoCorrelations)
    print("Mean elmo word embeddings", x)

    # median of the elmo correlations
    y = np.median(elmoCorrelations)
    print("Median elmo word embeddings",y)

    (delRstim, delPstim) = downSampleProjectStimuli(semanticseqs)
    (zRresp, zPresp) = loadFRMIResponses()
    semanticCorrelations = runRegression(delRstim, delPstim, zRresp, zPresp)

    # the mean of the semantic correlations
    x = np.mean(semanticCorrelations)
    print("Mean english1000sm word embeddings", x)

    # the median of the semantic correlations
    y = np.median(semanticCorrelations)
    print("Median english1000sm word embeddings",y)

    f = figure(figsize=(8,8))
    ax = f.add_subplot(1,1,1)
    # ax.hist(elmoCorrelations, 100 , color='r' , label="elmoCorrelations") # histogram correlations with 100 bins
    # ax.hist(semanticCorrelations, 100, color='b', label="semanticCorrelations") # histogram correlations with 100 bins
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Num. voxels")

    sns.distplot(elmoCorrelations, hist=True, kde=True,
             bins=100, color = 'blue',
             hist_kws={'edgecolor':'white'},
             kde_kws={'linewidth': 4},
             label="elmoCorrelations")

    sns.distplot(semanticCorrelations, hist=True, kde=True,
             bins=100, color = 'orange',
             hist_kws={'edgecolor':'white'},
             kde_kws={'linewidth': 4},
             label="semanticCorrelations")

    print(len(elmoCorrelations))
    print(len(semanticCorrelations))
    ax.set_title('Histogram of correlations')

    ax.legend(title='Legend', loc='upper right', labels=['elmoCorrelations', 'semanticCorrelations'])
    plt.show()

if __name__ == '__main__':
    main()
