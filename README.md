# Continuous Cross User 
As we discussed this project really consists of two parts:

1. **Offline Optimization**: Optimizing a neural network to get the best possible offline performance.  
2. **Online Evaluation**: Testing the model online to validate part 1. 

## Offline Optimization 
I've tried a lot of ways to optimize the offline model... I'll share some of my thoughts. 

1. Theoretically, we should get the best performance with a CNN. What I've seen so far confirms this. 
2. We need to figure out how to best pre-process the data. In theory, this should benefit all models (LDA, MLP, and CNN). So this is probably where I would start. 
    1. **Active Thresholding**: On a per subject and across the dataset. 
    2. **Removing Subjects**: Should we remove some subjects? 
    3. **Removing Transient Parts**: Should we remove the transient pieces of the discrete gestures? 
    4. **No Motion Data**: There is currently a large imbalance of NM data. Should we include all of it? 
3. We could look at the impact of temporal models. Once we have good preprocessing lets try adding LSTM, RNN, Transformer layers to our MLP and see if we can extract improved performance. 
    1. Note - we tried this and got improved offline performance but it didn't work online. 
4. Explore the use of manual feature extarction vs CNN-based feature extraction. 

Look in the Models/ folder for how your deep networks should be structured. 

## Online Evaluation
Once we have optimized the model from part 1 we will try the best ones online in a user-in-the-loop evaluation.