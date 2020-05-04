## Baseline:

The baseline model can be run through the

```python3
computeBaseline()
```

function call. Upon completion it yields a training accuracy and loss of 100%
(since epoch 60) and around 2.4*10^(-5) respectively. In comparrison, however, its test
accuracy and loss are 70.3% (since epoch 71) and 3.471 respectively. This obviously
shows a massive problem with overfitting. In future models an attempt to reduce this
will be made. In comparing these values to the 2010 baseline, we see a test accuracy
difference of 7.4%.

Note: In the future when I talk about a models accuracy I am referring to its test accuracy. 

## Models 2, 3, 4:

These models were focused around different dropout ratios. The resulting accuracies
of each model was 76.9%, 81.9%, and 81% respectively. Hence I chose to move forward
into the next test with a lower starting dropout. 

## Model 5:

Since the middle ground of the previous model set was the most accurate, I decided
to have the increasing dropout values around that value. Model 5 was about
changing the constand dropouts to a variable one. This was done by making the first,
second, and third dropouts 0.2, 0.35, and 0.5 respectively. This model yielded an
accuracy of 83.8%, the best yet.

## Models 6 and 7:

These models were about adding another dropout between the dense layers of the model.
The value of the dropouts were 0.3 and 0.5 respectively. From these the accuracies
84.6% and 82.1% were obtained. Hence the better value was clearly in model 6. 

## Conclusion:

Model 6 has the best accuracy on the data provided to it. Additionally, it has a
5.7% more accurate prediction on the testing data set. 

### Notes: 

Any model can be loaded through the function

```python3
loadModel(folder)
```

where the folder is the name of the model you wish to load. E.g. "baseline" or 
"model4". It will return the model, its loss, and its accuracy on the testing
dataset. 