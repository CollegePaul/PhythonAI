# Basic perceptron with 1 input and 1 weight
# Paul Smith 
# 05/06/2025

# ---------  Variables ----------------------
inputs = [1,2,3,4,5]    # x values
targets = [3,6,9,12,15] # y values
w = 0.1 # weight of the perceptron
lr = 0.1 # Learning rate
epochs = 40 #how many training loops

# ------ Prediction function ----------------
def predict(i):
    return w * i  # weight * input

# ------- Training Epochs -------------------
for epoch in range(epochs):

    # make a black predictions list, loop over inputs
    # create a prediction and add it to the predictions list
    predictions = []
    for i in inputs:
        predictions.append(predict(i))
   
    # Make a costs list, loop over the predictions and costs
    # find the cost (difference of target and prediction)
    # Cost is also known as Error, or Loss
    costs = []
    for p,t in zip(predictions,targets):
        costs.append(t - p)  # target - prediction
   
    # Mean absolute error - the average error
    average_cost = sum(costs)/len(costs)

    # Output
    print(f"Epoch: {epoch}, Weight: {w:.2f}, Cost: {average_cost:.2f}")

    if average_cost <= 0.001:
        break # Stop if the cost is very low

    # Updating the Weight with the learning rate
    w += lr * average_cost

# ------- Testing the perceptron with a new input --------
print("-------  Finished Training -------------")
print("- Testing the new input 2.5")
answer = predict(2.5)
print(f"Answer: {answer:.2f}")
# the answer should be 2.5
# The percepton has not be given a way to calculate the value,
# it arrives at an answer by adjusting its internal weight.
# The weight will equal the slop of the line,  f(x) = wx , the derivative, f'(x) = w,  which is 3
# Notice the exact value of answer is not 3 based on 40 epochs.
