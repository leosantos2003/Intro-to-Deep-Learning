# Intro to Deep Learning
  
## About

This repository documents the learnings from completing the [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning) course on [Kaggle](https://www.kaggle.com/).

The focus here is on the practical implementation of neural networks to solve a classification problem, using the TensorFlow framework and its high-level API, Keras. The goal is to learn how to build, train, and optimize dense (fully-connected) neural network architectures.

* `lesson_1`: a single neuron; linear units. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_1)

* `lesson_2`: deep neural networks; adding hidden layers to the network. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_2)

* `lesson_3`: stochastic gradient descent; using Keras and Tensorflow to train the first neural network. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_3)
 
* `lesson_4`: overfitting and underfitting; improving performance with extra capacity or early stopping. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/blob/main/lesson_4)
 
* `lesson_5`: dropout and batch normalization; adding special layers to prevent overfitting and stabilize training. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_5)
 
* `lesson_6`: binary classification; applying deep learning to another common task. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_6)

## Lesson 1

`lesson 1`: a single neuron; linear units.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* The graphic shows an untrained model which set the weights as random, small values, and biases with zero; it's a standard practice for any neuron.
* Now we see that a simple neuron is a linear function.
* The weights will be different each time the script is run.

</div>
      <img width="480" height="360" alt="linear_neuron_graphic" src="https://github.com/user-attachments/assets/d1ee2c6d-fbbc-484b-922d-2c653deb6c41" />
</div>

## Lesson 2

`lesson_2`: deep neural networks; adding hidden layers to the network.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">

* The Activation Function acts as a "decision filter" at the end of each neuron.
* It introduces nonlinearity. If we didn't use activation functions (or used a linear function), no matter how many layers the network had, it would behave like a simple linear regression.

### Graphic 1:
* The graphic shows the "ReLU" Activation Function. "ReLU" stands for "Rectified Linear Unit".
  * If the input (x) is positive, the output is the input itself.
  * If the input (x) is negative or zero, the output is zero.
* ReLU is extremely popular because it is simple, fast to compute, and solves important problems in training deep networks, all with an incredibly basic rule.

</div>
      <img width="480" height="360" alt="activation_graphic" src="https://github.com/user-attachments/assets/5fce2889-6b84-434d-9c1d-e9bad4583be8" />
</div>

## Lesson 3

`lesson_3`: stochastic gradient descent; using Keras and Tensorflow to train the first neural network.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">

### Graphic 1:
* Compares the Loss with the increase of Epochs.
* The loss is decreasing alongside the increase of Epochs because the model is learning. Each epoch is an opportunity for it to adjust its internal parameters (weights and biases) to make increasingly better predictions and, consequently, make fewer errors.
* It stabilizes and becomes a nearly horizontal line at the end of training. At this point, the model has already learned the main pattern of the data and can no longer significantly improve.

</div>
      <img width="500" height="300" alt="training_loss_graphic" src="https://github.com/user-attachments/assets/468f3a54-d6ff-4f6e-9594-adef06a8d923" />
</div>

## Lesson 4

`lesson_4`: overfitting and underfitting; improving performance with extra capacity or early stopping.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* Simple linear model with low capacity.
* The model learned what little it could very quickly and was unable to improve further.
* Both training and validation losses are high because the model performs poorly in both scenarios.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_1" src="https://github.com/user-attachments/assets/88140202-c22a-4adf-8dee-23cc5fd575d2" />
<div class="texto-titulo">
      
### Graphic 2:
* The model's capacity was dramatically increased by adding two hidden layers with hundreds of neurons and the relu activation function.
* A classic overfitting example:
  * The training curve (`loss`, blue) continues to fall consistently throughout the 50 epochs, reaching a very low value.
  * The validation curve (`val_loss`, orange) initially drops, but then stops improving and starts rising.
* The model is "too good" at training, to the point that it became bad when predicting new data.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_2" src="https://github.com/user-attachments/assets/3a4011bf-9ec9-43fc-9831-4a00062c9733" />
<div class="texto-titulo">
      
### Graphic 3:
* `EarlyStopping`: training is stopped before completing 50 epochs.
* It monitors val_loss and when it notices that the model is no longer improving on the validation data for a certain number of epochs (`patience=5`), it stops training.
* This prevents overfitting by simply stopping the process before the model starts to "memorize" too much. It's one of the most effective and straightforward ways to find a good balance, resulting in a model that generalizes well to new data.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_3" src="https://github.com/user-attachments/assets/b0faf6d1-8b64-4f4c-84f8-aa3e1fb91c0b" />
</div>

## Lesson 5

`lesson_5`: dropout and batch normalization; adding special layers to prevent overfitting and stabilize training.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* `Dropout` helps prevent overfitting by randomly "turning off" some neurons during training.
* This forces the network to learn in more robust ways. The result is a model that generalizes well, as seen in the loss curves that converge steadily.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_1" src="https://github.com/user-attachments/assets/44a79052-8e48-4b1b-8f44-2a94006abae2" />
      
### Graphic 2:
* The features in the concrete dataset have very different scales (e.g., "Cement" is in the hundreds, while "Age" is in the tens).
* For an optimizer like `sgd`, the gradients calculated for large-scale features (like "Cement") will completely dominate the learning process, making weight adjustments unstable and ineffective.
* Learning is chaotic and inefficient. Loss curves are likely very high, unstable, or barely decreasing at all. This is why we don't see anything.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_2" src="https://github.com/user-attachments/assets/2abcedf1-52dd-4a31-82e7-222664eb052d" />
      
### Graphic 3:
* `BatchNormalization` solves the problem:
  * Before data enters a `Dense` layer, the `BatchNormalization` layer rescales it so that it has a mean close to 0 and a standard deviation of 1.
  * This ensures that no single feature dominates the learning. Given that the learning process is more stable, the model can learn much faster and more reliably, even with sensitive optimizers like `sgd`.

</div>
      <img width="500" height="300" alt="loss_vs_val_loss_3" src="https://github.com/user-attachments/assets/bc430c02-84cb-4410-aee4-87ff45c43050" />

</div>

## Lesson 6

`lesson_6`: binary classification; applying deep learning to another common task.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
### Graphic 1:
* `Binary Cross-entropy`: measures not only whether the model was right or wrong, but also the confidence of the prediction.
  * Low Penalty: The model predicts a 90% chance of "cancelled" (0.9), and the reservation is indeed canceled. The prediction was correct and confident.
  * Medium Penalty: The model predicts a 60% chance of "cancelled" (0.6), and the reservation is canceled. The prediction is correct, but less confident.
  * Extremely High Penalty: The model predicts a 1% chance of "cancelled" (0.01), and the reservation is canceled. The prediction was spectacularly wrong and very confident in its error.
* The goal of training is to minimize this penalty.

</div>
      <img width="500" height="300" alt="cross_entropy" src="https://github.com/user-attachments/assets/c2dbbed9-d205-4028-82d2-1043b9411f70" />
      
### Graphic 2:
* `Binary Accuracy`: represents the correcteness percentage of all the model's predictions.
  * If the accuracy is 0.85, it means the model was correct 85% of the time.
  * If the model predicts a 51% chance of cancellation (0.51), that counts as a hit (assuming a 50% threshold). If it predicts 99% (0.99), it also counts as a hit. Accuracy doesn't count the confidence.
* The goal is to maximize accuracy.

</div>
      <img width="500" height="300" alt="accuracy" src="https://github.com/user-attachments/assets/d4a767f8-c214-4a70-ad9e-bb08cdf1093c" />
</div>

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
