# Lesson 5

`lesson_5`: dropout and batch normalization; adding special layers to prevent overfitting and stabilize training.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
## Graphic 1:
* `Dropout` helps prevent overfitting by randomly "turning off" some neurons during training.
* This forces the network to learn in more robust ways. The result is a model that generalizes well, as seen in the loss curves that converge steadily.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_1" src="https://github.com/user-attachments/assets/44a79052-8e48-4b1b-8f44-2a94006abae2" />
      
## Graphic 2:
* The features in the concrete dataset have very different scales (e.g., "Cement" is in the hundreds, while "Age" is in the tens).
* For an optimizer like `sgd`, the gradients calculated for large-scale features (like "Cement") will completely dominate the learning process, making weight adjustments unstable and ineffective.
* Learning is chaotic and inefficient. Loss curves are likely very high, unstable, or barely decreasing at all. This is why we don't see anything.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_2" src="https://github.com/user-attachments/assets/2abcedf1-52dd-4a31-82e7-222664eb052d" />
      
## Graphic 3:
* `BatchNormalization` solves the problem:
  * Before data enters a `Dense` layer, the `BatchNormalization` layer rescales it so that it has a mean close to 0 and a standard deviation of 1.
  * This ensures that no single feature dominates the learning. Given that the learning process is more stable, the model can learn much faster and more reliably, even with sensitive optimizers like `sgd`.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_3" src="https://github.com/user-attachments/assets/bc430c02-84cb-4410-aee4-87ff45c43050" />

</div>
