# Lesson 4

`lesson_4`: overfitting and underfitting; improving performance with extra capacity or early stopping.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
## Graphic 1:
* Simple linear model with low capacity.
* The model learned what little it could very quickly and was unable to improve further.
* Both training and validation losses are high because the model performs poorly in both scenarios.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_1" src="https://github.com/user-attachments/assets/88140202-c22a-4adf-8dee-23cc5fd575d2" />
<div class="texto-titulo">
      
## Graphic 2:
* The model's capacity was dramatically increased by adding two hidden layers with hundreds of neurons and the relu activation function.
* A classic overfitting example:
  * The training curve (`loss`, blue) continues to fall consistently throughout the 50 epochs, reaching a very low value.
  * The validation curve (`val_loss`, orange) initially drops, but then stops improving and starts rising.
* The model is "too good" at training, to the point that it became bad when predicting new data.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_2" src="https://github.com/user-attachments/assets/3a4011bf-9ec9-43fc-9831-4a00062c9733" />
<div class="texto-titulo">
      
## Graphic 3:
* EarlyStopping: training is stopped before completing 50 epochs.
* It monitors val_loss and when it notices that the model is no longer improving on the validation data for a certain number of epochs (patience=5), it stops training.
* This prevents overfitting by simply stopping the process before the model starts to "memorize" too much. It's one of the most effective and straightforward ways to find a good balance, resulting in a model that generalizes well to new data.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="loss_vs_val_loss_3" src="https://github.com/user-attachments/assets/b0faf6d1-8b64-4f4c-84f8-aa3e1fb91c0b" />
</div>
