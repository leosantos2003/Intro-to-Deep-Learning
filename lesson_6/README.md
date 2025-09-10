# Lesson 6

`lesson_6`: binary classification; applying deep learning to another common task.

<div style="display: flex; justify-content: center;">
<div class="texto-titulo">
      
## Graphic 1:
* `Binary Cross-entropy`: measures not only whether the model was right or wrong, but also the confidence of the prediction.
  * Low Penalty: The model predicts a 90% chance of "cancelled" (0.9), and the reservation is indeed canceled. The prediction was correct and confident.
  * Medium Penalty: The model predicts a 60% chance of "cancelled" (0.6), and the reservation is canceled. The prediction is correct, but less confident.
  * Extremely High Penalty: The model predicts a 1% chance of "cancelled" (0.01), and the reservation is canceled. The prediction was spectacularly wrong and very confident in its error.
* The goal of training is to minimize this penalty.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="cross_entropy" src="https://github.com/user-attachments/assets/c2dbbed9-d205-4028-82d2-1043b9411f70" />
      
## Graphic 2:
* `Binary Accuracy`: represents the correcteness percentage of all the model's predictions.
  * If the accuracy is 0.85, it means the model was correct 85% of the time.
  * If the model predicts a 51% chance of cancellation (0.51), that counts as a hit (assuming a 50% threshold). If it predicts 99% (0.99), it also counts as a hit. Accuracy doesn't count the confidence.
* The goal is to maximize accuracy.

</div>
      <img style="width: 48%;" width="1000" height="600" alt="accuracy" src="https://github.com/user-attachments/assets/d4a767f8-c214-4a70-ad9e-bb08cdf1093c" />
</div>
