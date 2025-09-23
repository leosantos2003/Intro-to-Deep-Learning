# Intro to Deep Learning
  
## About

This repository documents the learnings from completing the [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning) course on [Kaggle](https://www.kaggle.com/).

The focus here is on the practical implementation of neural networks to solve a classification problem, using the TensorFlow framework and its high-level API, Keras. The goal is to learn how to build, train, and optimize dense (fully-connected) neural network architectures.

teste:
```mermaid
graph TD
subgraph Fase de Indexação (Preparação do Conhecimento)
  A[1. Ingestão de Dados<br>(Ex: Documentos PDF)] --> B(2. Chunking<br>Dividir em Pedaços);
  B --> C(3. Embedding<br>Converter em Vetores);
  C --> D[4. Indexação<br>(Armazenar em um Banco de Dados Vetorial)];
end

subgraph Fase de Recuperação e Geração (Em tempo real)
  E[Pergunta do Usuário] --> F(5. Embedding da Pergunta);
  F --> G{6. Recuperação<br>(Busca por Similaridade)};
  D --> G;
  G --> H(Contexto Relevante<br>+ Pergunta Original);
  H --> I[7. Geração<br>(LLM formula a resposta)];
  I --> J[Resposta Final];
end

style D fill:#D6EAF8,stroke:#333,stroke-width:2px
style I fill:#D5F5E3,stroke:#333,stroke-width:2px
```

* `lesson_1`: a single neuron; linear units. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_1)

* `lesson_2`: deep neural networks; adding hidden layers to the network. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_2)

* `lesson_3`: stochastic gradient descent; using Keras and Tensorflow to train the first neural network. [see graphic](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_3)
 
* `lesson_4`: overfitting and underfitting; improving performance with extra capacity or early stopping. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/blob/main/lesson_4)
 
* `lesson_5`: dropout and batch normalization; adding special layers to prevent overfitting and stabilize training. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_5)
 
* `lesson_6`: binary classification; applying deep learning to another common task. [see graphics](https://github.com/leosantos2003/Intro-to-Deep-Learning/tree/main/lesson_6)

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Leonardo Santos - <leorsantos2003@gmail.com>
