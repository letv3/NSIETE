# Neural Networks @ FIIT

This repository contains supporting materials for the subject __Neural Networks__.

It is a very important repository, where you will find the asisgnments prepared for you, for the first 4 weeks of the semester.

##  Week 1
This is a warm-up week and for some of you the __lecture is after your seminar__, so we'll take it easy on you. 
All you have to do is to prepare your computing environment. 
If you have never worked with Python, this is a good chance for you to start - you'll be using it a lot during this semester (or not, if you prefer C++).
We have prepared a few simple tasks that you can find in the directory "week_1" - Jupyter Notebook.
If you know all of that already, well than, congratulations ;)

But still... the attendance of the seminars is mandatory.

Regards, L.

## Week 2
All hands on board!

This week, you are going to implement a simple (naive) multilayer perceptron.
In the notebook, there are already prepared base classes and structures. Define the class Neuron and Dense Layer with all the necessary parameters they require. Implement simple feed forward function. The weakness of Rosenblatt's perceptron was it's linearity. Make the linear Neuron Layers' output into non-linear by adding non-linear activation functions (the ones that are declared).

Of course - you don't have to do it before the seminar... 
As stated during the lecture and seminars, this will be your work for the upcoming week's seminars.

Good luck,
see ya

Regards, L.

## Week 3
I cry, you cry, we cry 👍

After successful feed forward, the only logical step is to implement backward feed.
To make the task easier, we have implemented feed forward for you in a matrix form to ease the work with chain rule derivations and matrix multiplications. The backward pass requires the computation of loss function, implement the binary_cross_entropy and mean_squared_error functions. In the model, you can use either of them or both of them in summation. The derivation of these functions is also necessary, so remember the tears of years past of your studies and derive the activations and loss functions.

GLHF

Regards, L,M,S,I
(this task was really a collective effort)

PS: hopefully, this one is bugs free ;)

## Week 4
### Seminars
#### Let's do some training magic! 🧙

And here we have it. On this week's task, you will finally make your network able to train and to optimize for the given problem.
Stefan has created for you a very nice and small framework you will finish with your implementation of optimizers.

Wish you luck and fun with implementing and training your own little networks 😉

Regards, L.

KUDOS Stefan!

### Lecture
#### Training is not all it takes!

We all started at debugging our systems with lines of text into textual output... 
Sure, it can help, but when you are training neural network for hours with new output every few seconds, the data can scale up to few thousands, and you ... we all are lazy to go manually through so many lines.
Thankfully, we don't have to!! because... Visualization comes to help 📈📉📊.

Maybe, few of you have the pleasure to know [**Tensorboard**](https://www.tensorflow.org/tensorboard), which is a nice graphical interface for visualizing any data aggregated during training, most usually graphs with losses.
Graphical visualizations help us greatly with identifying when(+where) the training went wrong.

Another visualization toolkit is [**WandB**](https://wandb.ai/), that Stefan will introduce you on tomorrow's lecture. 
There is a folder named [__wandb__](https://github.com/vgg-fiit/neural_networks_at_fiit/tree/main/wandb) containing a jupyter notebook, that you might see helpful for the lecture and your future work with WandB. 
Take a careful look into it. 

Also - it is mandatory to track your trainings using **WandB**, because you will use it to present your progress on your semestral Assignments (😈).

Great Thanks to Stefan and Igor for their lectures

Regards
L,I,S,M