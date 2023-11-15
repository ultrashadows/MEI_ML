# Part 2 Worksheet 10

## Given only two variables, BMI and age, we changed the hyperparameters various times to attempt to obtain a better result.

### With the default values given by the teacher, we got:
> Hidden Layers: 3 (200, 200, 150)
> Activation: Logistic
> Solver: Adam
> Iterations: 1000
> Training Score: 69.8697% | Test Score: 66.8831%

### Trying to add more complexity to the neural network worsened the performance
> Hidden Layers: 4 (300, 300, 200, 150)
> Activation: Logistic
> Solver: Adam
> Iterations: 1000
> Training Score: 68.4039% | Test Score: 66.2338%

### Heavily reducing the neurons in the layers to better adapt to the number of variables increased the performance
> Hidden Layers: 3 (9, 9, 8)
> Activation: Tanh
> Solver: Sgd
> Iterations: 1000
> Training Score: 70.3583% | Test Score: 68.1818%