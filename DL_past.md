# 1. The sequence learning problem in deep learning: 
involves predicting the next element in a sequence based on previous elements. This is particularly useful in applications like natural language processing (NLP), speech recognition, time series prediction, and music generation.

# 2. Unfolding computational graphs: 
is a technique used to represent and visualize the structure of computations, especially in the context of recurrent neural networks (RNNs) and time series data. By unfolding a computational graph, we can transform a recursive or recurrent computation into a sequence of operations that are easier to analyze and optimize. (`Computational graphs are a type of graph that can be used to represent mathematical expressions.` This is similar to descriptive language in the case of deep learning models, providing a functional description of the required computation.)

# 3. Drawbacks of standard neural network
- Standard neural architecture will not perform well for sequence
models
- Feed fwd network accepts a fixed size vector as the input and
produce a fixed size vector as the output
- Does not share the features learned across the different
positions of the text
- Sequence and length has to be maintained in a network for
further processing

# 4. Feed forward vs RNN:
![alt text](image-6.png)

# 5. Recurrent Neural Networks (RNNs): 
are a type of neural network designed specifically for `sequential data`. They are widely used in tasks where the order of the data points matters, such as time series analysis, natural language processing, and speech recognition. It is a type of Neural Network where the output from the previous step is fed as input to the current step. The main and `most important feature of RNN is its Hidden state`, which remembers some information about a sequence. The state is also referred to as Memory State since it remembers the previous input to the network. The fundamental processing unit in a `Recurrent Neural Network (RNN)` is a `Recurrent Unit`.

### Types Of RNN
There are four types of RNNs based on the number of inputs and outputs in the network.
- One to One 
- One to Many 
- Many to One 
- Many to Many 

### Key Components of an RNN Architecture

- **Input Layer:** 
accepts sequential data. Each step in the sequence has a set of features (for example, words in a sentence or data points in a time series).
- **Hidden Layer (Recurrent Layer):**
The hidden layer is the core of the RNN, where the recurrent connections are present.
Each hidden state `ℎ𝑡` ​at time step `𝑡` depends not only on the current input `𝑥𝑡` ​but also on the previous hidden state `ℎ𝑡 − 1`, creating a `feedback loop`.
The recurrent layer uses an activation function, often a `hyperbolic tangent (tanh) or ReLU`.

- **Output Layer:**
The output layer generates predictions, which can vary based on the problem.
In classification, it might have a softmax activation to produce probabilities for different classes. For regression, it might have a linear activation function.
- **Weight Sharing:**
RNNs share the same weights across all time steps, making them efficient for long sequences.

### Key Equations in a Basic RNN For a given time step `𝑡`, Hidden State Update:
### `ℎ𝑡 = 𝑓(𝑊𝑥ℎ * 𝑥𝑡 + 𝑊ℎℎ * ℎ𝑡−1 + 𝑏ℎ)`, **Where:** (**xh, hh, hy, t, h are in subscripts**)
- `ht` is the current hidden state.
- `𝑊𝑥ℎ` and `𝑊ℎℎ` are the weight at recurrent neuron and the weight at input neuron
- `bh` is the bias term
- `f` is the activation function (often tanh or ReLU).

### Output Generation: `𝑦𝑡 = 𝑔(𝑊ℎ𝑦 * ℎ𝑡 + 𝑏𝑦)`, **where:**<br>
- `yt` is the output at time step `t`,
- `Why` is the weight matrix from hidden to output,
- `by` is the output bias,
- `g` is the output activation function (e.g., softmax for classification).

### Forward Propagation in a Nutshell
During forward propagation, the RNN moves through the sequence one step at a time, updating the hidden state and producing an output at each time step. Each hidden state `ℎ𝑡` contains information from all previous inputs in the sequence, enabling the RNN to make context-aware predictions based on the entire input sequence up to that point. This sequential processing is what allows RNNs to handle tasks with temporal dependencies, such as language and time-series prediction.

### Advantages

- An RNN remembers each and every piece of information through time. 
- It is useful in time series prediction only because of the feature to remember previous inputs as well. This is called Long Short Term Memory.
- Recurrent neural networks are even used with convolutional layers to extend the effective pixel neighborhood.

### Disadvantages

- Gradient vanishing and exploding problems.
- Training an RNN is a very difficult task.
- It cannot process very long sequences if using tanh or relu as an activation function.

### Applications of Recurrent Neural Network
- Language Modelling and Generating Text
- Speech Recognition
- Machine Translation
- Image Recognition, Face detection
- Time series Forecasting

# 6. Limitations of vanilla RNN:
Vanilla Recurrent Neural Networks (RNNs) do face significant limitations, primarily due to the phenomena of vanishing and exploding gradients. These issues are particularly problematic when dealing with long-term dependencies in sequential data. Here’s a closer look:

### Vanishing Gradients
- **Definition:** Occurs when the gradients used to update the weights during training become exceedingly small, effectively preventing the network from learning.
- **Consequence:** The network struggles to learn long-range dependencies because the earlier layers receive almost negligible gradient updates.
- **Why It Happens:** In each layer, gradients are multiplied by the weights. If these weights are small, the gradients exponentially decrease as they propagate back through the layers.

### Exploding Gradients
- **Definition:** When the gradients grow excessively large, causing the weights to update too drastically, which can lead to network instability.
- **Consequence:** This can result in numerical overflow and erratic changes in the weights, making the model difficult to train and causing it to diverge.
- **Why It Happens:** If the weights are large, the gradients can grow exponentially as they are propagated back through the network, leading to overflow.

### Impact on Vanilla RNNs

- **Training Difficulty:** Both vanishing and exploding gradients make training vanilla RNNs challenging, especially for tasks that require learning from long sequences of data.
- **Performance:** These issues significantly impact the model's ability to retain information from earlier time steps, resulting in poor performance on tasks that involve long-term dependencies.

### Mitigation Strategies

- **Gradient Clipping:** Limiting the size of gradients to prevent them from getting too large.
- **LSTM and GRU:** Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) are advanced variants of RNNs designed to mitigate these issues by maintaining more stable gradients over long sequences.

# 7. Bidirectional Recurrent Neural Networks (BRNNs) 
is an extension of the standard RNN that processes sequential data in both forward and backward directions. This architecture allows the network to access both past and future information about a particular time step, making it more effective for tasks where context from both directions is useful.

### Key Characteristics of BRNNs

- BRNNs have two RNN layers: 
one processes the input sequence in a forward direction (from start to end), and the other processes it in a backward direction (from end to start).
Each time step’s output is influenced by information from both the previous and future states.

- Concatenated Outputs:
At each time step, the hidden states from both the forward and backward RNNs are concatenated (or otherwise combined), allowing the BRNN to generate a more contextually rich representation of each time step. This combined representation is then used as input to the output layer for predictions.

- For each time step `𝑡`, combine the forward and backward hidden states, resulting in a bidirectional representation 
`ℎ𝑡 = [ℎ𝑡forward ; ℎ𝑡backward]`
Output Generation: Use this combined hidden state to produce the output `𝑦𝑡` at each time step.

# 8. Backpropagation Through Time (BPTT): 
is the process used to train Recurrent Neural Networks (RNNs) by applying backpropagation over each time step in a sequence. Unlike feedforward networks, which have a straightforward backpropagation process, RNNs require a more complex approach because they maintain a hidden state that evolves across time steps. BPTT adapts the standard backpropagation algorithm to account for this sequence-based dependency.

### Key Steps in BPTT
- **Unrolling the RNN:**
In BPTT, the RNN is `"unrolled" across the sequence length`, creating a separate `copy` of the network for `each time step`.
This `unrolling allows the RNN to be visualized as a feedforward network` with one layer for each time step in the sequence.
- **Forward Pass Through Time:**
The forward pass is computed for `each time step` in the sequence, `storing the hidden states and outputs at each step`.
This allows the model to `capture dependencies`, as each hidden state depends on both the current input and the previous hidden state.
- **Backward Pass Through Time:**
After the forward pass, BPTT calculates `gradients` by propagating errors backward `through each time step`, starting from the final time step.
`Gradients are computed for each weight with respect to each hidden state and each output`.
The weight updates are `accumulated` and applied after calculating gradients over all time steps in the sequence.
- **Weight Update:**
Once gradients are calculated for each time step, they are summed and applied to update the shared weights of the network, allowing it to learn dependencies across the entire sequence.

### Challenges in BPTT
- Vanishing and Exploding Gradients
- Computational Cost: BPTT is computationally intensive, as it requires storing and processing multiple copies of the network for each time step.

# 9. Truncated BPTT
To address the challenges of long sequences, Truncated Backpropagation Through Time is often used:
- Instead of backpropagating through the entire sequence, `the sequence is divided into shorter segments`, and BPTT is applied within each segment.
- This reduces computation and mitigates the vanishing/exploding gradient problem while still capturing some dependencies.

# 10. Long Short-Term Memory (LSTM): 
is a type of Recurrent Neural Network (RNN) architecture designed to better capture long-term dependencies in sequential data, addressing the common issues of vanishing and exploding gradients that standard RNNs face. 
it incorporate special gating mechanisms that regulate information flow, making them particularly effective for tasks with long-term dependencies like language modeling, time-series prediction, and more.

### Key Components of an LSTM

An LSTM cell has several components, including three gates `(forget, input, and output)` and a cell state. Together, these elements allow it to selectively retain or discard information, improving the network’s ability to learn dependencies over long sequences.
- **Cell State (`𝐶𝑡`):**
The cell state acts as the `"memory"` of the LSTM, allowing information to flow unchanged across `time-steps` unless modified by the gates. It is crucial for retaining long-term information.
- **Hidden State (`ℎ𝑡`):**
The hidden state is the `short-term output` of the LSTM at each `time-step`. This hidden state is also used as input for predictions and is updated at every time step.
- ### **Gates:**
    - **Forget Gate (`𝑓𝑡`):** Decides which information from the cell state should be kept or discarded. It takes the current input 
    `𝑥𝑡` and previous hidden state `ℎ𝑡−1` ​to produce a value between 0 (forget) and 1 (retain).<br>
        `𝑓𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏𝑓)`
    - **Input Gate (`𝑖𝑡`):** Controls how much of the new input `𝑥𝑡` should be added to the cell state. It also uses the current input and previous hidden state to decide what to update.<br>
        `it = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏i)`
    - **Output Gate (`ot`):** It Decides which part of the `cell state` will be in the output.<br>
        `ot = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏o)`

### Pitfalls:
- First, they are `more complicated` than traditional RNNs and
require more training data in order to learn effectively.
- Second, they are `not well-suited` for `learning tasks, such
as prediction or classification` tasks where the input data is not a
sequence.
- Third, LSTMs can be `slow to train` on large datasets. This is due
to the fact that they must learn the parameters of the LSTM
cells, which can be `computationally intensive`.
- Finally, LSTMs may not be appropriate for all types of data. For
example, they may not work well with `highly nonlinear data or
data with a lot of noise`.

LSTMs effectively handle the vanishing gradient problem by controlling the flow of information with gates, allowing gradients to remain stable over long sequences. They can capture both short-term and long-term dependencies, making them suitable for a variety of sequence-based tasks.

# 11. The Gated Recurrent Unit (GRU): 
is a type of Recurrent Neural Network (RNN) architecture that was introduced to address some of the limitations of traditional RNNs and LSTMs (Long Short-Term Memory networks). GRUs are similar to LSTMs but with a simpler structure, using fewer gates and parameters while maintaining comparable performance for many sequential tasks.

### Key Features of GRU
The GRU architecture includes two main gates: 
`the reset gate` and `the update gate`. These gates control the flow of information and help the network maintain relevant information over time while forgetting irrelevant data.

- Update Gate (`𝑧𝑡`):
The update gate decides how much of the previous hidden state should be retained.
It ranges from 0 to 1, where 0 means completely forget the previous state, and 1 means completely retain the previous state.
The update gate is computed using the sigmoid function:
    `z𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏z)`
- Reset Gate (`𝑟𝑡`):
The reset gate controls how much of the previous hidden state should be ignored when calculating the candidate hidden state.
It is computed similarly to the update gate:
`r𝑡 = 𝜎(𝑊𝑓 ⋅ [ℎ𝑡−1,𝑥𝑡] + 𝑏r)`

- The `candidate hidden state` is a new potential memory influenced by the reset gate.
- The `final hidden state` is a weighted combination of the previous hidden state and the candidate state, with the update gate determining how much of the new information to retain.

### Advantages of GRU

- Simpler Architecture: Fewer parameters than LSTMs, which makes them computationally more efficient and easier to train.
- Good Performance: Despite their simplicity, GRUs achieve performance comparable to LSTMs on many tasks, making them a popular choice for sequence modeling.
- Faster Training: Due to fewer parameters, GRUs are faster to train compared to LSTMs, especially on large datasets.

------------------------------------------------------------------------------------------------------------------------------------------

# `UNIT - 4`

# 1. What is Convolution, stride, padding in CNN?
- **Convolution:** is a `mathematical operation` that involves `sliding a small matrix`, called a filter or kernel, over an input image or feature map. Each position of the filter computes a `weighted sum`, which generates a new matrix called a feature map or activation map. This feature map captures important spatial features from the input, such as edges, textures, and shapes.

- **Stride:** refers to the number of pixels the filter moves after each operation. Stride influences the size of the output feature map.

- **Padding:** is the process of adding extra pixels around the border of an image or input matrix. Padding helps control the output's spatial dimensions and has two main purposes:
    - **Maintain Output Size:** Padding can be used to keep the output the same size as the input.
    - **Reduce Information Loss:** By padding the borders, all parts of the input have the opportunity to contribute to the feature map, including the edges.

# 2. Relation Between Input, Output, and Filter Size
The output size of a convolutional layer depends on the input size, the filter size, stride, and padding. For a given 2D input matrix with dimensions `Hin * Win`, the output dimensions `Hout * Wout` can be calculated with the following formulas:

![](image-10.png), where:

- `Hin` and `Win` Height and width of the input.
- K: Size of the filter (kernel) (assuming a square kernel like 3×3).
- P: Padding size (how many pixels are added around the input).
- S: Stride.

# 3. CNN Architecture :
is a specialized type of neural network that is particularly effective for processing `grid-like data`, such as images. CNNs are commonly used for image classification, object detection, and other visual recognition tasks.

![cnnimage](image-11.png)

### Core Layers of a CNN

- **Convolutional Layer:** 
This layer performs the convolution operation. `It applies several filters` (kernels) to the input image, producing multiple feature maps. Each filter is a small matrix that slides over the input image and captures specific patterns (e.g., edges, textures). The output of the convolutional layer, called the `activation map` or feature map, `highlights areas where specific patterns are detected`.

- **Activation Layer:** 
By adding an activation function to the output of the preceding layer, activation layers add `nonlinearity` to the network. it will apply an element-wise activation function to the output of the convolution layer. Some common activation functions are RELU: max(0, x),  Tanh, Leaky RELU, etc.

- **Pooling layer:** 
reduces the `spatial dimensions` of the feature map (height and width) while `retaining important features`. This decreases the computational load and `helps prevent overfitting`. A `2×2` max pooling operation with a `stride of 2` on a `32×32` feature map reduces it to `16×16`.

- **Fully Connected Layer:**
Flattening is used to convert all the resultant 2-Dimensional arrays from
pooled feature maps into a single long continuous linear vector. The
flattened matrix is fed as input to the fully connected layer to classify the
image.
`(These layers are used after a sequence of convolutional and pooling layers to perform classification based on the features extracted.
Each neuron in a fully connected layer is connected to every neuron in the previous layer.)`

- **Output Layer:**
uses an activation function `(usually softmax for multi-class classification or sigmoid for binary classification)` to convert the final outputs into probabilities.

# 4. What is Weight Sharing?
In CNNs, weight sharing means that the `same set of weights` (the filter or kernel) is applied across different regions of the input image. Instead of having `unique weights for every pixel` (as in a fully connected layer), the `same filter is "shared"` across the entire input. This allows the CNN to detect the `same pattern` or feature (e.g., edges, textures) at `different locations` in the image.
#### Advantages:
- Reduced Number of Parameters:
- Lower Memory and Computation Requirements

# 5. FCNN vs CNN 
- **Architecture**
    - **Fully Connected Neural Network (FCNN):**
        - In an `FCNN`, every neuron in one layer is connected to every neuron in the next layer. These layers are called fully connected or dense layers.
        - Each layer contains `independent weights and biases`, meaning a large number of parameters as each neuron must learn its own set of weights for every connection.

    - **Convolutional Neural Network (CNN):**
        - `CNNs` have specialized layers, primarily convolutional layers and pooling layers, followed by fully connected layers at the end.
        - Instead of connecting each neuron to every pixel, CNNs use a `small filter (kernel) that slides across the input`, allowing the network to focus on local spatial patterns.

- **Parameter Efficiency** 
`CNNs` use `weight sharing` in convolutional layers, significantly reducing the number of parameters compared to `FCNNs`, which are more memory-intensive.
- **Best Use Cases**
`FCNNs` are ideal for `tabular data and tasks without spatial structure`. `CNNs` are well-suited for `image, video, and other tasks with spatial patterns`, as they can recognize features regardless of position.
- **Computational Cost and Memory Usage**
`FCNN` Has `high` computational cost and memory usage due to the large number of parameters, especially for high-dimensional inputs like images, as each pixel has a unique weight.
`CNN` Is much more memory-efficient due to weight sharing and smaller filter sizes, which reduces computational cost while maintaining performance, especially for tasks involving spatial patterns.
- **Translation invarience:** CNNs are translation `invariant`, meaning they `can detect features at different positions` within the input, unlike FCNNs.

# 6. Convolution Types:
- Multichannel Convolution: Used for images with multiple channels (like RGB). `Each filter` matches the image's depth, `applies separately to each channel`, and then `sums the results to produce one feature map`.

- 2D Convolution: Most common for images, sliding a 2D filter over the height and width of the image to capture spatial features. Typically used for image processing where data is in two dimensions.

- 3D Convolution: Extends 2D convolution to include depth, allowing the filter to slide over height, width, and depth. It’s useful for video data (where depth could represent time) or 3D medical scans, capturing patterns across all three dimensions.

# 7. LeNet: 
is one of the `earliest` and most famous Convolutional Neural Network (CNN) architectures, developed by `Yann LeCun` in the late `1980s`. Originally created to recognize handwritten digits, LeNet laid the groundwork for modern deep learning applications in image processing.
It consists of seven layers that alternate between convolutional and pooling layers, followed by fully connected layers for classification:
![lenet-5architecture](image-12.png)
- Input Layer: 28x28 sized image from MNIST are resized (augmented) into 32x32 to give as input to LeNet-5
- C1 - Convolutional Layer: 6 filters, 5x5 each, output size 28x28x6.
- S2 - Pooling Layer: 2x2 average pooling, output size 14x14x6.
- C3 - Convolutional Layer: 16 filters, 5x5 each, output size 10x10x16.
- S4 - Pooling Layer: 2x2 average pooling, output size 5x5x16.
- C5 - Fully Connected Layer: 120 neurons.
- F6 - Fully Connected Layer: 84 neurons.
- Output Layer: 10 neurons for classifying digits (0-9).

**All layers from `C1` to `F6` use `tanH` activation function and output layer uses `softmax` activation function.**

# 8. AlexNet:
It consists of `8 layers` in total, out of which the `first 5 are convolutional` layers and the `last
3 are fully-connected`. 
- The first `two convolutional` layers are connected to `overlapping max-pooling layers` to
extract a `maximum number of features`. 
- The third, fourth, and fifth convolutional layers are directly connected to the `fully-connected layers`.
- All the outputs of the convolutional and fully-connected layers are connected to ReLu
non-linear activation function.
- The `final output layer` is connected to a `softmax activation layer`, which produces a distribution of `1000 class labels`.
- The input dimensions of the network are (256 × 256 × 3), meaning that AlexNet is capable of taking
input an RGB (3 channels) image of size upto (256 × 256) pixels.
- If the input image is `grayscale`, it is `converted to an RGB` image by `replicating the single channel` to
obtain a `3-channel` RGB image. 
- There are more than `60 million parameters` and `6,50,000 neurons` involved in the architecture.

### Key Innovations in AlexNet

- **ReLU Activation:**
AlexNet was one of the first networks to use ReLU (Rectified Linear Unit) as the activation function, speeding up training.
- **Dropout:**
Introduced dropout in fully connected layers to reduce overfitting.
- **Data Augmentation:**
Applied techniques like random cropping and flipping to increase data diversity.
- **Use of GPUs:**
Trained on dual GPUs, allowing the model to handle larger architectures and datasets.

------------------------------------------------------------------------------------------------------------------------------------------

# `UNIT - 3`

# 1. Autoencoders `AE's`: 
are a type of artificial neural network used primarily for unsupervised learning, specifically for `feature learning` and `dimensionality reduction`. They are designed to `learn efficient, compressed representations` of input data and are often used to `remove noise, compress data, or even generate new data samples`. The general architecture of an autoencoder includes an encoder, decoder, and bottleneck layer.
- **Encoder:**
Input layer take raw input data. The hidden layers progressively reduce the dimensionality of the input, capturing important features and patterns. These layer compose the encoder.
- **Bottleneck layer:** `(latent space:  lower-dimensional representation of the input data)` is the final hidden layer, where the `dimensionality is significantly reduced`. This layer represents the compressed encoding of the input data.
- **Decoder:**
The hidden layers progressively increase the dimensionality and aim to reconstruct the original input. The output layer produces the reconstructed output, which ideally should be as close as possible to the input data.

- The loss function used during training is typically a `reconstruction loss`, measuring the difference between the input and the reconstructed output. Common choices include `mean squared error (MSE)` for continuous data or `binary cross-entropy` for binary data.

- During training, the autoencoder learns to `minimize the reconstruction loss`, forcing the network to capture the most important features of the input data in the bottleneck layer.

Considering the applications for `data-compression`, `autoencoders are preferred 
over PCA`. PCA makes one stringent but powerful assumption that is linearity i.e. there 
must be linearity in the data set; which is not the case in real-life datasets. However, an autoencoder can learn non-linear transformations with a non linear activation function and multiple layers.

# 2. Types of AutoEncoders `AE's`:
- **Linear Autoencoder:** is a specific type of autoencoder where both the encoder and decoder are linear transformations, meaning they use only linear functions (without activation functions like ReLU or sigmoid) to map the input to the latent space and back. This simple structure makes it mathematically equivalent to Principal Component Analysis (PCA) in terms of its ability to perform dimensionality reduction.

- **Undercomplete Autoencoder:** has a latent space dimension (bottleneck) that is smaller than the input dimension `(the hidden layer has fewer neurons than the input layer)`. The `goal` is to `force the autoencoder` to learn an `efficient, compressed representation` of the data by capturing its `most significant features` and discarding redundant or unimportant information.

- **Overcomplete Autoencoder:** has a latent space dimension that is equal to or larger than the input dimension. `(the hidden layer has more units than the input layer.)` The Model expands the input data into a higher dimensional space, which allows for a potentially richer and detailed representation of the data. These `AE's` have `more capacity to learn a wide range of features`, sometimes even allowing it to reconstruct the input perfectly by learning an identity function.

# 3. Regularization in autoencoders: 
helps prevent the model from simply memorizing or copying the input data (especially in cases with high capacity, like overcomplete autoencoders). Regularization encourages the model to learn meaningful, generalizable representations of the data. Here are several common regularization techniques used in autoencoders:
- **Sparsity Regularization:** forces only a small number of neurons in the hidden layer to be active for a given input, encouraging the network to capture only the most salient features of the data.

- **Denoising:** Adds noise to input data, forcing the autoencoder to learn robust features that resist noise.

- **Contractive:**  add a penalty term to the loss function that encourages the network to resist changes in the input, making the learned representations less sensitive to small input variations.

- **Dropout:** Randomly deactivates neurons during training, helping prevent overfitting and encouraging redundancy.

- **VAE Regularization:** a regularization term is added to make the latent space more continuous and structured, allowing for more controlled generation of new samples.

- **L2 Regularization:** Penalizes large weights to reduce overfitting and promote generalization.

# 4. Denoising Autoencoders `DAEs`: 
are a type of autoencoder specifically designed to make the `learned representations robust to noise`. They are trained to `reconstruct the original, clean input from a corrupted version`, effectively `forcing` the network to `focus on the most essential features` in the data.

### How Denoising Autoencoders Work:

- **Add Noise to Input:** During training, random noise (such as Gaussian noise or masking noise) is added to the input data. This corrupts the input while the output remains the original, clean data.
- **Reconstruction Objective:** The autoencoder then tries to reconstruct the original, noise-free input from this noisy version, minimizing the difference between the reconstructed output and the clean input.
- **Learning Robust Features:** By reconstructing the clean data from noisy input, the DAE learns features that are resilient to irrelevant variations, capturing important patterns and structures in the data.

### Benefits of Denoising Autoencoders

- **Noise Reduction:** DAEs are effective for removing noise from data, like enhancing low-quality images.
- **Feature Learning:** By ignoring noise, DAEs learn robust, high-quality features for downstream tasks (e.g., classification, clustering).
- **Anomaly Detection:** High reconstruction errors for unusual inputs make DAEs useful for identifying anomalies.

### Applications

- **Image Denoising:** DAEs are often used to clean up noisy images.
- **Speech Enhancement:** Improve speech quality by reducing background noise.
- **Anomaly Detection:** Detecting unusual patterns by comparing reconstruction errors for typical vs. anomalous data.

# 5. Sparse Autoencoders: 
are a type of autoencoder that introduces `sparsity constraints` on the hidden layer, `encouraging` the network to learn more `distinct, compressed representations` by activating `only a few neurons for each input`. This sparsity promotes learning of important features `without redundancy`, making sparse autoencoders especially useful for feature extraction.

### How Sparse Autoencoders Work

These `AE's` Enforce sparsity by adding a `regularization term to the loss function`, which `penalizes` the network `if too many neurons are active simultaneously`. This constraint `encourages` only a few neurons in the hidden layer `to respond strongly to each input`, thus learning distinct, critical features.

The `sparsity constraint` forces the autoencoder to only activate `specific neurons` in `response to distinctive features`, helping capture important aspects of the data.

### Applications

- **Image and Text Feature Extraction:** Identifies the key components of images and text data.
- **Anomaly Detection:** Sparse representations make it easier to detect unusual patterns as they stand out from regular features.
- **Pretraining for Deep Networks:** Used to initialize weights in deep networks, especially when labeled data is scarce.

# 6. Contractive Autoencoders (CAEs): 
- are a type of autoencoder that introduce a "contractive" regularization term to make the learned representations less sensitive to small variations in the input.
- In addition to the reconstruction loss, CAEs add a `regularization term` to the `loss function` based on the `Jacobian matrix` of the `encoder’s output with respect to the input`. The Jacobian measures `how much the hidden layer activations change when the input changes`.
- This regularization `penalizes large changes in the hidden layer activations for small input changes`, making the latent representation "contractive," or resistant to variations in input.
### Benefits of Contractive Autoencoders
- **Robustness to Noise:** CAEs learn features that are less affected by small input variations, which helps with denoising and robustness to input perturbations.
- **Smoother Latent Space:** The latent space becomes more stable and continuous, where similar inputs map closely in the representation space, which is useful for interpretability and clustering.
- **Feature Learning:** CAEs are effective for tasks requiring robust feature extraction from complex data.
### Applications
- **Denoising:** CAEs help remove small, irrelevant noise by focusing on core patterns in the data.
- **Anomaly Detection:** Anomalous inputs may have larger reconstruction errors as they don’t align with the learned stable features.
- **Image and Signal Processing:** Commonly used to extract robust features in images or signals where minor variations need to be ignored.

# 7. Applications of `AE's`:
- **Medical Imaging:**
Autoencoders have shown great promise in medical imaging applications such as 
Magnetic Resonance Imaging (MRI), Computed Tomography (CT), and X-Ray imaging. The ability of
autoencoders to learn feature representations from high-dimensional data has made them useful for
compressing medical images while preserving diagnostic information.
- **Video Compression:**
Autoencoders have also been used for video compression, where the
goal is to compress a sequence of images into a compact
representation that can be transmitted or stored efficiently. One
example of this is the video codec AV1, which uses a combination of
autoencoders and traditional compression methods to achieve higher
compression rates while maintaining video quality
- **Autonomous Vehicles:**
Autoencoders are also useful for autonomous vehicle applications, where the goal
is to compress high-resolution camera images captured by the vehicle’s sensors
while preserving critical information for navigation and obstacle detection.
- **Social Media and Web Applications:**
Autoencoders have also been used in social media and web applications, where
the goal is to reduce the size of image files to improve website loading times and
reduce bandwidth usage. For example, Facebook uses an autoencoder-based
approach for compressing images uploaded to their platform, which achieves high
compression ratios while preserving image quality. This has led to faster loading
times for images on the platform and reduced data usage for users.

------------------------------------------------------------------------------------------------------------------------------------------

# `UNIT - 2`

# 1. Feedforward Neural Network: 
is a type of artificial neural network where connections between the nodes do not form cycles. This characteristic differentiates it from recurrent neural networks (RNNs). The network consists of an input layer, one or more hidden layers, and an output layer. Information flows in one direction—from input to output—hence the name “feedforward.”
Structure of a Feedforward Neural Network
- **Input Layer:**
consists of neurons that receive the input data. Each neuron in the input layer represents a feature of the input data.
- **Hidden Layers:**
One or more hidden layers are placed between the input and output layers. These layers are responsible for learning the complex patterns in the data. Each neuron in a hidden layer applies a weighted sum of inputs followed by a non-linear activation function.
- **Output Layer:**
provides the final output of the network. The number of neurons in this layer corresponds to the number of classes in a classification problem or the number of outputs in a regression problem.

# 2. The learning of a neural network is influenced by:

- **Learning Rate:** Controls the size of weight updates; balancing speed and stability.
- **Architecture:** Depth and width determine the model’s capacity but can affect overfitting.
- **Activation Functions:** Enable non-linearity, crucial for learning complex patterns.
- **Weight Initialization:** Proper initialization aids in stable convergence.
- **Batch Size:** Influences stability and memory usage; impacts generalization.
- **Optimizer:** Algorithms like Adam or SGD guide weight updates; affect convergence speed.
- **Loss Function:** Measures prediction errors, shaping learning goals.
- **Regularization:** Techniques like dropout prevent overfitting.
- **Data Quality & Quantity:** Sufficient, relevant data is essential for generalization.
- **Epochs:** Determines training duration; needs balance to avoid under/overfitting.
- **Hyperparameter Tuning:** Finding optimal settings (e.g., learning rate, batch size) improves performance.


# 3. Activation functions: 
in neural networks is a mathematical function applied to the output of a neuron. Its primary purpose is to introduce non-linearity into the model, enabling the network to learn and represent complex patterns in the data. Here are four common activation functions:
- **The sigmoid function:** 
`(logistic function)` maps any input to a value between 0 and 1. It’s often used in binary classification problems. `y = 1 / 1 + e ^ -x`
	- **Pros:** Smooth gradient, output range (0, 1).
	- **Cons:** Can cause vanishing gradient problems, slow convergence.
- **Tanh function:** 
is very similar to the sigmoid/logistic activation function, and even has the same S-shape with the difference in output range of -1 to 1. In Tanh, the larger the input (more positive), the closer the output value will be to `1.0`, whereas the smaller the input (more negative), the closer the output will be to `-1.0`. `f(x) = (e^x - e^-x) / (e^x +e^-x)`
-  **ReLU:**
is one of the most popular activation functions. It outputs the input directly if it is positive; otherwise, it outputs zero. `f(x) = max(0, x)`
	- **Pros:** Computationally efficient, helps mitigate the vanishing gradient problem.
	- **Cons:** Can cause “dying ReLU” problem where neurons can become inactive.
- **Leaky ReLU (Leaky Rectified Linear Unit):**
is a variation of the ReLU activation function designed to address the “dying ReLU” problem, where neurons can become inactive and only output zero for any input. `f(x) = max(0.1x, x)`
	- The amount of leak is determined by the value of `hyper-parameter α`. It’s value is small and generally varies between 0.01 to 0.1-0.2.
- **Softmax:** 
Primarily used in the output layer for multi-class classification. It converts the input into probability range between 0 to 1. `f(x) = e^xi / ∑ e^xj`

# 4. Types of Gradient Descent
- **Batch Gradient Descent:** 
Uses the entire training dataset to compute the gradient and update the parameters. It is computationally expensive for large datasets but provides a stable convergence.

- **Stochastic Gradient Descent (SGD):** 
Updates weights based on the gradient from a single training example at each iteration. It is faster and can escape local minima but introduces more noise in the updates.
<img width="272" alt="image" src="https://github.com/user-attachments/assets/49d6ac6a-f1d8-4c2f-a4d9-aeb73dbed6a6">
	<img width="265" alt="image" src="https://github.com/user-attachments/assets/20a34c74-cbb2-41f9-8710-ca3a5e59bde9">

- **Mini-Batch Gradient Descent:** 
A compromise between batch and stochastic gradient descent. It uses a small batch of training examples to compute the gradient and update the parameters. It balances the efficiency and stability of the updates.
<img width="218" alt="image" src="https://github.com/user-attachments/assets/1f5e8a4b-10be-4422-a000-95870c35e4c8">

- **Momentum gradient descent:** 
enhances the standard gradient descent by adding a `momentum term`. This helps accelerate the convergence of the training process, reduces oscillations and better handle the local minima / smooth out the updates.
	<img width="350" alt="image" src="https://github.com/user-attachments/assets/f7e9e64a-de87-4de7-b5c8-d7f173dedab7">
    


- **Nesterov Accelerated GD (`NAG`):**
modifies the Momentum-based Gradient Descent by calculating the gradient `not at the current parameters` but with a `look-ahead step` based on the `velocity`.
	- **Look-Ahead:** Instead of calculating the gradient at the current parameters, NAG first performs a look-ahead step to estimate where the parameters will be if the current velocity were applied.
	- **Gradient Calculation:** The gradient is then computed at this look-ahead point, providing a more accurate estimate of the direction in which the parameters should be updated.
	- **Velocity Update:** The velocity term is updated using this more accurate gradient, making the updates more informed and potentially more 	efficient.
	- **Parameter Update:** Finally, the parameters are updated using the updated velocity.
		<img width="317" alt="image" src="https://github.com/user-attachments/assets/746bb2e1-7849-458e-b442-c247eeeedb50">
		- By considering the future position of the parameters, NAG often converges faster than momentum-based gradient descent.
		- The look-ahead mechanism provides more informed updates, which can lead to better convergence properties

- **AdaGrad:**
is an optimization algorithm `designed to adapt the learning rate for each parameter individually based on the historical gradients`. This adaptive nature allows AdaGrad to perform well in scenarios with sparse data and features, where different parameters may have different  degrees of importance and frequency.
  - Key Concepts
	1. **Adaptive Learning Rate:** Unlike traditional gradient descent, which uses a single learning rate for all parameters, AdaGrad adjusts the 		learning rate for each parameter dynamically.
	2. **Accumulation of Squared Gradients:** AdaGrad keeps track of the sum of the squares of the gradients for each parameter. This accumulated value 	is then used to adjust the learning rate.
  - <img width="446" alt="image" src="https://github.com/user-attachments/assets/4c63dfc5-2e70-498b-b97a-3fb775f250f3">
  -  **Advantages**
		- **Adaptivity:** Automatically adjusts learning rates for each parameter, making it effective for problems with sparse features.
		- **Stability:** Reduces the learning rate over time for frequently updated parameters, which can help stabilize convergence.
  - **Disadvantages:** 
  	- **Aggressive Decay:** For some problems, the learning rate might decay too aggressively, causing the learning process to stop too early 	 before reaching the optimal solution

- **RMSProp (Root Mean Square Propagation):** 
Modifies AdaGrad by `decaying the sum of past gradients`, preventing the `learning rate from decaying too quickly.`<br>
(is an adaptive learning rate optimization algorithm `designed to address some of the limitations of AdaGrad`, particularly the `issue of rapidly decaying learning rates`. RMSProp aims to maintain a balance by controlling the learning rate decay, which allows for more stable and faster convergence, especially in deep learning applications.)
![rmsformula](image-14.png)

- **Adam(Adaptive Moment Estimation):**
is an optimization algorithm that `combines the best properties of the AdaGrad and RMSProp` algorithms to provide an efficient and adaptive learning rate. It is particularly `well-suited for` problems involving `large datasets and highdimensional parameter spaces`.

# 5. The bias-variance trade-off: 
is a delicate balance between two types of errors:

- Bias: 
	The difference between the model’s predictions and the true values (high bias leads to underfitting).
	A model with high bias tends to make simplistic assumptions about the data and may underfit the training data.
- Variance: 
	The variability of model predictions for different training datasets (high variance leads to overfitting).
	A model with high variance is sensitive to small fluctuations in the training data and may overfit the training data.

- An ideal model strikes a balance between bias and variance.   Increasing model complexity reduces bias but increases variance, and vice versa. The goal is to find the sweet spot where both bias and variance are minimized, leading to optimal generalization.

# 6. Regularization Methods
- **Early stopping:** One more way to reduce overfitting
    - Early stopping is a regularization technique used in machine learning and deep learning to prevent overfitting during the training of a model.
    - Monitors the model’s performance on a validation set and stops training when performance starts to degrade.
    - Prevents the model from overfitting by halting training before it starts to memorize the training data.


# 7. Generative Adversarial Network (GAN)
Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed for generating new data samples that resemble a given training dataset. Introduced by Ian Goodfellow and his colleagues in 2014, GANs have revolutionized fields like computer vision, natural language processing, and more due to their ability to create highly realistic synthetic data.


![gan image](image-3.png)

> ## Core Components of GANs

A standard GAN architecture consists of two neural networks—the Generator and the Discriminator—that are trained simultaneously through an adversarial process.

> ###  1. Generator:

- Purpose: 	
The Generator's role is to produce synthetic data that mimics the real data distribution. For instance, in image generation tasks, the Generator creates images that resemble those in the training set.

- Architecture:

	- Input: Typically starts with a random noise vector (often sampled from a Gaussian or uniform distribution).
	
	- Layers: Consists of multiple layers such as fully connected layers, convolutional layers (in the case of image 
	data), and activation functions like ReLU or Leaky ReLU.
	- Output: Generates data in the same format as the training data (e.g., images with pixel values).
	
	- Function: The Generator learns to map the input noise to data space, attempting to produce outputs that are 
	indistinguishable from real data to the Discriminator.

> ### 2. Discriminator
- Purpose: 
The Discriminator's role is to distinguish between real data samples and those generated by the Generator.

- Architecture:

	- Input: Receives either real data from the training set or fake data produced by the Generator.

	- Layers: Typically consists of convolutional layers (for image data), pooling layers, fully connected layers, and 
	activation functions like Leaky ReLU and sigmoid.
	- Output: Produces a scalar output representing the probability that the input data is real (ranging between 0 
	and 1).
	- Function: The Discriminator evaluates the authenticity of data samples, providing feedback to both itself and the 
	Generator to improve over time.

> ### Adversarial Training Process

The Generator and Discriminator are trained simultaneously in a two-player minimax game with opposing objectives:
-  Discriminator's Objective: 
Maximize the probability of correctly classifying real and fake data
- Generator's Objective: 
Minimize the Discriminator's ability to distinguish fake data from real data.

This adversarial process continues iteratively, with the Generator improving its ability to produce realistic data and the Discriminator enhancing its capability to detect fakes. Ideally, this process converges when the Generator produces data indistinguishable from real data, and the Discriminator cannot reliably tell the difference.

> ### Applications of GANs
- Image Generation
- Data Augmentation
- Text-to-Image Synthesis
- Video Generation
- Medical Imaging