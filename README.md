# Single Layer Neural Network for Image Classification (C)

This project is a pure C implementation of a **binary classification** model using 28x28 pixel PGM images. The model is designed as a single-layer neural network (or logistic/linear classifier) utilizing the hyperbolic tangent (`tanh`) activation function.

The project includes three different gradient-based optimization algorithms to train the model's weights:

1.  **Gradient Descent (GD)**
2.  **Stochastic Gradient Descent (SGD)**
3.  **ADAM (Adaptive Moment Estimation)**

## 🚀 Key Features

  * **PGM Image Loader:** Reads and processes 28x28 PGM (P2 format) grayscale images.
  * **Data Normalization:** Normalizes pixel values from the [0, 255] range to the [0, 1] range.
  * **Bias Term:** Automatically includes a bias term (a constant '1') in every input vector.
  * **Three Optimizers:** Allows for training with GD, SGD, and ADAM algorithms.
  * **Error and Weight Logging:** Logs the training error, test error, elapsed time, and model weights for each iteration into `.txt` files. This is useful for visualizing the training progress (e.g., with Python/Matplotlib or Gnuplot).

## ⚙️ Project Details

### Model Architecture

  * **Input Vector Size:** `VECTOR_SIZE = 785`
      * $28 \times 28 = 784$ pixel values
      * $+$ 1 bias term
  * **Activation Function:** Hyperbolic Tangent ($\tanh$)
  * **Output:** A single scalar value (prediction).
      * $\text{Prediction} = \tanh(W^T X)$
      * $W$: Weight vector (785-dimensional)
      * $X$: Input vector (785-dimensional)
  * **Loss Function:** Least Squares Error
      * $E = \sum_{i=1}^{N} (Y_i - \tanh(W^T X_i))^2$

### Code Configuration (`main.c`)

  * **Dataset Sizes:**
      * `DATASET_SIZE`: 800 (Training)
      * `TESTSET_SIZE`: 200 (Testing)
  * **Optimization Parameters:**
      * `STEP_SIZE`: 0.0001 (Learning rate for GD)
      * `MINI_BATCH_SIZE`: 80 (Batch size for SGD)
      * `MAX_ITERATION`: 1000 (Maximum iterations)

## 📦 Dataset Structure

This program expects the PGM files to be present in the working directory, following a specific naming convention. The `imageName()` function dynamically generates these filenames.

**Required File Structure:**

```
project_directory/
├── 0_train.pgm
├── 1_train.pgm
...
├── 799_train.pgm
├── 0_test.pgm
├── 1_test.pgm
...
├── 199_test.pgm
├── main.c
└── classifier (compiled executable)
```

**Note:** The program labels the first half of the dataset (e.g., `0` - `399` for training, `0` - `99` for testing) as class **+1** and the second half as class **-1**.

## ▶️ Compilation and Execution

The program uses standard C libraries along with `math.h` and `time.h` (for the POSIX `clock_gettime` function).

1.  **Compilation (GCC):**
    You can compile the program using `gcc`. You must link the math library (`-lm`) and the real-time library (`-lrt`).

    ```bash
    gcc main.c -o classifier -lm -lrt
    ```

2.  **Execution:**
    Run the compiled executable (ensure the PGM files are in the same directory):

    ```bash
    ./classifier
    ```

3.  **Algorithm Selection:**
    You can choose which optimization algorithm to run from within the `main` function in `main.c`. In the current code, the `gradientDescent` function is called. To try other algorithms (SGD or ADAM), simply comment out the current call and uncomment the function you wish to use.

## 📊 Outputs

After running, the program will generate the following files, (the numbers in the filenames come from the `saveError` and `saveW` functions in the code):

  * **Error Logs** (`gdError2701.txt`, `sgdError1911.txt`, `adamError0.txt`):

      * Each line represents one iteration.
      * Format: `Training_Error,Test_Error,Elapsed_Time_ms`

  * **Weight Logs** (`gdW2701.txt`, `sgdW1911.txt`, `adamW0.txt`):

      * Each line contains the 785 weight values for that iteration, separated by commas.
