# Project 2: Naive Bayes

The Naïve Bayes classifier is being implemented for this assignment, and the tasks are outlined as follows:

1. **naivebayesPY.py:**
   - Implement the function to estimate the class probability P(Y).
   - This returns the probability of a sample in the training set being positive or negative, regardless of its features.

2. **naivebayesPXY.py:**
   - Implement the function to estimate conditional probabilities P(X|Y).
   - This uses a categorical distribution as the model.
   - Utilize array operations in numpy instead of for loops.

3. **naivebayes.py:**
   - Solve for the log ratio using Bayes Rule.
   - Implemented the calculation of log ratio in this file.

4. **naivebayesCL.py:**
   - Implement the Naïve Bayes as a linear classifier.

5. **classifyLinear.py:**
   - Implement the `classifyLinear.py` function that applies a linear weight vector and bias to a set of input vectors and outputs their predictions.

Testing:
- Generate training features using `genTrainFeatures()`.
- Train Naïve Bayes using `naivebayesCL(x, y)`.
- Use the trained model to predict on the training set and calculate training error.

(f) **whoareyou.py:**
   - Implement the `whoareyou.py` function to test names interactively using the trained Naïve Bayes model.

Once implemented, I tested the overall performance of the Naïve Bayes classifier and its interactive name classification using `whoareyou.py`.


View 02NaiveBayes.html for detailed instructions on the assignment.
