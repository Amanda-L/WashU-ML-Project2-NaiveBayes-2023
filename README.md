# WashU ML Project 2: Naive Bayes
![image](https://github.com/Amanda-L/WashU-ML-Project2-NaiveBayes-2023/assets/52643725/82d65194-2b62-4da3-998d-79136926ffea)


View 02NaiveBayes.html under Instructions folder for detailed instructions on the assignment.

The Naïve Bayes classifier is being implemented to predict if a name is male or female for this assignment, and the files edited are outlined as follows:

1. **naivebayesPY.py:**
   - Implement the function to estimate the class probability P(Y).
   - This returns the probability of a sample in the training set being positive or negative, regardless of its features.

2. **naivebayesPXY.py:**
   - Implement the function to estimate conditional probabilities P(X|Y).
   - This uses a categorical distribution as the model.
   - Utilize array operations in numpy instead of for loops.

3. **naivebayes.py:**
   - Solve for the log ratio using Bayes Rule.
   - Implement the calculation of log ratio.

4. **naivebayesCL.py:**
   - Implement the Naïve Bayes as a linear classifier.

5. **classifyLinear.py:**
   - Implement the `classifyLinear.py` function that applies a linear weight vector and bias to a set of input vectors and outputs their predictions.

Testing:
- Generate training features using `genTrainFeatures()`.
- Train Naïve Bayes using `naivebayesCL(x, y)`.
- Use the trained model to predict on the training set and calculate training error.

6. **whoareyou.py:**
   - Implement to test names interactively using the trained Naïve Bayes model.

Once implemented, I tested the overall performance of the Naïve Bayes classifier and its interactive name classification using `whoareyou.py`.

In a script:
```
xTr, yTr = genTrainFeatures()

w,b = naivebayesCL(xTr,yTr)

whoareyou(w,b)
```

In the terminal:
```
Who are you>David

log( ) P(Y=1|X)

P(Y=−1|X)

David, I am sure you are a nice boy.

Who are you>Anne

Anne, I am sure you are a nice girl.

Who are you>byebye
```



