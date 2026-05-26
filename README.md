# weka

A simple Java Maven project demonstrating how to use WEKA with a small ARFF dataset.

## Build and run

1. Open a terminal in this project folder.
2. Run:

   ```bash
   mvn clean package
   ```

3. Run one of the sample classes:

   ```bash
   java -jar target/weka-demo-1.0-SNAPSHOT.jar
   ```

   or for the other examples:

   ```bash
   java -cp target/weka-demo-1.0-SNAPSHOT.jar com.example.weka.NaiveBayesExample
   java -cp target/weka-demo-1.0-SNAPSHOT.jar com.example.weka.IBkExample
   ```

## What it does

- Loads a small Iris dataset from `src/main/resources/iris.arff`
- Trains a WEKA `J48` decision tree classifier
- Evaluates the model with 5-fold cross-validation
- Prints summary statistics, class details, confusion matrix, and the learned tree
