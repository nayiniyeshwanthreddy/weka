package com.example.weka;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 * Example program that demonstrates how to use the WEKA IBk k-nearest neighbors classifier.
 *
 * <p>This class loads the Iris dataset from a resource file, trains an IBk model with k=3,
 * evaluates it with 5-fold cross-validation, and prints the evaluation results.
 */
public class IBkExample {
    public static void main(String[] args) throws Exception {
        // Load the Iris dataset from resources on the classpath.
        try (InputStream stream = IBkExample.class.getResourceAsStream("/iris.arff")) {
            if (stream == null) {
                throw new IllegalStateException("Resource not found: /iris.arff");
            }

            // Read the dataset and set the class attribute to the last column.
            Instances data = new Instances(new InputStreamReader(stream));
            data.setClassIndex(data.numAttributes() - 1);

            // Create the k-NN classifier with k=3 neighbors.
            IBk classifier = new IBk(3);
            classifier.buildClassifier(data);

            // Evaluate the model using 5-fold cross-validation.
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 5, new Random(1));

            // Print the results so users can compare performance with other algorithms.
            System.out.println("=== WEKA IBk (k=3) Classification Example ===");
            System.out.println("Dataset: " + data.relationName());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("\nLearned model:\n" + classifier);
        }
    }
}
