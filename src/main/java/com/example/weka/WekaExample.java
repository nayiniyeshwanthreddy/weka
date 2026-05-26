package com.example.weka;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * Example program that demonstrates how to use the WEKA J48 decision tree classifier.
 *
 * <p>This class loads the Iris dataset from a resource file, trains a decision tree,
 * evaluates it with 5-fold cross-validation, and prints the evaluation results.
 */
public class WekaExample {
    public static void main(String[] args) throws Exception {
        // Load the Iris dataset from the application resources.
        try (InputStream stream = WekaExample.class.getResourceAsStream("/iris.arff")) {
            if (stream == null) {
                throw new IllegalStateException("Resource not found: /iris.arff");
            }

            // Parse the dataset from the ARFF file and set the last attribute as the class label.
            Instances data = new Instances(new InputStreamReader(stream));
            data.setClassIndex(data.numAttributes() - 1);

            // Create and train the J48 decision tree classifier on the dataset.
            J48 tree = new J48();
            tree.buildClassifier(data);

            // Evaluate the trained classifier using 5-fold cross-validation.
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 5, new Random(1));

            // Print the evaluation summary and learned model to the console.
            System.out.println("=== WEKA J48 Classification Example ===");
            System.out.println("Dataset: " + data.relationName());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("\nLearned model:\n" + tree);
        }
    }
}
