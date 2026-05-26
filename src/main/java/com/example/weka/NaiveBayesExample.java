package com.example.weka;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 * Example program that demonstrates how to use the WEKA NaiveBayes classifier.
 *
 * <p>This class loads the Iris dataset from a resource file, trains a NaiveBayes model,
 * evaluates it with 5-fold cross-validation, and prints the evaluation results.
 */
public class NaiveBayesExample {
    public static void main(String[] args) throws Exception {
        // Load the Iris dataset from the classpath resources.
        try (InputStream stream = NaiveBayesExample.class.getResourceAsStream("/iris.arff")) {
            if (stream == null) {
                throw new IllegalStateException("Resource not found: /iris.arff");
            }

            // Read the dataset and tell WEKA which attribute is the class label.
            Instances data = new Instances(new InputStreamReader(stream));
            data.setClassIndex(data.numAttributes() - 1);

            // Create and train the NaiveBayes classifier.
            NaiveBayes classifier = new NaiveBayes();
            classifier.buildClassifier(data);

            // Evaluate the classifier using 5-fold cross-validation.
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 5, new Random(1));

            // Print results so users can inspect accuracy and confusion matrix.
            System.out.println("=== WEKA NaiveBayes Classification Example ===");
            System.out.println("Dataset: " + data.relationName());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("\nLearned model:\n" + classifier);
        }
    }
}
