package com.example.weka;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

public class IBkExample {
    public static void main(String[] args) throws Exception {
        try (InputStream stream = IBkExample.class.getResourceAsStream("/iris.arff")) {
            if (stream == null) {
                throw new IllegalStateException("Resource not found: /iris.arff");
            }

            Instances data = new Instances(new InputStreamReader(stream));
            data.setClassIndex(data.numAttributes() - 1);

            IBk classifier = new IBk(3);
            classifier.buildClassifier(data);

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data, 5, new Random(1));

            System.out.println("=== WEKA IBk (k=3) Classification Example ===");
            System.out.println("Dataset: " + data.relationName());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("\nLearned model:\n" + classifier);
        }
    }
}
