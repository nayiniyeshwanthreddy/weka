package com.example.weka;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class WekaExample {
    public static void main(String[] args) throws Exception {
        try (InputStream stream = WekaExample.class.getResourceAsStream("/iris.arff")) {
            if (stream == null) {
                throw new IllegalStateException("Resource not found: /iris.arff");
            }

            Instances data = new Instances(new InputStreamReader(stream));
            data.setClassIndex(data.numAttributes() - 1);

            J48 tree = new J48();
            tree.buildClassifier(data);

            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(tree, data, 5, new Random(1));

            System.out.println("=== WEKA J48 Classification Example ===");
            System.out.println("Dataset: " + data.relationName());
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
            System.out.println("\nLearned model:\n" + tree);
        }
    }
}
