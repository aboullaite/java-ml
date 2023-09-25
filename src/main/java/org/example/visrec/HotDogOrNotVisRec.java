package org.example.visrec;;

import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import java.awt.image.BufferedImage;
import java.nio.file.Paths;

/**
 * Example how to train the hot dog image classifier in Java using VisRec API JSR381
 * See UseTrainedHotDogClassifier for how to use trained hot dog classifier.
 *
 * Scene from TV show Silicon Valley https://www.youtube.com/watch?v=vIci3C4JkL0
 * Data set: https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
 *
 * Visual Recognition API JSR381 https://jcp.org/en/jsr/detail?id=381
 * JSR project on GitHub https://github.com/JavaVisRec
 *
 */
public class HotDogOrNotVisRec {

  public static void train() {
    try {
      ImageClassifier<BufferedImage> classifier =
          NeuralNetImageClassifier.builder()
              .inputClass(BufferedImage.class) // input class for classifier
              .imageWidth(224) // width of the input image
              .imageHeight(224) // height of the input image
              .labelsFile(Paths.get("src/main/resources/visrec/labels.txt"))// list of image labels
              .trainingFile(Paths.get("src/main/resources/dataset/index.txt")) // index of images with corresponding labels
              .networkArchitecture(Paths.get("src/main/resources/visrec/hotdog.json"))// architecture of the convolutional neural network in json
              .maxError(0.03f) // error level to stop the training (maximum acceptable error)
              .maxEpochs(1000) // maximum number of training iterations (epochs)
              .learningRate(0.01f)// amount of error to use for adjusting internal parameters in each training iteration
              .exportModel(Paths.get("hotdog.dnet")) // name of the file to save trained model
              .build();

    } catch (ModelCreationException e) { // if something goes wrong an exception is thrown
      System.out.println("Model creation failed! " + e.getMessage());
    }
  }
}