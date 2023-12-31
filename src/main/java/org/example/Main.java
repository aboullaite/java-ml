package org.example;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;
import org.example.visrec.HotDogOrNotVisRec;

public class Main {

  public static void main(String[] args) throws IOException, ClassNotFoundException {
//    HotDogOrNotDl4j.training();
//    HotDogOrNotResnet50Dl4j.transferLearningResnet50();
    HotDogOrNotVisRec.train();


//    // load a trained model/neural network
//    ConvolutionalNetwork convNet =  FileIO.createFromFile("hotdog.dnet", ConvolutionalNetwork.class);
//    // create an image classifier using trained model
//    ImageClassifier<BufferedImage> classifier = new ImageClassifierNetwork(convNet);
//
//    // load image to classify
//    BufferedImage image = ImageIO.read(new File("src/main/resources/dataset/train/hotdog/7.jpg"));
//    // feed image into a classifier to recognize it
//    Map<String, Float> results = classifier.classify(image);
//
//    // interpret the classification result / class probability
//    float hotDogProbability = results.get("hotdog");
//    if (hotDogProbability > 0.5) {
//      System.out.println("There is a high probability that this is a hot dog");
//    } else {
//      System.out.println("Most likely this is not a hot dog");
//    }
  }
}