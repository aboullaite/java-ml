package org.example.visrec;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;
import org.example.dl4j.HotDogOrNotDl4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class InferenceHotdogVisRec {

  private static final Logger LOGGER = LoggerFactory.getLogger(HotDogOrNotDl4j.class);

  public static void main(String[] args) throws IOException, ClassNotFoundException {

//     HotDogOrNotVisRec.train();

    // load a trained model/neural network
    ConvolutionalNetwork convNet =  FileIO.createFromFile("hotdog.dnet", ConvolutionalNetwork.class);
    // create an image classifier using trained model
    ImageClassifier<BufferedImage> classifier = new ImageClassifierNetwork(convNet);

    // load image to classify
    BufferedImage image = ImageIO.read(new File("src/main/resources/visrec/hotdog.jpg"));
    // feed image into a classifier to recognize it
    Map<String, Float> results = classifier.classify(image);

    // interpret the classification result / class probability
    float hotDogProbability = results.get("hotdog");
    if (hotDogProbability > 0.5) {
      LOGGER.info("There is a high probability that this is a hot dog");
    } else {
      LOGGER.info("Most likely this is not a hot dog");
    }

  }
}