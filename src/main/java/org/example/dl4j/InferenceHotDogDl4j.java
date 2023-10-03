package org.example.dl4j;

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

public class InferenceHotDogDl4j {

  public static void main(String[] args)
      throws IOException, InterruptedException {
//    HotDogOrNotDl4j.training();
    HotDogOrNotResnet50Dl4j.transferLearningResnet50();
  }
}