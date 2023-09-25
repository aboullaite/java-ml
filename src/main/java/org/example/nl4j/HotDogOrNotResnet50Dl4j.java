package org.example.nl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

public class HotDogOrNotResnet50Dl4j {
  private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(HotDogOrNotResnet50Dl4j.class);

  private static final int width = 224;
  private static final int height = 224;
  private static final int channel = 3;
  private static final int batchSize = 64;
  private static final int numOfClass = 2;
  private static ComputationGraph resnet50Transfer;
  private static int seed = 123;
  private static Random rng = new Random(seed);

  private static final String TRAINING_DATASET = "/Users/maboullaite/Projects/ml-java/dl4j/src/main/resources/dataset/train";
  private static final String TESTING_DATASET = "/Users/maboullaite/Projects/ml-java/dl4j/src/main/resources/dataset/test";


  public static void transferLearningResnet50() throws IOException {

    int seed = 1234;
    Random randNumGen = new Random(seed);
    String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    File testPath = new File(TESTING_DATASET);
    File trainPath = new File(TRAINING_DATASET);
    FileSplit testData = new FileSplit(testPath, NativeImageLoader.ALLOWED_FORMATS, rng);
    FileSplit trainData = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS, rng);

    ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();
    BalancedPathFilter balancedPathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelGenerator);



    ImageRecordReader trainRecordReader = new ImageRecordReader(height, width,channel,  labelGenerator);
    ImageRecordReader testRecordReader = new ImageRecordReader(height, width, channel, labelGenerator);

    trainRecordReader.initialize(trainData);
    testRecordReader.initialize(testData);

    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, numOfClass);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, numOfClass);

    trainIter.setPreProcessor(scaler);
    testIter.setPreProcessor(scaler);

    ZooModel zooModel = ResNet50.builder().build();
    ComputationGraph resnet = (ComputationGraph) zooModel.initPretrained();
    LOGGER.info(resnet.summary());

    FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
        .updater(new Adam(1e-3))
        .seed(seed)
        .build();

    ComputationGraph resnet50Transfer = new TransferLearning.GraphBuilder(resnet)
        .fineTuneConfiguration(fineTuneConf)
        .setFeatureExtractor("bn5b_branch2c") //"block5_pool" and below are frozen
        .addLayer("fc",new DenseLayer
            .Builder().activation(Activation.RELU).nIn(1000).nOut(256).build(),"fc1000") //add in a new dense layer
        .addLayer("newpredictions",new OutputLayer
            .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(256)
            .nOut(numOfClass)
            .build(),"fc") //add in a final output dense layer,
        // configurations on a new layer here will be override the finetune confs.
        // For eg. activation function will be softmax not RELU
        .setOutputs("newpredictions") //since we removed the output vertex and it's connections we need to specify outputs for the graph
        .build();

    UIServer server = UIServer.getInstance();
    StatsStorage storage = new InMemoryStatsStorage();
    server.attach(storage);
    resnet50Transfer.setListeners(new ScoreIterationListener(50), new StatsListener(storage));
    double lowest = 10;

    for (int i = 1; i < 50 + 1; i++) {
      trainIter.reset();
      resnet50Transfer.fit(trainIter);
      if (resnet50Transfer.score() < lowest) {
        lowest = resnet50Transfer.score();
        String modelFilename = new File(".").getAbsolutePath() + "/CatsDogsClassifier_loss" + lowest + "_ep" + i + "ResNet50.zip";
//                ModelSerializer.writeModel(resnet50Transfer, modelFilename, false);
      }
      LOGGER.info("Completed epoch {}", i);
//        System.out.println(NetworkUtils.getLearningRate(resnet50Transfer, "output"));
//            System.out.println(String.format("%d,%.2f", i, tunedModel.evaluate(trainIterator).accuracy()));
    }
    ModelSerializer.writeModel(resnet50Transfer, "Final_ResNet50_v2.zip", false);
    Evaluation trainEval = resnet50Transfer.evaluate(trainIter);
    Evaluation testEval = resnet50Transfer.evaluate(testIter);
    LOGGER.info(trainEval.stats());
    LOGGER.info(testEval.stats());
  }
}
