package org.example.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HotDogOrNotDl4j {

  private static final Logger LOGGER = LoggerFactory.getLogger(HotDogOrNotDl4j.class);
  private static int height = 100;
  private static int width = 100;
  private static int channels = 3;
  private static int numLabels = 2;
  private static int batchSize = 50;
  private static int seed = 123;
  private static final String TRAINING_DATASET = "src/main/resources/dataset/train";
  private static final String TESTING_DATASET = "src/main/resources/dataset/test";
  private static Random rng = new Random(seed);
  private static int epochs = 2;

  public static void training() throws IOException {

    LOGGER.info("Loading data....");

    /**
     * Setting up data
     */
    ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    File testPath = new File(TESTING_DATASET);
    File trainPath = new File(TRAINING_DATASET);
    FileSplit test = new FileSplit(testPath, NativeImageLoader.ALLOWED_FORMATS, rng);
    FileSplit train = new FileSplit(trainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
    numLabels = test.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.


    /**
     *  Create extra synthetic training data by flipping, rotating
     #  images on our data set.
     */
    ImageTransform resizeTransform = new ResizeImageTransform(rng, width, height);

    /**
     * Normalization
     **/
    LOGGER.info("Fitting to dataset");
    ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
    /**
     * Define our network architecture:
     */
    LOGGER.info("Build model....");
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .seed(seed)
        .l2(0.0005) // tried 0.0001, 0.0005
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Nesterovs())
        .list()
        .layer(0, new ConvolutionLayer.Builder(5, 5)
            .nIn(channels)
            .stride(1, 1)
            .nOut(20)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(2, new ConvolutionLayer.Builder(5, 5)
            .stride(1, 1)
            .nOut(50)
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .build())
        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())
        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
            .nOut(500).build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numLabels)
            .dropOut(0.3)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutional(height, width, channels))
        .build();


    MultiLayerNetwork network = new MultiLayerNetwork(conf);

    network.init();
    // Visualizing Network Training
    UIServer uiServer = UIServer.getInstance();
    StatsStorage statsStorage = new InMemoryStatsStorage();
    uiServer.attach(statsStorage);
    network.setListeners(new ScoreIterationListener(10), new StatsListener(statsStorage));

    /**
     * Load data
     */
    LOGGER.info("***** LOADING DATA *****");
    ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
    DataSetIterator dataIter;
    MultiLayerNetwork model = new MultiLayerNetwork(conf);


    LOGGER.info("***** SETTING UP MODEL CONFIG *****");
    // Train without transformations
    recordReader.initialize(train, resizeTransform);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
    preProcessor.fit(dataIter);
    dataIter.setPreProcessor(preProcessor);
    // The Score iteration Listener will log
    // output to show how well the network is training
    model.setListeners(new ScoreIterationListener(10));

    LOGGER.info("*****TRAINING MODEL STARTED ********");
    for(int i = 0; i<epochs; i++){
      model.fit(dataIter);
    }


    LOGGER.info("***** EVALUATING MODEL *****");
    recordReader.initialize(test, resizeTransform);
    dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
    preProcessor.fit(dataIter);
    dataIter.setPreProcessor(preProcessor);
    Evaluation eval = network.evaluate(dataIter);
    LOGGER.info(eval.stats(true));


      LOGGER.info("***** SAVING MODEL *****");
//      ModelSerializer.writeModel(network,  "hotdog.bin", true);

    LOGGER.info("**************** HotDog Classification finished ********************");
  }
}
