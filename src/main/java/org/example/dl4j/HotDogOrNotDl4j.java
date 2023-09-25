package org.example.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HotDogOrNotDl4j {

  private static final Logger LOGGER = LoggerFactory.getLogger(HotDogOrNotDl4j.class);
  private static int height = 256;
  private static int width = 256;
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
    int numExamples = Math.toIntExact(test.length());
    numLabels = test.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
    BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);


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
        .l2(0.005) // tried 0.0001, 0.0005
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .updater(new Adam())
        .list()
        .layer(new ConvolutionLayer.Builder().name("cnn1").kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.RELU)
            .nIn(channels).nOut(64).dropOut(0.25).build())
        .layer(new SubsamplingLayer.Builder().name("maxpool1").kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
        .layer(new ConvolutionLayer.Builder().name("cnn2").kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.RELU)
            .nIn(channels).nOut(64).dropOut(0.25).build())
        .layer(new SubsamplingLayer.Builder().name("maxpool2").kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
        .layer(new ConvolutionLayer.Builder().name("cnn3").kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.RELU)
            .nIn(channels).nOut(128).dropOut(0.3).build())
        .layer(new SubsamplingLayer.Builder().name("maxpool2").kernelSize(2,2).stride(2,2).poolingType(
            PoolingType.AVG).build())
        .layer(new ConvolutionLayer.Builder().name("cnn3").kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.RELU)
            .nIn(channels).nOut(128).dropOut(0.3).build())
        .layer(new SubsamplingLayer.Builder().name("maxpool3").kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .name("output")
            .nOut(numLabels)
            .dropOut(0.3)
            .activation(Activation.SOFTMAX)
            .build())
        .setInputType(InputType.convolutional(height, width, channels))
        .build();

//    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//        .seed(seed)
//        .updater(new AdaDelta())
//        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//        .weightInit(WeightInit.XAVIER)
//        .list()
//        .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nIn(channels).nOut(96).build())
//        .layer(new BatchNormalization())
//        .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
//
//        .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(256).build())
//        .layer(new BatchNormalization())
//        .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(512).build())
//        .layer(new BatchNormalization())
//        .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())
//
//        .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(384).build())
//        .layer(new BatchNormalization())
//        .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(256).build())
//        .layer(new BatchNormalization())
//        .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(96).build())
//        .layer(new BatchNormalization())
//        .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
//            .nOut(numLabels).build())
//        .layer(new BatchNormalization())
//        .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())
//        .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//            .name("output")
//            .nOut(numLabels)
//            .dropOut(0.8)
//            .activation(Activation.SOFTMAX)
//            .build())
//        .setInputType(InputType.convolutional(height, width, channels))
//        .build();
    MultiLayerNetwork network = new MultiLayerNetwork(conf);

    network.init();
    // Visualizing Network Training
//    UIServer uiServer = UIServer.getInstance();
//    StatsStorage statsStorage = new InMemoryStatsStorage();
//    uiServer.attach(statsStorage);
//    network.setListeners((IterationListener) new StatsListener( statsStorage),new ScoreIterationListener(iterations));

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
