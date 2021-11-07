package ch.zhaw.widmemor;

import ch.zhaw.hassebjo.Word2VecTest;
import ch.zhaw.widmemor.model.UserStory;
import ch.zhaw.widmemor.model.UserStoryContainer;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ClassifyUserStoriesImpl implements ClassifyUserStories {

    private static final Logger LOGGER = LoggerFactory.getLogger(ClassifyUserStoriesImpl.class);

    private static final int GLOVE_DIM = 100;

    private static final String RESOURCE = "ch/zhaw/hassebjo/";
    private static final String GLOVE_VECTORS = RESOURCE + "glove.6B." + GLOVE_DIM + "d.txt";
    private static final String LABELLED_TURNS = RESOURCE + "train_set_mini.txt";
    private static final String TEST_SET = RESOURCE + "test_set.txt";

    private static final String MODELS_DIRECTORY = "output/models";

    // captures everything that's not a letter greedily, e.g. ", "
    private static final String WORD_SPLIT_PATTERN = "\\P{L}+";
    // columns are split by tabs
    private static final String INPUT_TURN_DELIMITER_PATTERN = "\\t+";

    @Override
    public UserStoryContainer processFile(Integer projectId,
                                          byte[] inputBytes) throws IOException {
        WordVectors wordVectors = getWordVectors();

        MultiLayerNetwork model = getModel();

        //URL validationSet = loadResource(TEST_SET);

        //input columns must be 10000 for test_set.txt - idk why
        //words per Turn is a guess at how many words a person speaks?
        return evaluate(projectId, inputBytes, model, wordVectors, 10000, 100);

    }

    private WordVectors getWordVectors() throws FileNotFoundException {
        URL glove = loadResource(GLOVE_VECTORS);
        return WordVectorSerializer.readWord2VecModel(new File(glove.getFile()));
    }

    private MultiLayerNetwork getModel() throws IOException {
        String modelFileName = "model_6b_" + GLOVE_DIM + "d_v1_0.bin";
        return ModelSerializer.restoreMultiLayerNetwork(new File(MODELS_DIRECTORY, modelFileName));
    }

    /**
     * @return an INDArray with the shape: #wordsPerTurn x #inputColumns
     */
    private static INDArray pad(INDArray wvm, int wordsPerTurn, int inputGloveDimension) {
        if (wvm.rows() < wordsPerTurn) {
            wvm = Nd4j.vstack(wvm, Nd4j.zeros(wordsPerTurn - wvm.rows(), inputGloveDimension)).reshape(wordsPerTurn,
                    inputGloveDimension);
        }

        if (wvm.rows() > wordsPerTurn) {
            wvm = wvm.getRows(IntStream.rangeClosed(0, wordsPerTurn - 1).toArray());
        }

        return wvm; // .reshape(wordsPerTurn, inputColumns);
    }

    private static UserStoryContainer evaluate(Integer projectId,
                                 byte[] inputBytes,
                                 MultiLayerNetwork model,
                                 WordVectors wordVectors,
                                 int inputColumns,
                                 int wordsPerTurn) throws IOException {


        int numberOfLines = getNumberOfLines(new ByteArrayInputStream(inputBytes));

        List<INDArray> inputList = new ArrayList<>(numberOfLines);
        double[][] labelsList = new double[numberOfLines][];

        UserStoryContainer userStoryContainer = new UserStoryContainer();

        ByteArrayInputStream inputStream = new ByteArrayInputStream(inputBytes);
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {

            String line;
            int labelsId = 0;
            while ((line = reader.readLine()) != null) {
                String[] lineAr = line.split(INPUT_TURN_DELIMITER_PATTERN);
                Optional<INDArray> inputMatrix = getInputValueMatrix(lineAr, wordVectors, wordsPerTurn);
                inputList.add(inputMatrix.get().ravel());


                labelsList[labelsId++] = getEvalLabel(lineAr);

                String labelStr = lineAr[2];
                if (labelStr.equals("F")) {
                    UserStory userStory = new UserStory(projectId, lineAr[1].trim());
                    userStoryContainer.addUserStory(userStory);

                    /*ObjectMapper objectMapper = new ObjectMapper();
                    String userStoryAsString = objectMapper.writeValueAsString(userStory);
                    System.out.println(userStoryAsString);*/
                }
            }
        }

        ObjectMapper objectMapper = new ObjectMapper();
        String userStoryContainerAsString = objectMapper.writeValueAsString(userStoryContainer);
        System.out.println(userStoryContainerAsString);

        INDArray evalInput = Nd4j.create(inputList, shape(numberOfLines, inputColumns));
        INDArray evalLabels = Nd4j.create(labelsList);
        Evaluation eval = new Evaluation(3);
        eval.eval(evalLabels, evalInput, model);
        LOGGER.info(eval.stats());

        return userStoryContainer;
    }

    private static int getNumberOfLines(ByteArrayInputStream textForClassification) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(textForClassification))) {
            int noOfLines = 0;
            try (LineNumberReader lineNumberReader = new LineNumberReader(reader)) {
                lineNumberReader.skip(Integer.MAX_VALUE);
                noOfLines = lineNumberReader.getLineNumber() + 1;
            }
            return noOfLines;
        }
    }

    private static double[] getEvalLabel(String[] lineAr) {
        String labelStr = lineAr[2];
        switch (labelStr) {
            case "NULL":
                return new double[]{1, 0, 0};
            case "A":
                return new double[]{0, 1, 0};
            case "F":
                return new double[]{0, 0, 1};
        }

        throw new IllegalArgumentException("Unknown label: " + labelStr);
    }

    private static int getLabelValue(String[] lineAr) {
        String labelStr = lineAr[2];
        return getLabel(labelStr);
    }

    private static int getLabel(String labelStr) {
        switch (labelStr) {
            case "NULL":
                return 0;
            case "A":
                return 1;
            case "F":
                return 2;
        }

        throw new IllegalArgumentException("Unknown label: " + labelStr);
    }

    /**
     * @return input values or null if they couldn't be matched
     */
    private static Optional<INDArray> getInputValueMatrix(String[] lineAr, WordVectors wordVectors, int maxWords) {
        String text = lineAr[1].trim();
        String[] textWordAr = text.split(WORD_SPLIT_PATTERN); // split by non letter characters

        String[] ar;
        if (textWordAr.length > maxWords) {
            ar = (String[]) ArrayUtils.subarray(textWordAr, 0, maxWords);
        } else {
            ar = textWordAr;
        }

        List<String> wordList = getWordList(ar, wordVectors);
        if (wordList.isEmpty()) {
            return Optional.empty();
        }

        return Optional.ofNullable(wordVectors.getWordVectors(wordList));
    }

    /**
     * Filter words which are not in the vocabulary, to prevent failing edge cases
     */
    private static List<String> getWordList(String[] textWordAr, WordVectors wordVectors) {
        return List.of(textWordAr).stream().filter(w -> wordVectors.hasWord(w)).collect(Collectors.toList());
    }

    private static int[] shape(int rows, int columns) {
        return new int[]{rows, columns};
    }

    private static final URL loadResource(String name) throws FileNotFoundException {
        LOGGER.info("loading " + name);
        URL url = Word2VecTest.class.getClassLoader().getResource(name);
        return Optional.ofNullable(url).orElseThrow(() -> new FileNotFoundException(name));
    }

    public static MultiLayerNetwork createMultiLayerNetwork(int nIn, double learningRate) throws IOException {

        int rngSeed = 123; // random number seed for reproducibility
        int optimizationIterations = 1;
        int outputNum = 3; // number of output classes: FR, NFR, None

        //@formatter:off
        MultiLayerConfiguration multiLayerConf = new NeuralNetConfiguration.Builder()
                // high level configuration
                .seed(rngSeed)
                .learningRate(learningRate)
                .regularization(true).l2(1e-4)
                .iterations(optimizationIterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAM)
                .weightInit(WeightInit.XAVIER)
                // layer configuration
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(nIn)
                        .nOut(10000)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(1, new DropoutLayer.Builder(0.6)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(10000)
                        .nOut(5000)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(3, new DenseLayer.Builder()
                        .nIn(5000)
                        .nOut(100)
                        .activation(Activation.TANH)
                        .build())
                .layer(4, new OutputLayer.Builder()
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .build())

                // Pretraining and Backprop configuration
                .pretrain(false)
                .backprop(true)

                .build();
        //@formatter:on
        // System.out.println(multiLayerConf.toJson());
        return new MultiLayerNetwork(multiLayerConf);
    }
}
