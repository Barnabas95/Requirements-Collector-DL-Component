package ch.zhaw.hassebjo;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.standalone.ClassPathResource;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Word2VecTest {

	private static final Logger LOGGER = LoggerFactory.getLogger(Word2VecTest.class);

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

	// https://deeplearning4j.org/tutorials/setup

	public static void main(String[] args) throws IOException {
		MultiLayerNetwork model = createMultiLayerNetwork(10000, 0.25);
		System.out.println(model.toString());
		System.out.println(model.conf().toJson());
	}

	public static void main2(String[] args) throws IOException {

		URL glove = loadResource(GLOVE_VECTORS);
		WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(glove.getFile()));

		String modelFileName = "model_6b_" + GLOVE_DIM + "d_v1_0.bin";

		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File(MODELS_DIRECTORY, modelFileName));

		URL validationSet = loadResource(TEST_SET);

		evaluate(validationSet, 15, model, wordVectors, 5000, 100);

	}

	public static void main3(String[] args) throws IOException, URISyntaxException {

		// the training set
		URL labelledTurns = loadResource(LABELLED_TURNS);
		// the test set (but used for validation..)
		URL validationSet = loadResource(TEST_SET);
		// load pre-trained GloVe w2v
		URL glove = loadResource(GLOVE_VECTORS);
		WordVectors wordVectors = WordVectorSerializer.readWord2VecModel(new File(glove.getFile()));

		String modelFileName = "model_6b_" + GLOVE_DIM + "d_v1_0.bin";

		int nrOfExamples = 388; // the number of lines (turns) in #LABELLED_TURNS
		int nrOfBatches = 4;
		int nrOfExamplesPerBatch = 97;

		int nrOfEpochs = 100;

		int wordsPerTurn = 100; // estimated. Longer will be cut, shorter will be empty padded.
		int gloveDimension = GLOVE_DIM; // vector dimension of the word representation in the GloVe model

		// columns of input. The rows are the number of examples.
		int inputColumns = wordsPerTurn * gloveDimension;

		// create the model
		LOGGER.info("Creating model");
		MultiLayerNetwork model = createMultiLayerNetwork(inputColumns, 0.3);
		model.init();

		// save it to file
		File modelFile = new File(MODELS_DIRECTORY, modelFileName);
		modelFile.createNewFile();
		LOGGER.info("Saving model to " + modelFile);
		// TODO what does "saveUdpater" mean?
		ModelSerializer.writeModel(model, modelFile, true);

		LOGGER.info("Start training...");
		train(labelledTurns, wordVectors, nrOfExamples, nrOfBatches, nrOfExamplesPerBatch, nrOfEpochs, inputColumns,
				wordsPerTurn, gloveDimension, model, validationSet, 15);
		LOGGER.info("Finished training");

		LOGGER.info("Save trained model to " + modelFile);
		// save parameter
		ModelSerializer.writeModel(model, modelFile, true);

		LOGGER.info("Evaluate model");
		evaluate(validationSet, 15, model, wordVectors, inputColumns, wordsPerTurn);
	}

	private static void train(URL labelledTurns, WordVectors wordVectors, int nrOfExamples, int nrOfBatches,
			int nrOfExamplesPerBatch, int nrOfEpochs, int inputColumns, int wordsPerTurn, int gloveDimension,
			MultiLayerNetwork model, URL validationSet, int inputSize) throws IOException, FileNotFoundException {

		// prepare test data
		List<INDArray> inputList = new ArrayList<>(inputSize);
		double[][] labelsList = new double[inputSize][];
		try (BufferedReader reader = new BufferedReader(new FileReader(validationSet.getFile()))) {

			String line;
			int labelsId = 0;
			while ((line = reader.readLine()) != null) {
				String[] lineAr = line.split(INPUT_TURN_DELIMITER_PATTERN);
				Optional<INDArray> inputMatrix = getInputValueMatrix(lineAr, wordVectors, wordsPerTurn);
				inputList.add(inputMatrix.get().ravel());
				labelsList[labelsId++] = getEvalLabel(lineAr);
			}

		}
		INDArray evalInput = Nd4j.create(inputList, shape(inputList.size(), inputColumns));
		INDArray evalLabels = Nd4j.create(labelsList);

		LOGGER.info("Train with " + nrOfExamplesPerBatch + " examples in " + nrOfBatches + " batches for " + nrOfEpochs
				+ " epochs");
		for (int epoch = 0; epoch < nrOfEpochs; ++epoch) {
			LOGGER.info("Epoch " + epoch);

			try (BufferedReader reader = new BufferedReader(new FileReader(labelledTurns.getFile()))) {

				String line = reader.readLine();
				for (int batch = 0; line != null && batch < nrOfBatches; ++batch) {
					LOGGER.info("Batch " + batch);

					List<INDArray> inputsBatch = new ArrayList<>(nrOfExamplesPerBatch);
					int[] labelsBatch = new int[nrOfExamplesPerBatch];

					for (int example = 0; example < nrOfExamplesPerBatch; ++example, line = reader.readLine()) {
						// LOGGER.info("Example " + example);

						if (line == null) {
							LOGGER.warn("Premature end of test set file. Expected: {}. Actual: {}.", nrOfExamples,
									(batch + 1) * nrOfExamplesPerBatch + example + 1);
							for (; example < nrOfExamplesPerBatch; ++example) {
								// fill labels array, so input and label shape match
								labelsBatch[example] = -1;
							}
							break;
						}

						String[] lineAr = line.split(INPUT_TURN_DELIMITER_PATTERN);

						// check if matrix has been built
						Optional<INDArray> wordVectorMatrixOpt = getInputValueMatrix(lineAr, wordVectors, wordsPerTurn);
						if (!wordVectorMatrixOpt.isPresent()) {
							labelsBatch[example] = -1;
							continue;
						}

						// pad
						INDArray wvm = wordVectorMatrixOpt.get();
						wvm = pad(wvm, wordsPerTurn, gloveDimension);

						inputsBatch.add(wordVectorMatrixOpt.get().ravel());
						labelsBatch[example] = getLabelValue(lineAr);
					}

					INDArray input = Nd4j.create(inputsBatch, shape(inputsBatch.size(), inputColumns));
					while (ArrayUtils.contains(labelsBatch, -1)) {
						labelsBatch = ArrayUtils.removeElement(labelsBatch, -1);
					}

					model.fit(input, labelsBatch);
				}

			}

			Evaluation eval = new Evaluation(3);
			eval.eval(evalLabels, evalInput, model);
			LOGGER.info(eval.stats());

		}

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

	private static void evaluate(URL validationSet, int inputSize, MultiLayerNetwork model, WordVectors wordVectors,
			int inputColumns, int wordsPerTurn) throws IOException {

		List<INDArray> inputList = new ArrayList<>(inputSize);
		double[][] labelsList = new double[inputSize][];

		try (BufferedReader reader = new BufferedReader(new FileReader(validationSet.getFile()))) {

			String line;
			int labelsId = 0;
			while ((line = reader.readLine()) != null) {
				String[] lineAr = line.split(INPUT_TURN_DELIMITER_PATTERN);
				Optional<INDArray> inputMatrix = getInputValueMatrix(lineAr, wordVectors, wordsPerTurn);
				inputList.add(inputMatrix.get().ravel());
				labelsList[labelsId++] = getEvalLabel(lineAr);
			}

		}

		INDArray evalInput = Nd4j.create(inputList, shape(inputSize, inputColumns));
		INDArray evalLabels = Nd4j.create(labelsList);
		Evaluation eval = new Evaluation(3);
		eval.eval(evalLabels, evalInput, model);
		LOGGER.info(eval.stats());
	}

	private static double[] getEvalLabel(String[] lineAr) {
		String labelStr = lineAr[2];
		switch (labelStr) {
		case "NULL":
			return new double[] { 1, 0, 0 };
		case "A":
			return new double[] { 0, 1, 0 };
		case "F":
			return new double[] { 0, 0, 1 };
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
		return new int[] { rows, columns };
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
						.lossFunction(LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
						.build())
			
			// Pretraining and Backprop configuration
			.pretrain(false)
			.backprop(true)
			
			.build();
			//@formatter:on
		// System.out.println(multiLayerConf.toJson());
		return new MultiLayerNetwork(multiLayerConf);
	}

	/**
	 * https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec
	 * 
	 * @param args
	 * @throws Exception
	 */

	@SuppressWarnings("unused")
	private static void trainYourOwnWord2VecModel(String[] args) throws Exception {
		String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

		LOGGER.info("Load & Vectorize Sentences....");
		// Strip white space before and after for each line
		SentenceIterator iter = new BasicLineIterator(filePath);

		// Split on white spaces in the line to get words
		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		//@formatter:off
		LOGGER.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();
		//@formatter:on

		LOGGER.info("Fitting Word2Vec model....");
		vec.fit();

	}
}