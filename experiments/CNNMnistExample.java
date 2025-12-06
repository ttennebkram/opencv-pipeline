// CNNMnistExample.java
//
// A small CNN on MNIST, roughly equivalent to the PyTorch version we wrote.
// Uses Deeplearning4j.
//
// Maven dependencies (pom.xml) would need something like:
//
// <dependencies>
//   <dependency>
//     <groupId>org.deeplearning4j</groupId>
//     <artifactId>deeplearning4j-core</artifactId>
//     <version>1.0.0-M2.1</version>
//   </dependency>
//   <dependency>
//     <groupId>org.nd4j</groupId>
//     <artifactId>nd4j-native-platform</artifactId>
//     <version>1.0.0-M2.1</version>
//   </dependency>
//   <dependency>
//     <groupId>org.deeplearning4j</groupId>
//     <artifactId>deeplearning4j-datasets</artifactId>
//     <version>1.0.0-M2.1</version>
//   </dependency>
// </dependencies>

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CNNMnistExample {

    public static void main(String[] args) throws Exception {

        // ========================================================
        // Step 1: "Architect writes the blueprint"
        //
        // Here we DEFINE the network architecture:
        //   - Two conv+pool blocks
        //   - One dense layer
        //   - One output layer (10 digits)
        //
        // This is just a configuration / blueprint, not trained yet.
        // ========================================================

        int height = 28;        // MNIST height
        int width = 28;         // MNIST width
        int channels = 1;       // grayscale
        int outputNum = 10;     // digits 0-9
        int batchSize = 128;
        int epochs = 3;
        double learningRate = 0.001;

        // Conv1: 1 -> 8 filters, 3x3 kernel
        // After Conv1 (no padding, stride 1):
        //   28x28 -> 26x26
        // After Pool1 (2x2):
        //   26x26 -> 13x13
        //
        // Conv2: 8 -> 16 filters, 3x3 kernel
        // After Conv2:
        //   13x13 -> 11x11
        // After Pool2 (2x2):
        //   11x11 -> 5x5
        //
        // Final conv output shape: 16 x 5 x 5
        // Flatten size: 16 * 5 * 5 = 400

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learningRate))          // optimizer: Adam
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                // ----- Conv block 1 -----
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(channels)                    // 1 input channel
                        .nOut(8)                          // 8 filters
                        .kernelSize(3, 3)                 // 3x3
                        .stride(1, 1)                     // stride 1
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)                 // 2x2 max pool
                        .stride(2, 2)
                        .build()
                )
                // ----- Conv block 2 -----
                .layer(2, new ConvolutionLayer.Builder()
                        .nOut(16)                         // 16 filters
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build()
                )
                // ----- Dense + Output -----
                .layer(4, new DenseLayer.Builder()
                        .nOut(64)                         // small dense layer
                        .activation(Activation.RELU)
                        .build()
                )
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)                  // 10 classes
                        .activation(Activation.SOFTMAX)   // probabilities over 10 digits
                        .build()
                )
                // Tell DL4J the input is images: (channels, height, width)
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();

        // ========================================================
        // Step 3: "Construction crew builds the house and brings tools"
        //
        // Now we:
        //   - Instantiate the model from the blueprint (conf)
        //   - Attach a loss function & optimizer (already in 'conf')
        //   - Add listeners to see the score during training
        //
        // This is the working (but untrained) CNN machine.
        // ========================================================

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        System.out.println("Model summary:");
        System.out.println(model.summary());   // Shows layer shapes and params

        // ========================================================
        // Step 4: Load the data (MNIST)
        // ========================================================

        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest  = new MnistDataSetIterator(batchSize, false, 12345);

        // ========================================================
        // Step 5: Train the model
        // ========================================================

        System.out.println("Starting training...");
        for (int i = 1; i <= epochs; i++) {
            System.out.println("Epoch " + i + " / " + epochs);
            model.fit(mnistTrain);
            mnistTrain.reset();
        }
        System.out.println("Training complete.");

        // ========================================================
        // Step 6: Evaluate the model
        // ========================================================

        System.out.println("Evaluating on test set...");
        org.deeplearning4j.eval.Evaluation eval = model.evaluate(mnistTest);
        System.out.println(eval.stats());

        // ========================================================
        // Step 7: Predict a single sample
        // ========================================================

        mnistTest.reset();
        if (mnistTest.hasNext()) {
            DataSet batch = mnistTest.next();
            // Take first example from the batch
            org.nd4j.linalg.api.ndarray.INDArray features = batch.getFeatures();
            org.nd4j.linalg.api.ndarray.INDArray labels   = batch.getLabels();

            // One sample: needs shape [1, 784] (batch of 1), not [784]
            org.nd4j.linalg.api.ndarray.INDArray single = features.getRow(0, true);  // true = keep leading dimension
            org.nd4j.linalg.api.ndarray.INDArray output = model.output(single);

            // output shape is [1, 10], labels row shape is [10]
            // Use argMax() without dimension for 1D arrays, or flatten first
            int predicted = org.nd4j.linalg.factory.Nd4j.argMax(output.getRow(0)).getInt(0);
            int actual    = org.nd4j.linalg.factory.Nd4j.argMax(labels.getRow(0)).getInt(0);

            System.out.println("Single-sample prediction:");
            System.out.println("  Predicted: " + predicted);
            System.out.println("  Actual:    " + actual);
            System.out.println("  Raw output: " + output);
        }
    }
}


