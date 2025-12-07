// DJLMnistExample.java
//
// PURE JAVA CNN - No Python required!
//
// Uses DJL (Deep Java Library) with PyTorch backend.
// DJL downloads native PyTorch libraries automatically - no Python install needed.
// This runs entirely in the JVM.
//
// Note: Despite using PyTorch under the hood, DJL's PyTorch doesn't support
// Apple Silicon GPU (MPS). Training is CPU-only (~7s/epoch).
// For GPU-accelerated training, see ProcessMnistExample.java (Python subprocess).
//
// Run with:
//   mvn exec:exec@ml -Dml.class=DJLTrainer
//
// Or standalone:
//   mvn compile
//   java -cp "$(cat /tmp/cp.txt):target/classes:experiments" DJLMnistExample
//
// Dependencies already in pom.xml:
//   - ai.djl:api:0.30.0
//   - ai.djl.pytorch:pytorch-engine:0.30.0
//   - ai.djl:basicdataset:0.30.0
//   - ai.djl:model-zoo:0.30.0

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Paths;

public class DJLMnistExample {

    public static void main(String[] args) throws IOException, TranslateException {

        // Print engine info
        System.out.println("Engine: " + Engine.getInstance().getEngineName());
        System.out.println("Engine version: " + Engine.getInstance().getVersion());
        System.out.println("GPU available: " + Engine.getInstance().getGpuCount());

        int batchSize = 128;
        int epochs = 3;
        float learningRate = 0.001f;

        // ========================================================
        // Step 1: Build the CNN architecture
        // ========================================================

        SequentialBlock block = new SequentialBlock();

        // Conv block 1: 1 -> 8 filters, 3x3 kernel
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(8)
                .build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        // Conv block 2: 8 -> 16 filters, 3x3 kernel
        block.add(Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .setFilters(16)
                .build());
        block.add(Activation::relu);
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));

        // Flatten and dense layers
        block.add(Blocks.batchFlattenBlock());
        block.add(Linear.builder().setUnits(64).build());
        block.add(Activation::relu);
        block.add(Linear.builder().setUnits(10).build());  // 10 output classes

        // ========================================================
        // Step 2: Create model and configure training
        // ========================================================

        try (Model model = Model.newInstance("mnist-cnn")) {
            model.setBlock(block);

            // Training configuration
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                    .addEvaluator(new Accuracy())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.fixed(learningRate))
                            .build())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            // ========================================================
            // Step 3: Load MNIST dataset
            // ========================================================

            Mnist trainDataset = Mnist.builder()
                    .optUsage(Dataset.Usage.TRAIN)
                    .setSampling(batchSize, true)
                    .build();
            trainDataset.prepare();

            Mnist testDataset = Mnist.builder()
                    .optUsage(Dataset.Usage.TEST)
                    .setSampling(batchSize, true)
                    .build();
            testDataset.prepare();

            // ========================================================
            // Step 4: Train the model
            // ========================================================

            try (Trainer trainer = model.newTrainer(config)) {
                // Initialize with input shape: batch x channels x height x width
                trainer.initialize(new Shape(batchSize, 1, 28, 28));

                trainer.setMetrics(new Metrics());

                System.out.println("\nStarting training...");
                long startTime = System.currentTimeMillis();

                for (int epoch = 0; epoch < epochs; epoch++) {
                    long epochStart = System.currentTimeMillis();

                    // Train
                    for (var batch : trainer.iterateDataset(trainDataset)) {
                        EasyTrain.trainBatch(trainer, batch);
                        trainer.step();
                        batch.close();
                    }

                    // Validate
                    for (var batch : trainer.iterateDataset(testDataset)) {
                        EasyTrain.validateBatch(trainer, batch);
                        batch.close();
                    }

                    trainer.notifyListeners(listener -> listener.onEpoch(trainer));

                    long epochTime = System.currentTimeMillis() - epochStart;
                    System.out.println("Epoch " + (epoch + 1) + "/" + epochs + " completed in " + epochTime + "ms");
                }

                long totalTime = System.currentTimeMillis() - startTime;
                System.out.println("\nTraining complete in " + totalTime + "ms");

                // Print final metrics
                System.out.println("\nFinal training metrics:");
                System.out.println(trainer.getTrainingResult());
            }
        }
    }
}
