package com.ttennebkram.pipeline.util;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
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
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.ndarray.NDList;

import java.io.*;
import java.nio.file.*;

/**
 * DJL-based trainer for MNIST CNN.
 * Pure Java implementation - no Python required.
 * Used as fallback when Python is not available.
 */
public class DJLTrainer implements AutoCloseable {

    private Model model;
    private Trainer trainer;
    private Mnist trainDataset;
    private Mnist testDataset;
    private int batchSize;
    private NDManager manager;

    public DJLTrainer() {
        manager = NDManager.newBaseManager();
    }

    /**
     * Create CNN model matching the Python architecture.
     */
    public void createModel(int numClasses) throws Exception {
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
        block.add(Linear.builder().setUnits(numClasses).build());

        model = Model.newInstance("mnist-cnn");
        model.setBlock(block);
    }

    /**
     * Load MNIST dataset.
     */
    public DatasetInfo loadMnist(int batchSize) throws Exception {
        this.batchSize = batchSize;

        trainDataset = Mnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .build();
        trainDataset.prepare();

        testDataset = Mnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .build();
        testDataset.prepare();

        return new DatasetInfo(
                (int) trainDataset.size(),
                (int) testDataset.size()
        );
    }

    /**
     * Configure training.
     */
    public void configureTraining(float learningRate) {
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optOptimizer(Optimizer.adam()
                        .optLearningRateTracker(Tracker.fixed(learningRate))
                        .build());

        trainer = model.newTrainer(config);
        trainer.initialize(new Shape(batchSize, 1, 28, 28));
    }

    /**
     * Train for one epoch.
     */
    public TrainResult trainEpoch() throws Exception {
        long startTime = System.currentTimeMillis();
        float totalLoss = 0;
        int correct = 0;
        int total = 0;
        int batchCount = 0;

        for (Batch batch : trainer.iterateDataset(trainDataset)) {
            EasyTrain.trainBatch(trainer, batch);
            trainer.step();

            // Accumulate metrics
            NDArray predictions = trainer.evaluate(batch.getData()).singletonOrThrow();
            NDArray labels = batch.getLabels().singletonOrThrow();

            // Get loss
            NDArray lossArray = Loss.softmaxCrossEntropyLoss().evaluate(new NDList(labels), new NDList(predictions));
            totalLoss += lossArray.getFloat();

            // Get accuracy
            NDArray predIndices = predictions.argMax(1);
            correct += (int) predIndices.eq(labels).sum().getLong();
            total += (int) labels.size();

            batchCount++;
            batch.close();
        }

        long elapsed = System.currentTimeMillis() - startTime;
        float avgLoss = totalLoss / batchCount;
        float accuracy = (float) correct / total;

        return new TrainResult(avgLoss, accuracy, elapsed);
    }

    /**
     * Evaluate on test set.
     */
    public EvalResult evaluate() throws Exception {
        float totalLoss = 0;
        int correct = 0;
        int total = 0;
        int batchCount = 0;

        for (Batch batch : trainer.iterateDataset(testDataset)) {
            NDArray predictions = trainer.evaluate(batch.getData()).singletonOrThrow();
            NDArray labels = batch.getLabels().singletonOrThrow();

            // Get loss
            NDArray lossArray = Loss.softmaxCrossEntropyLoss().evaluate(new NDList(labels), new NDList(predictions));
            totalLoss += lossArray.getFloat();

            // Get accuracy
            NDArray predIndices = predictions.argMax(1);
            correct += (int) predIndices.eq(labels).sum().getLong();
            total += (int) labels.size();

            batchCount++;
            batch.close();
        }

        float avgLoss = totalLoss / batchCount;
        float accuracy = (float) correct / total;

        return new EvalResult(avgLoss, accuracy);
    }

    /**
     * Export weights in format compatible with JavaCNNInference.
     */
    public void exportWeightsForJava(String path) throws Exception {
        // Get parameters from model
        var params = model.getBlock().getParameters();

        try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))) {
            // Magic number "JCNN" and version
            dos.writeInt(0x4A434E4E);
            dos.writeInt(1);

            // We need to extract weights in the right order
            // Conv1 weight [8][1][3][3], bias [8]
            // Conv2 weight [16][8][3][3], bias [16]
            // FC1 weight [64][400], bias [64]
            // FC2 weight [10][64], bias [10]

            int paramIndex = 0;
            for (var pair : params) {
                NDArray arr = pair.getValue().getArray();
                float[] data = arr.toFloatArray();

                for (float v : data) {
                    dos.writeFloat(v);
                }
                paramIndex++;
            }
        }
    }

    @Override
    public void close() {
        if (trainer != null) {
            trainer.close();
            trainer = null;
        }
        if (model != null) {
            model.close();
            model = null;
        }
        if (manager != null) {
            manager.close();
            manager = null;
        }
    }

    // ========== Result Classes ==========

    public static class DatasetInfo {
        public final int trainSize;
        public final int testSize;

        public DatasetInfo(int trainSize, int testSize) {
            this.trainSize = trainSize;
            this.testSize = testSize;
        }
    }

    public static class TrainResult {
        public final float loss;
        public final float accuracy;
        public final long timeMs;

        public TrainResult(float loss, float accuracy, long timeMs) {
            this.loss = loss;
            this.accuracy = accuracy;
            this.timeMs = timeMs;
        }

        @Override
        public String toString() {
            return String.format("Loss: %.4f, Accuracy: %.2f%%, Time: %dms", loss, accuracy * 100, timeMs);
        }
    }

    public static class EvalResult {
        public final float loss;
        public final float accuracy;

        public EvalResult(float loss, float accuracy) {
            this.loss = loss;
            this.accuracy = accuracy;
        }

        @Override
        public String toString() {
            return String.format("Loss: %.4f, Accuracy: %.2f%%", loss, accuracy * 100);
        }
    }

    // ========== Main for testing ==========

    public static void main(String[] args) throws Exception {
        System.out.println("DJLTrainer Test (Pure Java)\n");

        try (DJLTrainer trainer = new DJLTrainer()) {
            System.out.println("Creating model...");
            trainer.createModel(10);

            System.out.println("Loading MNIST...");
            DatasetInfo info = trainer.loadMnist(128);
            System.out.println("Train: " + info.trainSize + ", Test: " + info.testSize);

            System.out.println("\nConfiguring training...");
            trainer.configureTraining(0.001f);

            System.out.println("\nTraining:");
            for (int epoch = 1; epoch <= 2; epoch++) {
                TrainResult result = trainer.trainEpoch();
                System.out.println("  Epoch " + epoch + ": " + result);
            }

            System.out.println("\nEvaluating:");
            EvalResult eval = trainer.evaluate();
            System.out.println("  " + eval);

            System.out.println("\nExporting weights...");
            String path = "/tmp/djl_model.weights";
            trainer.exportWeightsForJava(path);
            System.out.println("  Saved to: " + path);
        }

        System.out.println("\nDone!");
    }
}
