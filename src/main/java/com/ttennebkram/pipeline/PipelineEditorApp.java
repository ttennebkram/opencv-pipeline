package com.ttennebkram.pipeline;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.stage.Stage;

/**
 * Main JavaFX Application class for the Pipeline Editor.
 * This is a stub that will be expanded as the migration progresses.
 */
public class PipelineEditorApp extends Application {

    @Override
    public void start(Stage primaryStage) {
        // Temporary stub UI - will be replaced with full editor
        Label label = new Label("OpenCV Pipeline Editor\n\nJavaFX Migration in Progress...\n\nOpenCV loaded: " +
            (org.opencv.core.Core.VERSION != null ? org.opencv.core.Core.VERSION : "unknown"));
        label.setStyle("-fx-font-size: 18px; -fx-text-alignment: center;");

        StackPane root = new StackPane(label);
        root.setStyle("-fx-padding: 50;");

        Scene scene = new Scene(root, 800, 600);

        primaryStage.setTitle("OpenCV Pipeline Editor");
        primaryStage.setScene(scene);
        primaryStage.show();

        System.out.println("JavaFX Pipeline Editor started");
        System.out.println("OpenCV version: " + org.opencv.core.Core.VERSION);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
