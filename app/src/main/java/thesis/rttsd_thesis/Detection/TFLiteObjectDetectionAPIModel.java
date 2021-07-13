/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package thesis.rttsd_thesis.Detection;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Trace;


import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API: -
 * https://github.com/tensorflow/models/tree/master/research/object_detection where you can find the
 * training code.
 *
 * <p>To use pretrained models in the API or convert to TF Lite models, please see docs for details:
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
 * -
 * https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md#running-our-model-on-android
 *
 * <p>For more information about Metadata and associated fields (eg: `labels.txt`), see <a
 * href="https://www.tensorflow.org/lite/convert/metadata#read_the_metadata_from_models">Read the
 * metadata from models</a>
 */
public class TFLiteObjectDetectionAPIModel implements Detector {
  private static final String TAG = "TFLiteObjectDetectionAPIModelWithTaskApi";

  /** Only return this many results. */
  private static final int NUM_DETECTIONS = 10;

  private final MappedByteBuffer modelBuffer;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private ObjectDetector objectDetector;

  /** Builder of the options used to config the ObjectDetector. */
  private final ObjectDetector.ObjectDetectorOptions.Builder optionsBuilder;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * <p>{@code labelFilename}, {@code inputSize}, and {@code isQuantized}, are NOT required, but to
   * keep consistency with the implementation using the TFLite Interpreter Java API. See <a
   * href="https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/android/lib_interpreter/src/main/java/org/tensorflow/lite/examples/detection/tflite/TFLiteObjectDetectionAPIModel.java">lib_interpreter</a>.
   *
   * @param modelFilename The model file path relative to the assets folder
   * @param labelFilename The label file path relative to the assets folder
   * @param inputSize The size of image input
   * @param isQuantized Boolean representing model is quantized or not
   */

  private Classifier classifier;
  private Bitmap rgbFrameBitmap = null;
  private long lastProcessingTimeMs;
  private int[] rgbBytes = null;
  private Runnable imageConverter;
  private int numThreads;

  //private static final String TFLITE = "43signs.tflite";
  //private static final String LABELS = "43signs.txt";


  public static Detector create(
          final Context context,
          final String modelFilename,
          final String labelFilename,
          final int inputSize,
          final boolean isQuantized)
          throws IOException {
    return new TFLiteObjectDetectionAPIModel(context, modelFilename);
  }
  
  private TFLiteObjectDetectionAPIModel(Context context, String modelFilename) throws IOException {
    modelBuffer = FileUtil.loadMappedFile(context, modelFilename);
    optionsBuilder = ObjectDetector.ObjectDetectorOptions.builder().setMaxResults(NUM_DETECTIONS);
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) throws IOException {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    List<Detection> results = objectDetector.detect(TensorImage.fromBitmap(bitmap));

    // Converts a list of {@link Detection} objects into a list of {@link Recognition} objects
    // to match the interface of other inference method, such as using the <a
    // href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android/lib_interpreter">TFLite
    // Java API.</a>.
    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int cnt = 0;
    for (Detection detection : results) {

      if (detection != null) {
      /*  classifier = new Classifier(null, Classifier.Device.GPU, numThreads) {
          @Override
          protected String getModelPath() {
            return TFLITE;
          }

          @Override
          protected String getLabelPath() {
            return LABELS;
          }

          @Override
          protected TensorOperator getPreprocessNormalizeOp() {
            return null;
          }

          @Override
          protected TensorOperator getPostprocessNormalizeOp() {
            return null;
          }
        }; */
/*
        Matrix matrix = new Matrix();
        matrix.postRotate(90);
        Bitmap crop = Bitmap.createBitmap(rgbFrameBitmap,
                (int) detection.getBoundingBox().left,
                (int) detection.getBoundingBox().top,
                (int) detection.getBoundingBox().width(),
                (int) detection.getBoundingBox().height(),
                matrix,
                true);

        final long startTime = SystemClock.uptimeMillis();
        //final List<Classifier.Recognition> classresults =
         //       classifier.recognizeImage(prepareImageForClassification(crop), 90);
        SpeedLimitClassifier speedLimitClassifier =null;
        try {
          speedLimitClassifier = SpeedLimitClassifier.classifier(
                  getAssets(), MODEL_FILENAME);
        } catch (IOException e) {
          e.printStackTrace();
        }
        List<ClassificationEntity> recognitions2 =
                                     speedLimitClassifier.recognizeImage(prepareImageForClassification(crop));
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        System.out.print(recognitions2.toString());
*/
      }

      recognitions.add(
          new Recognition(
              "" + cnt++, //id
              detection.getCategories().get(0).getLabel(), // title
              detection.getCategories().get(0).getScore(), //confidence
              detection.getBoundingBox())); //location
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    if (objectDetector != null) {
      objectDetector.close();
    }
  }

  @Override
  public void setNumThreads(int numThreads) {
    this.numThreads = numThreads;
    if (objectDetector != null) {
      optionsBuilder.setNumThreads(numThreads);
      recreateDetector();
    }
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    throw new UnsupportedOperationException(
        "Manipulating the hardware accelerators is not allowed in the Task"
            + " library currently. Only CPU is allowed.");
  }

  private void recreateDetector() {
    objectDetector.close();
    objectDetector = ObjectDetector.createFromBufferAndOptions(modelBuffer, optionsBuilder.build());
  }
}
