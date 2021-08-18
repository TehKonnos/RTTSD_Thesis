/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package thesis.rttsd_thesis;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.Location;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.widget.SwitchCompat;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import thesis.rttsd_thesis.Detection.Classifier;
import thesis.rttsd_thesis.Detection.Classifier.Recognition;
import thesis.rttsd_thesis.Detection.YoloV5Classifier;
import thesis.rttsd_thesis.customview.OverlayView;
import thesis.rttsd_thesis.env.BorderedText;
import thesis.rttsd_thesis.env.ImageUtils;
import thesis.rttsd_thesis.env.Logger;
import thesis.rttsd_thesis.mediaplayer.MediaPlayerHolder;
import thesis.rttsd_thesis.model.entity.Data;
import thesis.rttsd_thesis.tracking.MultiBoxTracker;

import static thesis.rttsd_thesis.ImageUtils.prepareImageForClassification;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 640;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "sign_recognition.tflite";
  public static final String TF_OD_API_LABELS_FILE = "sign_recognition.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  //For Classification
  public static final float CLASSIFICATION_THRESHOLD = 0.8f;
  public static String MODEL_FILENAME = "model82Q.tflite";

  private int maximumResults = 3;
  OverlayView trackingOverlay;

  private Integer sensorOrientation;

  private YoloV5Classifier detector;
  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private Data data;
  private TextView currentSpeed;
  private SwitchCompat notification;
  private BorderedText borderedText;
  private MediaPlayerHolder mediaPlayerHolder;


  protected void onSaveInstanceState(Bundle outState) {
    super.onSaveInstanceState(outState);
  }

  protected void onRestoreInstanceState(Bundle savedInstanceState) {
    super.onRestoreInstanceState(savedInstanceState);
  }

  @SuppressLint("DefaultLocale")
  public void setupViews() {
    TextView confidence = findViewById(R.id.confidence_value);
    confidence.setText(String.format("%.2f", MINIMUM_CONFIDENCE_TF_OD_API));

      notification = findViewById(R.id.notification_switch);
      notification.setOnCheckedChangeListener((buttonView, isChecked) -> {
          if (!isChecked)
              mediaPlayerHolder.reset();
      });

    SeekBar confidenceSeekBar = findViewById(R.id.confidence_seek);
    confidenceSeekBar.setMax(100);
    confidenceSeekBar.setProgress((int) (MINIMUM_CONFIDENCE_TF_OD_API * 100));

    confidenceSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
        MINIMUM_CONFIDENCE_TF_OD_API = progress / 100.0F;
        confidence.setText(String.format("%.2f", MINIMUM_CONFIDENCE_TF_OD_API));
      }

      @Override
      public void onStartTrackingTouch(SeekBar seekBar) {
      }

      @Override
      public void onStopTrackingTouch(SeekBar seekBar) {
      }
    });
  }

    @Override
    public void onPreviewSizeChosen ( final Size size, final int rotation){
      final float textSizePx =
              TypedValue.applyDimension(
                      TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
      borderedText = new BorderedText(textSizePx);
      borderedText.setTypeface(Typeface.MONOSPACE);

      tracker = new MultiBoxTracker(this);

      int cropSize = TF_OD_API_INPUT_SIZE;
        try {
            detector =
                    YoloV5Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_API_INPUT_SIZE);
            //detector.useGpu();
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

      previewWidth = size.getWidth();
      previewHeight = size.getHeight();

      sensorOrientation = rotation - getScreenOrientation();
      LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

      LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
      rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
      croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

      frameToCropTransform =
              ImageUtils.getTransformationMatrix(
                      previewWidth, previewHeight,
                      cropSize, cropSize,
                      sensorOrientation, MAINTAIN_ASPECT);

      cropToFrameTransform = new Matrix();
      frameToCropTransform.invert(cropToFrameTransform);

      trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
      trackingOverlay.addCallback(
              new OverlayView.DrawCallback() {
                @Override
                public void drawCallback(final Canvas canvas) {
                  tracker.draw(canvas);
                  if (isDebug()) {
                    tracker.drawDebug(canvas);
                  }
                }
              });

      tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage () {
      ++timestamp;
      final long currTimestamp = timestamp;
      trackingOverlay.postInvalidate();

      // No mutex needed as this method is not reentrant.
      if (computingDetection) {
        readyForNextImage();
        return;
      }
      computingDetection = true;
      LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

      rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

      readyForNextImage();

      final Canvas canvas = new Canvas(croppedBitmap);
      canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
      // For examining the actual TF input.
      if (SAVE_PREVIEW_BITMAP) {
        ImageUtils.saveBitmap(croppedBitmap);
      }

      runInBackground(
              new Runnable() {
                @Override
                public void run() {

                  LOGGER.i("Running detection on image " + currTimestamp);
                  final long startTime = SystemClock.uptimeMillis();
                  List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);


                  cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    final Canvas canvas = new Canvas(cropCopyBitmap);
                    final Paint paint = new Paint();
                    paint.setColor(Color.RED);
                    paint.setStyle(Paint.Style.STROKE);
                    paint.setStrokeWidth(2.0f);

                  float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                  switch (MODE) {
                    case TF_OD_API:
                      minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                      break;
                  }

                  final List<Recognition> mappedRecognitions =
                          new ArrayList<>();

                  int cResults = 0;
                  for (Recognition result : results) {
                    RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                      result = classify(result);

                      cResults++;
                      if (cResults > maximumResults) break;
                      canvas.drawRect(location, paint);

                      cropToFrameTransform.mapRect(location);

                      result.setLocation(location);
                      mappedRecognitions.add(result);

                      if(getNotificationSpeed()) playSound(result.getTitle());

                    }
                  }
                  lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                  tracker.trackResults(mappedRecognitions, currTimestamp);
                  trackingOverlay.postInvalidate();

                  computingDetection = false;

                  runOnUiThread(
                          new Runnable() {
                            @Override
                            public void run() {
                              showFrameInfo(previewWidth + "x" + previewHeight);
                              showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                              showInference(lastProcessingTimeMs + "ms");
                            }
                          });
                }
              });
    }

    @SuppressLint("ResourceType")
    private void playSound(String title) {
        setNotificationSpeed(false);
        switch(title)
        {
            case "regulatory--stop":
                mediaPlayerHolder.loadMedia(R.raw.stop);
                break;
            default:
                break;
        }
    }

    //This method gets a recognised box of sign and returns the classified sign.
    private Recognition classify (Recognition result){
        Matrix matrix = new Matrix();
        matrix.postRotate(0);

        Bitmap crop = Bitmap.createBitmap(croppedBitmap,
                (int) result.getLocation().left,
                (int) result.getLocation().top,
                (int) result.getLocation().width(),
                (int) result.getLocation().height(),
                matrix,
                true);

        if (crop != null) {
            try {
                ImageView view = findViewById(R.id.signImg);
                crop = prepareImageForClassification(crop);
                view.setImageBitmap(crop);


// Method #2
                // Initialization
                ImageClassifier.ImageClassifierOptions options =
                        ImageClassifier.ImageClassifierOptions.builder().setMaxResults(1).setScoreThreshold(CLASSIFICATION_THRESHOLD).setNumThreads(4).build();

                ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(
                        getApplicationContext(), MODEL_FILENAME, options);

                // Run inference
                List<Classifications> results2 = imageClassifier.classify(
                        TensorImage.fromBitmap(crop));


                result.setTitle(results2.get(0).getCategories().get(0).getLabel());
                result.setConfidence(results2.get(0).getCategories().get(0).getScore());
            } catch (Exception e) {
              Log.e("SLClassifier error:", e.getMessage(),e);
              result.setTitle("Sign");
            }
      }
      return result;
    }


    private boolean isTimeDifferenceValid (Date date1, Date date2){
      long milliseconds = date1.getTime() - date2.getTime();
      Log.i("sign", "isTimeDifferenceValid " + ((milliseconds / (1000)) > 30));
      return (int) (milliseconds / (1000)) > 30;
    }

    private boolean isLocationDifferenceValid (Location location1, Location location2){
      if (location1 == null || location2 == null)
        return false;
      return location1.distanceTo(location2) > 50;
    }

    @Override
    protected int getLayoutId () {
      return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize () {
      return DESIRED_PREVIEW_SIZE;
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
      TF_OD_API;
    }

    @Override
    protected void setUseNNAPI ( final boolean isChecked){
      runInBackground(
              () -> {
                try {
                  detector.setUseNNAPI(isChecked);
                } catch (UnsupportedOperationException e) {
                  LOGGER.e(e, "Failed to set \"Use NNAPI\".");
                  runOnUiThread(
                          () -> {
                            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                          });
                }
              });
    }

    @Override
    protected void setNumThreads ( final int numThreads){
      runInBackground(() -> detector.setNumThreads(numThreads));
    }
    public void setMaximumResults(int maximumResults) {
        this.maximumResults = maximumResults;
    }
}