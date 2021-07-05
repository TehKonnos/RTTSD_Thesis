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
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.Location;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;


import androidx.annotation.RequiresApi;
import androidx.appcompat.widget.SwitchCompat;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.model.Model;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import thesis.rttsd_thesis.Detection.Detector;
import thesis.rttsd_thesis.Detection.TFLiteObjectDetectionAPIModel;
import thesis.rttsd_thesis.adapter.SignAdapter;
import thesis.rttsd_thesis.customview.OverlayView;
import thesis.rttsd_thesis.env.BorderedText;
import thesis.rttsd_thesis.env.ImageUtils;
import thesis.rttsd_thesis.env.Logger;
import thesis.rttsd_thesis.mediaplayer.MediaPlayerHolder;
import thesis.rttsd_thesis.ml.SignRecogn4;
import thesis.rttsd_thesis.ml.Xronis;
import thesis.rttsd_thesis.model.entity.ClassificationEntity;
import thesis.rttsd_thesis.model.entity.Data;
import thesis.rttsd_thesis.model.entity.SignEntity;
import thesis.rttsd_thesis.tracking.MultiBoxTracker;

//import static thesis.rttsd_thesis.DetectorActivity.DetectorMode.TF_OD_API;
import static thesis.rttsd_thesis.ImageUtils.prepareImageForClassification;
import static thesis.rttsd_thesis.SpeedLimitClassifier.MODEL_FILENAME;
import static thesis.rttsd_thesis.SpeedLimitClassifier.loadModelFile;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 1024;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "sign_recogn_5.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/sign_recogn.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(1024, 1024);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;

  private Integer sensorOrientation;

  private Detector detector;

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
  private SignAdapter adapter;
  private final String SIGN_LIST = "sign_list";
  private Boolean speedNotification = true;


 /* protected void onSaveInstanceState(@NonNull Bundle outState) {
    super.onSaveInstanceState(outState);
    outState.putString(SIGN_LIST, new Gson().toJson(adapter.getSigns()));
  }

  protected void onRestoreInstanceState(Bundle savedInstanceState) {
    super.onRestoreInstanceState(savedInstanceState);
    String json = savedInstanceState.getString(SIGN_LIST);
    ArrayList<SignEntity> items = null;
    try {
      items = (new Gson()).fromJson(json, new TypeToken<ArrayList<SignEntity>>() {
      }.getType());
    } catch (Exception ignored) {
      items = new ArrayList<>();
    }
    adapter.setSigns(items);

  } */

  @SuppressLint({"ResourceType", "DefaultLocale"})
  private void setCurrentSpeed(Data data) {
    this.data = data;

    if (data.getLocation().hasSpeed()) {
      double speed = data.getLocation().getSpeed() * 3.6;
      if (speed > 50 && notification.isChecked() && speedNotification) {
        speedNotification = false;
        //mediaPlayerHolder.loadMedia(R.raw.exceeded_speed_limit);
      }
      currentSpeed.setText(String.format("%.0f", speed));
    }
  }

  @SuppressLint("DefaultLocale")
  private void setupView() {
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
                TFLiteObjectDetectionAPIModel.create(
                        this,
                        TF_OD_API_MODEL_FILE,
                        TF_OD_API_LABELS_FILE,
                        TF_OD_API_INPUT_SIZE,
                        TF_OD_API_IS_QUANTIZED);
        cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final IOException e) {
        e.printStackTrace();
        LOGGER.e(e, "Exception initializing Detector!");
        Toast toast =
                Toast.makeText(
                        getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
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
                @RequiresApi(api = Build.VERSION_CODES.O)
                @Override
                public void run() {
                  LOGGER.i("Running detection on image " + currTimestamp);
                  final long startTime = SystemClock.uptimeMillis();
                  List<Detector.Recognition> results = null;
 //Android studio given code

  /*                try {
                    Xronis model = Xronis.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1,3,1024, 1024}, DataType.FLOAT32);
                    ByteBuffer byteBuffer=  convertBitmapToByteBuffer(croppedBitmap);
                    Log.e("Gamieste",byteBuffer+"");
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    Xronis.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                    TensorBuffer outputFeature1 = outputs.getOutputFeature1AsTensorBuffer();
                    TensorBuffer outputFeature2 = outputs.getOutputFeature2AsTensorBuffer();
                    TensorBuffer outputFeature3 = outputs.getOutputFeature3AsTensorBuffer();
                    Log.e("Gamieste",outputFeature0.toString());
                    // Releases model resources if no longer used.
                    model.close();
                  } catch (IOException e) {
                    // TODO Handle the exception
                  } */
/*
                  try {
                    Model.Options option=null; //TODO Edw pairnei kapoia options kai mallon prepei na setaroume oti einai float point or not
                    SignRecogn4 model = SignRecogn4.newInstance(getApplicationContext());

                    // Creates inputs for reference.
                    TensorImage image = TensorImage.fromBitmap(croppedBitmap);

                    // Runs model inference and gets result.
                    SignRecogn4.Outputs outputs = model.process(image);
                    SignRecogn4.DetectionResult detectionResult = outputs.getDetectionResultList().get(0);

                    // Gets result from DetectionResult.
                    RectF location = detectionResult.getLocationAsRectF();
                    String category = detectionResult.getCategoryAsString();
                    float score = detectionResult.getScoreAsFloat();
                    Log.e("ress",location.toString()+" "+category+" "+score);

                    // Releases model resources if no longer used.
                    model.close();
                  } catch (IOException e) {
                    // TODO Handle the exception
                  }
*/

                  try {
                    results = detector.recognizeImage(croppedBitmap);
                  } catch (IOException e) {
                    e.printStackTrace();
                  }
                  lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                  cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                  final Canvas canvas = new Canvas(cropCopyBitmap);
                  final Paint paint = new Paint();
                  paint.setColor(Color.RED);
                  paint.setStyle(Style.STROKE);
                  paint.setStrokeWidth(2.0f);

                  float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                  switch (MODE) {
                    case TF_OD_API:
                      minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                      break;
                  }

                  final List<Detector.Recognition> mappedRecognitions =
                          new ArrayList<Detector.Recognition>();

                  for (Detector.Recognition result : results) {
                    RectF location = result.getLocation();
                    if (location != null && result.getConfidence() >= minimumConfidence) {
                      //result = classify(result);
                      //location = result.getLocation();
                      //For testing:
                      Detector.Recognition result2 = classify(result);
                      canvas.drawRect(location, paint);

                      cropToFrameTransform.mapRect(location);

                      result.setLocation(location);
                      mappedRecognitions.add(result);

                      //runOnUiThread(() -> updateSignList(result, croppedBitmap));
                    }
                  }

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
  private static final int IMAGE_MEAN = 0;
  private static final float IMAGE_STD = 255.0f;

  static ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*1024*1024*3);
    byteBuffer.order(ByteOrder.nativeOrder());
    int[] intValues = new int[1024 * 1024];
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    int pixel = 0;
    for (int i = 0; i < 1024; ++i) {
      for (int j = 0; j < 1024; ++j) {
        final int val = intValues[pixel++];
        byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
      }
    }
    return byteBuffer;}

//This method gets a recognised box of sign and returns the classified sign.
    @RequiresApi(api = Build.VERSION_CODES.O)
    private Detector.Recognition classify (Detector.Recognition result){
      Matrix matrix = new Matrix();
      matrix.postRotate(90);

      Bitmap crop = null;
      try {
        crop = Bitmap.createBitmap(rgbFrameBitmap,
                (int) result.getLocation().left,
                (int) result.getLocation().top,
                (int) result.getLocation().width(),
                (int) result.getLocation().height(),
                matrix,
                true);
      } catch (Exception e) {
        Log.e("Debugging", e.getMessage());
      }
      if (crop != null) {
        try {
          SpeedLimitClassifier speedLimitClassifier = null;

          speedLimitClassifier = SpeedLimitClassifier.classifier(
                  getAssets(), MODEL_FILENAME);

          List<ClassificationEntity> recognitions2 = null;
          if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            recognitions2 = speedLimitClassifier.recognizeImage(prepareImageForClassification(crop), getAssets());
          }

          //List<ClassificationEntity> recognitions2 =
                  //speedLimitClassifier.recognizeImage(prepareImageForClassification(crop), getAssets());

          Log.e("Classifier", recognitions2.toString());
          Toast.makeText(getApplicationContext(),recognitions2.toString(),Toast.LENGTH_LONG).show();
        } catch (Exception e) {
          Log.e("SpeedLimitClassifier", e.toString());
        }
      }


      return null;
    }


    private boolean isRemoveValid (SignEntity sign1, SignEntity sign2){
      return isTimeDifferenceValid(sign1.getDate(), sign2.getDate())
              || isLocationDifferenceValid(sign1.getLocation(), sign2.getLocation());
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
}