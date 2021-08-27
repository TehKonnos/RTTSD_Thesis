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

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.location.GpsStatus;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Trace;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.WindowManager;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;
import androidx.core.content.ContextCompat;

import com.google.android.material.bottomsheet.BottomSheetBehavior;

import java.nio.ByteBuffer;
import java.util.concurrent.TimeUnit;

import io.reactivex.Observable;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.CompositeDisposable;
import thesis.rttsd_thesis.env.ImageUtils;
import thesis.rttsd_thesis.env.Logger;
import thesis.rttsd_thesis.mediaplayer.MediaPlayerHolder;
import thesis.rttsd_thesis.model.bus.MessageEventBus;
import thesis.rttsd_thesis.model.bus.model.EventGpsDisabled;
import thesis.rttsd_thesis.model.bus.model.EventUpdateLocation;
import thesis.rttsd_thesis.model.entity.Data;

import static android.Manifest.permission.ACCESS_FINE_LOCATION;

public abstract class CameraActivity extends AppCompatActivity
        implements OnImageAvailableListener,
        Camera.PreviewCallback,
        CompoundButton.OnCheckedChangeListener,
        View.OnClickListener {
  private static final Logger LOGGER = new Logger();
  private static final int PERMISSIONS_REQUEST = 1;
  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  protected int previewWidth = 0;
  protected int previewHeight = 0;
  private Handler handler;
  private HandlerThread handlerThread;
  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private final byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;
  private Runnable postInferenceCallback;
  private Runnable imageConverter;

  private LinearLayout gestureLayout;
  private BottomSheetBehavior<LinearLayout> sheetBehavior;

  protected TextView frameValueTextView, cropValueTextView, inferenceTimeTextView;
  protected ImageView bottomSheetArrowImageView;
  private TextView threadsTextView,signsTextView;

  private static Boolean notificationSpeed = true;
  private SwitchCompat notification;

  private LocationManager mLocationManager;
  private int speedLimit = 0;

  private CompositeDisposable compositeDisposable;
  Data data;
  protected MediaPlayerHolder mediaPlayerHolder;

  @SuppressLint("CheckResult")
  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

    setContentView(R.layout.tfe_od_activity_camera);

    Observable.interval(5L, TimeUnit.SECONDS)
            .timeInterval()
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe(v -> notificationSpeed = true);

    setCallBack();
    setupViews();

    if (hasPermission()) {
      setupLocation();
      setFragment();
    } else {
      requestPermission();
    }

    threadsTextView = findViewById(R.id.threads);
    ImageView plusImageView = findViewById(R.id.plus);
    ImageView minusImageView = findViewById(R.id.minus);
    signsTextView = findViewById(R.id.signs);
    ImageView plus2ImageView = findViewById(R.id.plus2);
    ImageView minus2ImageView = findViewById(R.id.minus2);
    LinearLayout bottomSheetLayout = findViewById(R.id.bottom_sheet_layout);
    gestureLayout = findViewById(R.id.gesture_layout);
    sheetBehavior = BottomSheetBehavior.from(bottomSheetLayout);
    bottomSheetArrowImageView = findViewById(R.id.bottom_sheet_arrow);
    notification = findViewById(R.id.notification_switch);

    mediaPlayerHolder = new MediaPlayerHolder(getApplicationContext());

    ViewTreeObserver vto = gestureLayout.getViewTreeObserver();
    vto.addOnGlobalLayoutListener(
            new ViewTreeObserver.OnGlobalLayoutListener() {
              @Override
              public void onGlobalLayout() {
                gestureLayout.getViewTreeObserver().removeOnGlobalLayoutListener(this);
                //int width = bottomSheetLayout.getMeasuredWidth();
                int height = gestureLayout.getMeasuredHeight();

                sheetBehavior.setPeekHeight(height);
              }
            });
    sheetBehavior.setHideable(false);

    sheetBehavior.setBottomSheetCallback(
            new BottomSheetBehavior.BottomSheetCallback() {
              @Override
              public void onStateChanged(@NonNull View bottomSheet, int newState) {
                switch (newState) {
                  case BottomSheetBehavior.STATE_HIDDEN:
                  case BottomSheetBehavior.STATE_DRAGGING:
                  case BottomSheetBehavior.STATE_HALF_EXPANDED:
                    break;
                  case BottomSheetBehavior.STATE_EXPANDED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_down);
                  }
                  break;
                  case BottomSheetBehavior.STATE_COLLAPSED: {
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                  }
                  break;
                  case BottomSheetBehavior.STATE_SETTLING:
                    bottomSheetArrowImageView.setImageResource(R.drawable.icn_chevron_up);
                    break;
                }
              }

              @Override
              public void onSlide(@NonNull View bottomSheet, float slideOffset) {
              }
            });

    frameValueTextView = findViewById(R.id.frame_info);
    cropValueTextView = findViewById(R.id.crop_info);
    inferenceTimeTextView = findViewById(R.id.inference_info);


    plusImageView.setOnClickListener(this);
    minusImageView.setOnClickListener(this);

    signsTextView.setOnClickListener(this);
    plus2ImageView.setOnClickListener(this);
    minus2ImageView.setOnClickListener(this);

  }

  protected abstract void setupViews();

  private void setCallBack() {
    compositeDisposable = new CompositeDisposable();
    compositeDisposable.add(MessageEventBus.INSTANCE
            .toObservable()
            .observeOn(AndroidSchedulers.mainThread())
            .subscribe(eventModel -> {

              if (eventModel instanceof EventUpdateLocation) {
                refresh(((EventUpdateLocation) eventModel).getData());
              }
              if (eventModel instanceof EventGpsDisabled) {
                showGpsDisabledDialog();
              }
            }));
  }

  private void showGpsDisabledDialog() {
    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    builder.setTitle(getString(R.string.gps_disabled))
            .setMessage(getString(R.string.please_enable_gps))
            .setNegativeButton(android.R.string.cancel, (dialog, id) -> {
              dialog.cancel();
            })
            .setPositiveButton(android.R.string.ok, (dialog, id) -> startActivity(new Intent("android.settings.LOCATION_SOURCE_SETTINGS")));
    AlertDialog dialog = builder.create();
    dialog.setCancelable(true);
    dialog.setOnShowListener(arg -> {
              dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                      .setTextColor(ContextCompat.getColor(this, R.color.cod_gray));
              dialog.getButton(AlertDialog.BUTTON_NEGATIVE)
                      .setTextColor(ContextCompat.getColor(this, R.color.cod_gray));
            }
    );
    dialog.show();
  }

  @SuppressLint({"ResourceType", "DefaultLocale", "SetTextI18n"})
  private void refresh(Data data) {
    this.data = data;
    TextView currentSpeed = findViewById(R.id.currentSpeed);
    if (data.getLocation().hasSpeed()) {
      double speed = data.getLocation().getSpeed() * 3.6;

      if (speed > this.speedLimit && notification.isChecked() && getNotificationSpeed()) {
        setNotificationSpeed(false);
        mediaPlayerHolder.loadMedia(R.raw.speed_limit_exceeded);
      }
      currentSpeed.setText("Τρέχων ταχύτητα: "+ (int) speed+" χλμ");
    }
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }


  /**
   * Callback for android.hardware.Camera API
   */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
            () -> ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);

    postInferenceCallback =
            () -> {
              camera.addCallbackBuffer(bytes);
              isProcessingFrame = false;
            };
    processImage();
  }

  /**
   * Callback for Camera2 API
   */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
              () -> ImageUtils.convertYUV420ToARGB8888(
                      yuvBytes[0],
                      yuvBytes[1],
                      yuvBytes[2],
                      previewWidth,
                      previewHeight,
                      yRowStride,
                      uvRowStride,
                      uvPixelStride,
                      rgbBytes);

      postInferenceCallback =
              () -> {
                image.close();
                isProcessingFrame = false;
              };

      processImage();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }

  @Override
  public synchronized void onPause() {
    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (final InterruptedException e) {
      LOGGER.e(e, "Exception!");
    }
    super.onPause();
  }

  @Override
  public synchronized void onStop() {
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    super.onDestroy();
    if (compositeDisposable != null) {
      compositeDisposable.dispose();
      compositeDisposable = null;
    }
  }

  protected synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }

  @Override
  public void onRequestPermissionsResult(
          final int requestCode, @NonNull final String[] permissions, final int[] grantResults) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    if (requestCode == PERMISSIONS_REQUEST) {
      if (allPermissionsGranted(grantResults)) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private static boolean allPermissionsGranted(final int[] grantResults) {
    for (int result : grantResults) {
      if (result != PackageManager.PERMISSION_GRANTED) {
        return false;
      }
    }
    return true;
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED && checkSelfPermission(ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }


  protected void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (!shouldShowRequestPermissionRationale(PERMISSION_CAMERA)) {
        shouldShowRequestPermissionRationale(ACCESS_FINE_LOCATION);
      }
      requestPermissions(new String[]{PERMISSION_CAMERA, ACCESS_FINE_LOCATION}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(CameraCharacteristics characteristics) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return false;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return android.hardware.camera2.CameraMetadata.INFO_SUPPORTED_HARDWARE_LEVEL_FULL <= deviceLevel;
  }

  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
                characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API =
                (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
                        || isHardwareLevelSupported(
                        characteristics);
        LOGGER.i("Camera API lv2?: %s", useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();
    Fragment fragment;
    if (useCamera2API) {
      CameraConnectionFragment camera2Fragment =
              CameraConnectionFragment.newInstance(
                      (size, rotation) -> {
                        previewHeight = size.getHeight();
                        previewWidth = size.getWidth();
                        CameraActivity.this.onPreviewSizeChosen(size, rotation);
                      },
                      this,
                      getLayoutId(),
                      getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
              new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() { return false; }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @Override
  public void onClick(View v) {
    if (v.getId() == R.id.plus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads >= 9) return;
      numThreads++;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }else if (v.getId() == R.id.minus) {
      String threads = threadsTextView.getText().toString().trim();
      int numThreads = Integer.parseInt(threads);
      if (numThreads == 1) return;
      numThreads--;
      threadsTextView.setText(String.valueOf(numThreads));
      setNumThreads(numThreads);
    }else if(v.getId() == R.id.plus2){
      String signs = signsTextView.getText().toString().trim();
      int numSigns = Integer.parseInt(signs);
      if (numSigns >= 9) return;
      numSigns++;
      signsTextView.setText(String.valueOf(numSigns));
      setMaximumResults(numSigns);
    }else if(v.getId() == R.id.minus2){
      String signs = signsTextView.getText().toString().trim();
      int numSigns = Integer.parseInt(signs);
      if (numSigns == 1) return;
      numSigns--;
      signsTextView.setText(String.valueOf(numSigns));
      setMaximumResults(numSigns);
    }
  }

  protected abstract void setMaximumResults(int numSigns);

  protected void showFrameInfo(String frameInfo) {
    frameValueTextView.setText(frameInfo);
  }

  protected void showCropInfo(String cropInfo) {
    cropValueTextView.setText(cropInfo);
  }

  protected void showInference(String inferenceTime) {
    inferenceTimeTextView.setText(inferenceTime);
  }

  protected abstract void processImage();

  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);

  protected abstract int getLayoutId();

  protected abstract Size getDesiredPreviewFrameSize();

  protected abstract void setNumThreads(int numThreads);

  @SuppressLint("MissingPermission")
  private void setupLocation() {
    mLocationManager = (LocationManager) this.getSystemService(Context.LOCATION_SERVICE);

    mLocationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 500, 0, locationListener);

    if (!mLocationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
      MessageEventBus.INSTANCE.send(new EventGpsDisabled());
    }
  }

  private final LocationListener locationListener = new LocationListener() {

    @Override
    public void onLocationChanged(Location location) {
      data = new Data();
      data.setLocation(location);

      if (location.hasSpeed()) {
        data.setCurrentSpeed(location.getSpeed() * 3.6);
        if (location.getSpeed() == 0)
          new isStopped().execute();
      }
      MessageEventBus.INSTANCE.send(new EventUpdateLocation(data));
    }


    @Override
    public void onProviderEnabled(String s) {
    }

    @Override
    public void onProviderDisabled(String s) {
    }
  };

  @SuppressLint("StaticFieldLeak")
  private class isStopped extends AsyncTask<Void, Integer, String> {
    int timer = 0;

    @Override
    protected String doInBackground(Void... unused) {
      try {
        while (data.getCurrentSpeed() == 0) {
          Thread.sleep(1000);
          timer++;
        }
      } catch (InterruptedException interruptedException) {
        return ("Sleep operation failed");
      }
      return ("Return object when task is finished");
    }

    @Override
    protected void onPostExecute(String message) {
      data.setTimeStopped(timer);
    }

  }

  public static Boolean getNotificationSpeed() {
    return notificationSpeed;
  }

  public void setNotificationSpeed(Boolean notificationSpeed) {
    CameraActivity.notificationSpeed = notificationSpeed;
  }

  public void setSpeedLimit(int speedLimit) {
    this.speedLimit = speedLimit;
  }

}
