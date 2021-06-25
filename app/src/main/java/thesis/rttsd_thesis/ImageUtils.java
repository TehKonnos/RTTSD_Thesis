package thesis.rttsd_thesis;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;


public class ImageUtils {

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        Paint paint = new Paint();
        Bitmap finalBitmap = Bitmap.createScaledBitmap(
                bitmap,
                224,
                224,
                false);
        Canvas canvas = new Canvas(finalBitmap);
        canvas.drawBitmap(finalBitmap, 0, 0, paint);
        return finalBitmap;
    }

}
