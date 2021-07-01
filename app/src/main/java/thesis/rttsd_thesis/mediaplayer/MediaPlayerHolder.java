package thesis.rttsd_thesis.mediaplayer;

import android.content.Context;
import android.media.MediaPlayer;
import android.net.Uri;

import androidx.annotation.IdRes;

import thesis.rttsd_thesis.R;

import java.util.ArrayList;
import java.util.List;

public class MediaPlayerHolder implements MediaPlayerAdapter {
    private MediaPlayer mediaPlayer;
    private Context context;

    @IdRes
    private List<Integer> soundList = new ArrayList<>();

    public MediaPlayerHolder(Context context) {
        this.context = context;
    }

    private void initializeMediaPlayer() {
        if (mediaPlayer == null) {
            mediaPlayer = new MediaPlayer();
            mediaPlayer.setOnPreparedListener(mp -> {
                mediaPlayer.start();
            });
            mediaPlayer.setOnCompletionListener(mp -> {
                playNext();
            });
        }
    }

    @Override
    public void loadMedia(@IdRes int resId) {
        initializeMediaPlayer();

        if (isPlaying() && resId != R.raw.exceeded_speed_limit){
            soundList.add(resId);
            return;
        }
        Uri mediaPath = Uri.parse("android.resource://" + context.getPackageName() + "/" + resId);

        try {
            mediaPlayer.reset();
            mediaPlayer.setDataSource(context, mediaPath);
        } catch (Exception e) {}

        try {
            mediaPlayer.prepare();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private boolean isPlaying() {
        if (mediaPlayer != null) {
             return mediaPlayer.isPlaying();
        } else return false;
    }

    private void playNext() {
        if (!soundList.isEmpty()) {
            int media = soundList.get(0);
            loadMedia(media);
            soundList.remove(0);
        }
    }

    @Override
    public void reset() {
        if (mediaPlayer != null) {
            mediaPlayer.reset();
            mediaPlayer.release();
            mediaPlayer = null;
        }
        soundList.clear();
    }

}
