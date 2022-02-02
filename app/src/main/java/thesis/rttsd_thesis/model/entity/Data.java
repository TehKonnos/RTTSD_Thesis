package thesis.rttsd_thesis.model.entity;


import android.location.Location;

public class Data {
    private long time;
    private double distanceM;
    private double currentSpeed;
    private Location location;

    public Data() {
        distanceM = 0;
        currentSpeed = 0;
    }

    public void setDistance(double distance) {
        distanceM = distance;
    }

    public double getDistance() {
        return distanceM;
    }

    public void setCurrentSpeed(double currentSpeed) {
        this.currentSpeed = currentSpeed;
    }

    public double getCurrentSpeed() {
        return currentSpeed;
    }

    public long getTime() {
        return time;
    }

    public void setTime(long time) {
        this.time = time;
    }

    public void setLocation(Location location) {
        this.location = location;
    }

    public Location getLocation() {
        return location;
    }
}

