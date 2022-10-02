package com.neilyxin.blinder_android;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor mSensorGyroscope;
    private TextView txtMain;
    private TextView txtAcc;
    private TextView txtGyro;
    private final int SAMPLE_RATE_50_HZ = 20000; // 50 Hz
    private final int WINDOW_SIZE = 128;
    private final int STRIDE_LENGTH = 10;
    private final int BUFFER_SIZE = 500;
    private int mySamplingRate;

    private CircularArrayList<Float> accXList;
    private CircularArrayList<Float> accYList;
    private CircularArrayList<Float> accZList;
    private CircularArrayList<Float> gyroXList;
    private CircularArrayList<Float> gyroYList;
    private CircularArrayList<Float> gyroZList;

    private Button btn_start_mobi_g;
    private Button btn_start_mobi_w;
    private Button btn_start_motion_g;
    private Button btn_stop;
    private ScrollView scroll_view;
    private int accBatchCounter;
    private int gyroBatchCounter;
    private boolean isRecording = false;
    private boolean isUpdateText = true;
    private Thread myThread;
    private Module encoder;
    private Module decoder;
    private Module eval_mobi_act;
    private Module eval_mobi_gen;
    private Module eval_mobi_weight;
    private Module eval_motion_act;
    private Module eval_motion_gen;
    //    private Module aux;
    private ArrayList<Long> timeEncList;
    private ArrayList<Long> timeDecList;
    private ArrayList<Long> timeTotalList;
    private ArrayList<Long> timePrivateList;
    private ArrayList<Long> timePublicList;
    private boolean isClearing;
    private volatile boolean lock;
    private int privateAttrLen;
    private Random rn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        txtMain = findViewById(R.id.txt_main);
        txtAcc = findViewById(R.id.txt_acc);
        txtGyro = findViewById(R.id.txt_gyro);
        btn_start_mobi_g = findViewById(R.id.btn_start_mobi_gen);
        btn_start_mobi_w = findViewById(R.id.btn_start_mobi_w);
        btn_start_motion_g = findViewById(R.id.btn_start_motion_g);
        btn_stop = findViewById(R.id.btn_stop);
        scroll_view = findViewById(R.id.scroll_view);

        mySamplingRate = SAMPLE_RATE_50_HZ;
        rn = new Random();

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        mSensorGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        mSensorManager.registerListener(this, mAccelerometer, mySamplingRate);
        mSensorManager.registerListener(this, mSensorGyroscope, mySamplingRate);

        if (mSensorGyroscope == null) {
            Log.d(this.getClass().getSimpleName(), "No Gyroscope!");
        }

        btn_start_mobi_g.setOnClickListener(view -> {
            initialize();
            privateAttrLen = 2;
            encoder = Module.load(assetFilePath(this, "mobi_g_enc.ptl"));
            decoder = Module.load(assetFilePath(this, "mobi_g_dec.ptl"));
            eval_mobi_act = Module.load(assetFilePath(this, "eval_mobi_act.ptl"));
            eval_mobi_gen = Module.load(assetFilePath(this, "eval_mobi_gen.ptl"));
            startRecord(768, 31, encoder, decoder, eval_mobi_act, eval_mobi_gen);
        });

        btn_start_mobi_w.setOnClickListener(view -> {
            initialize();
            privateAttrLen = 3;
            encoder = Module.load(assetFilePath(this, "mobi_w_enc.ptl"));
            decoder = Module.load(assetFilePath(this, "mobi_w_dec.ptl"));
            eval_mobi_act = Module.load(assetFilePath(this, "eval_mobi_act.ptl"));
            eval_mobi_weight = Module.load(assetFilePath(this, "eval_mobi_weight.ptl"));
            startRecord(768, 32, encoder, decoder, eval_mobi_act, eval_mobi_weight);
        });

        btn_start_motion_g.setOnClickListener(view -> {
            initialize();
            privateAttrLen = 2;
            encoder = Module.load(assetFilePath(this, "motion_enc.ptl"));
            decoder = Module.load(assetFilePath(this, "motion_dec.ptl"));
            eval_motion_act = Module.load(assetFilePath(this, "eval_motion_act.ptl"));
            eval_motion_gen = Module.load(assetFilePath(this, "eval_motion_gen.ptl"));
            isRecording = true;
            startRecord(256, 31, encoder, decoder, eval_motion_act, eval_motion_gen);
        });

        btn_stop.setOnClickListener(view -> {
            stopRecord();
        });

        // Load module
        eval_mobi_act = Module.load(assetFilePath(this, "eval_mobi_act.ptl"));
        eval_mobi_gen = Module.load(assetFilePath(this, "eval_mobi_gen.ptl"));
        eval_mobi_weight = Module.load(assetFilePath(this, "eval_mobi_weight.ptl"));
        eval_motion_act = Module.load(assetFilePath(this, "eval_motion_act.ptl"));
        eval_motion_gen = Module.load(assetFilePath(this, "eval_motion_gen.ptl"));
    }

    public void initialize() {
        txtMain.setText("");
        isClearing = false;
        accBatchCounter = 0;
        gyroBatchCounter = 0;

        accXList = new CircularArrayList<>(BUFFER_SIZE);
        accYList = new CircularArrayList<>(BUFFER_SIZE);
        accZList = new CircularArrayList<>(BUFFER_SIZE);
        gyroXList = new CircularArrayList<>(BUFFER_SIZE);
        gyroYList = new CircularArrayList<>(BUFFER_SIZE);
        gyroZList = new CircularArrayList<>(BUFFER_SIZE);

        timeEncList = new ArrayList<>();
        timeDecList = new ArrayList<>();
        timeTotalList = new ArrayList<>();
        timePrivateList = new ArrayList<>();
        timePublicList = new ArrayList<>();
    }

    public void startRecord(int inputSize, int latentSize, Module enc, Module dec, Module eval_public, Module eval_private) {
        // your handler code here
        isRecording = true;

        try {
            // Prepare input tensor
            isUpdateText = false;
            long[] input_enc_shape = new long[]{1, inputSize};
            long[] input_dec_shape = new long[]{1, latentSize};
//            long[] input_aux_shape = new long[] {1, 25};

            myThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    // Run whatever background code you want here.
                    float[] windowAccX;
                    float[] windowAccY;
                    float[] windowAccZ;
                    float[] windowGyroX;
                    float[] windowGyroY;
                    float[] windowGyroZ;
                    while (true) {
//                    while (accBatchCounter <= 1000) {
                        if (!isRecording) {
                            break;
                        } else if (isRecording && hasNext() && !isClearing) {
                            while (lock) {
//                            Log.d("Lock: ", "Read waiting");
                            }
                            // Read lock
                            lock = true;
                            Long timeBegin = System.currentTimeMillis();

                            windowAccX = getWindowedData(accXList);
                            windowAccY = getWindowedData(accYList);
                            windowAccZ = getWindowedData(accZList);
                            accBatchCounter++;

                            windowGyroX = getWindowedData(gyroXList);
                            windowGyroY = getWindowedData(gyroYList);
                            windowGyroZ = getWindowedData(gyroZList);
                            gyroBatchCounter++;

                            // Release read lock
                            lock = false;

                            float[] sensorCat = new float[inputSize];

                            if (inputSize == 768) {
                                for (int i = 0; i < WINDOW_SIZE; i++) {
                                    sensorCat[i] = windowAccX[i];
                                    sensorCat[WINDOW_SIZE + i] = windowAccY[i];
                                    sensorCat[WINDOW_SIZE * 2 + i] = windowAccZ[i];
                                    sensorCat[WINDOW_SIZE * 3 + i] = windowGyroX[i];
                                    sensorCat[WINDOW_SIZE * 4 + i] = windowGyroY[i];
                                    sensorCat[WINDOW_SIZE * 5 + i] = windowGyroZ[i];
                                }
                            } else if (inputSize == 256) {
                                for (int i = 0; i < WINDOW_SIZE; i++) {
                                    sensorCat[i] = (float) ((Math.pow(windowAccX[i], 2) + Math.pow(windowAccY[i], 2) + Math.pow(windowAccZ[i], 2)) / 3);
                                    sensorCat[WINDOW_SIZE + i] = (float) ((Math.pow(windowGyroX[i], 2) + Math.pow(windowGyroY[i], 2) + Math.pow(windowGyroZ[i], 2)) / 3);
                                }
                            }

                            Tensor enc_input_tensor = Tensor.fromBlob(sensorCat, input_enc_shape);
//                        Long timePrivateBegin = System.currentTimeMillis();
//                        // Get pred gen
//                        Tensor pred_gen = eval_private.forward(IValue.from(enc_input_tensor)).toTensor();
//
//                        float[] private_attr = pred_gen.getDataAsFloatArray();
//                        float maxScore = -Float.MAX_VALUE;
//                        int maxScoreIdx = -1;
//                        for (int i = 0; i < private_attr.length; i++) {
//                            if (private_attr[i] > maxScore) {
//                                maxScore = private_attr[i];
//                                maxScoreIdx = i;
//                            }
//                        }
                            int random_private = rn.nextInt(privateAttrLen);
                            float[] private_attr = new float[privateAttrLen];
                            for (int i = 0; i < private_attr.length; i++) {
                                private_attr[i] = i==random_private ? 1 : 0;
//                            if (i == maxScoreIdx) {
//                                private_attr[i] = 1;
//                            } else {
//                                private_attr[i] = 0;
//                            }
                            }
//                        Long timePrivateEnd = System.currentTimeMillis();
//                        Long timePrivate = timePrivateEnd - timePrivateBegin;

                            // Get predicted public attribute
                            Tensor pred_act = eval_public.forward(IValue.from(enc_input_tensor)).toTensor();
                            float[] public_attr = pred_act.getDataAsFloatArray();
                            float maxScore = -Float.MAX_VALUE;
                            int maxScoreIdx = -1;
                            for (int i = 0; i < public_attr.length; i++) {
                                if (public_attr[i] > maxScore) {
                                    maxScore = public_attr[i];
                                    maxScoreIdx = i;
                                }
                            }
                            // One-hot public attribute encoding
                            for (int i = 0; i < public_attr.length; i++) {
                                if (i == maxScoreIdx) {
                                    public_attr[i] = 1;
                                } else {
                                    public_attr[i] = 0;
                                }
                            }

                            Long timePublicEnd = System.currentTimeMillis();
                            Long timePublic = timePublicEnd - timeBegin;

                            // Run Encoder inference
                            IValue[] enc_outputs = enc.forward(IValue.from(enc_input_tensor)).toTuple();
                            Tensor z = enc_outputs[0].toTensor();

                            Long timeEncFinish = System.currentTimeMillis();
                            Long timeEnc = timeEncFinish - timePublicEnd;
//                            Log.d("TENSOR OUT:", enc_outputs[0].toTensor().toString());

                            // Concatenate Z, public attribute, private attribute
                            float[] zArr = z.getDataAsFloatArray();
                            float[] catZ = new float[zArr.length + private_attr.length + public_attr.length];

                            System.arraycopy(zArr, 0, catZ, 0, zArr.length);
                            System.arraycopy(private_attr, 0, catZ, zArr.length, private_attr.length);
                            System.arraycopy(public_attr, 0, catZ, private_attr.length + zArr.length, public_attr.length);

                            // Run Decoder inference
                            Tensor dec_input_tensor = Tensor.fromBlob(catZ, input_dec_shape);
                            Tensor dec_output_tensor = dec.forward(IValue.from(dec_input_tensor)).toTensor();
                            float[] reconstructedData = dec_output_tensor.getDataAsFloatArray();

                            Long timeDecFinish = System.currentTimeMillis();
                            Long timeDec = timeDecFinish - timeEncFinish;
                            Long timeTotal = timeDecFinish - timeBegin;
//                            timePrivateList.add(timePrivate);
                            timePublicList.add(timePublic);
                            timeEncList.add(timeEnc);
                            timeDecList.add(timeDec);
                            timeTotalList.add(timeTotal);

                            // Update UI
                            runOnUiThread(() -> {
                                String txt = txtMain.getText().toString();
                                //  " Priv: " + timePrivate + " " +
                                txtMain.setText(String.format("%s\nBatch %d:  Total: %d ms Pub: %d  Enc: %d  Dec: %d", txt,
                                        accBatchCounter, timeTotal, timePublic, timeEnc, timeDec));
                                scroll_view.fullScroll(View.FOCUS_DOWN);
                            });
                        }
//                    final Runtime runtime = Runtime.getRuntime();
//                    final long usedMemInMB=(runtime.totalMemory() - runtime.freeMemory()) / 1048576L;
//                    final long maxHeapSizeInMB=runtime.maxMemory() / 1048576L;
//                    final long availHeapSizeInMB = maxHeapSizeInMB - usedMemInMB;
//                    Log.d("MEMORY:", "Used: "+usedMemInMB);

                    }
                }
            });
            myThread.start();
        } catch (Exception e) {
            Log.e(getClass().getSimpleName(), e.toString());
        }
    }

    private void clearCache() {
        isClearing = true;
        accXList = null;
        accYList = null;
        accZList = null;
        gyroXList = null;
        gyroYList = null;
        gyroZList = null;
        lock = false;
        isClearing = false;
    }

    private float getMean(ArrayList<Long> myList) {
        float sum = 0;
        for (int i = 0; i < myList.size(); i++) {
            sum += myList.get(i);
        }
        float res = sum / myList.size();
        return res;
    }

    public void stopRecord() {
        isRecording = false;
        isUpdateText = true;
        try {
            if (!myThread.isInterrupted()) {
                myThread.interrupt();
                myThread = null;
                Log.d("Interrupt my Thread:", "interrupt success");
            }
        } catch (Exception e) {
            Log.e("Interrupt my Thread:", "interrupt failed");
        }
        clearCache();

        // Compute average time and update UI
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.N) {
            double meanEncTime = timeEncList.stream().mapToDouble(a -> a).average().getAsDouble();
            double meanDecTime = timeDecList.stream().mapToDouble(a -> a).average().getAsDouble();
            double meanTotalTime = timeTotalList.stream().mapToDouble(a -> a).average().getAsDouble();
//            double meanPrivateTime = timePrivateList.stream().mapToDouble(a->a).average().getAsDouble();
            double meanPublicTime = timePublicList.stream().mapToDouble(a -> a).average().getAsDouble();
            txtMain.setText(String.format("%s\nMean Total: %s ms\nMean Public: %s ms\nMean Enc: %s ms\nMean Dec: %s ms",
                    txtMain.getText().toString(), meanTotalTime, meanPublicTime, meanEncTime, meanDecTime));
        } else {
            float meanEncTime = getMean(timeEncList);
            float meanDecTime = getMean(timeDecList);
            float meanTotalTime = getMean(timeTotalList);
//            float meanPrivateTime = getMean(timePrivateList);
            float meanPublicTime = getMean(timePublicList);
            txtMain.setText(String.format("%s\nMean Total: %s ms\nMean Public: %s ms\nMean Enc: %s ms\nMean Dec: %s ms",
                    txtMain.getText().toString(), meanTotalTime, meanPublicTime, meanEncTime, meanDecTime));
        }
        scroll_view.fullScroll(View.FOCUS_DOWN);
    }

    public float[] getWindowedData(CircularArrayList<Float> DataList) {
        float[] window = new float[WINDOW_SIZE];
        // Read windowed data
        for (int i = 0; i < WINDOW_SIZE; i++) {
            window[i] = DataList.get(i);
        }
        // Remove the oldest data of stride length
        for (int i = 0; i < STRIDE_LENGTH; i++) {
            DataList.remove(0);
        }
        return window;
    }

    public boolean hasNext() {
//        Log.d("HasNext:", "accXList size:" + accXList.size() );
//        return accXList.size() > (WINDOW_SIZE + tmpAccBatchCounter * STRIDE_LENGTH) && gyroXList.size() > (WINDOW_SIZE + tmpGyroBatchCounter * STRIDE_LENGTH);
        return accXList.size() >= (WINDOW_SIZE) && gyroXList.size() >= (WINDOW_SIZE);
    }

    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, mySamplingRate);
        mSensorManager.registerListener(this, mSensorGyroscope, mySamplingRate);
    }

    protected void onPause() {
        super.onPause();
        mSensorManager.unregisterListener(this);
    }

    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        Log.d(this.getClass().getSimpleName(), "Accuracy changed:" + sensor.getName() + accuracy);
    }

    public void onSensorChanged(SensorEvent event) {
        try {
            int sensorType = event.sensor.getType();
            // Wait if reading data
            while (lock) {
            }
            // Write lock
            lock = true;
            switch (sensorType) {
                // Event came from the sensor.
                case Sensor.TYPE_ACCELEROMETER:
                    float accX = event.values[0];
                    float accY = event.values[1];
                    float accZ = (float) (event.values[2] - 9.81); // Remove G=9.81
                    if (isRecording && !isClearing) {
                        if (accXList.size() == accXList.capacity()) {
                            accXList.remove(0);
                            accYList.remove(0);
                            accZList.remove(0);
                        }
                        accXList.add(accX);
                        accYList.add(accY);
                        accZList.add(accZ);
                    }
                    if (isUpdateText) {
                        txtAcc.setText(String.format("Accelerometer:\n%s\n%s\n%s\n", accX, accY, accZ));
                    }
                    break;
                case Sensor.TYPE_GYROSCOPE:
                    float[] gyros = event.values;
                    if (isRecording && !isClearing) {
                        if (gyroXList.size() == gyroXList.capacity()) {
                            gyroXList.remove(0);
                            gyroYList.remove(0);
                            gyroZList.remove(0);
                        }
                        gyroXList.add(gyros[0]);
                        gyroYList.add(gyros[1]);
                        gyroZList.add(gyros[2]);
                    }
                    if (isUpdateText) {
                        txtGyro.setText(String.format("Gyroscope:\n%s\n%s\n%s\n", gyros[0], gyros[1], gyros[2]));
                    }
                    break;
                default:
                    // do nothing
            }
            // Release write lock
            lock = false;
        } catch (Exception e) {
            Log.e(getClass().getSimpleName(), e.toString());
        }
    }

    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
//            Log.d("assetFilePath", file.getAbsolutePath().toString());
            return file.getAbsolutePath();
        } else {
            Log.d("assetFilePath", "File not exist!");
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(context.getClass().getSimpleName(), "Error process asset " + assetName + " to file path");
        }
        return null;
    }

}