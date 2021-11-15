import logo from './logo.svg';
import './App.css';
import { useCallback, useEffect, useRef } from 'react';
import { useOpenCv } from 'opencv-react';
import * as faceapi from 'face-api.js';

const getPoints4 = (landmarks) => {
  return [
    landmarks[30].x,
    landmarks[30].y, // nose tip
    landmarks[30].x,
    landmarks[30].y, // nose tip
    landmarks[36].x,
    landmarks[36].y, // left corner of left eye
    landmarks[45].x,
    landmarks[45].y, // right corner of right eye
  ];
};

const getPoints6 = (landmarks) => {
  return [
    landmarks[30].x,
    landmarks[30].y, // nose tip
    landmarks[8].x,
    landmarks[8].y, // chin
    landmarks[36].x,
    landmarks[36].y, // left corner of left eye
    landmarks[45].x,
    landmarks[45].y, // right corner of right eye
    landmarks[48].x,
    landmarks[48].y, // left corner of mouth
    landmarks[54].x,
    landmarks[54].y, // right corner of mouth
  ];
};

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef();
  const imgRef = useRef();
  const { loaded: openCvLoaded, cv } = useOpenCv();

  const startVideo = useCallback(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        let video = videoRef.current;
        video.srcObject = stream;
        video.play();
      })
      .catch((err) => {
        console.error('error:', err);
      });
  }, [videoRef]);

  useEffect(() => {
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
      faceapi.nets.faceExpressionNet.loadFromUri('/models'),
    ]).then(startVideo);
  }, [startVideo]);

  // useEffect(() => {
  //   if (openCvLoaded) {
  //     const video = videoRef.current;
  //     console.log('video: ', video.width);
  //     console.log('video: ', video.height);
  //   }
  // }, [openCvLoaded, cv]);

  const estimatePose = (positions) => {
    const numRows = 4;
    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, getPoints4(positions));
    // const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1);
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0,
      0,
      0, // Nose tip
      0,
      0,
      0, // Nose tip
      // 0, -330, -65, // Chin
      -225,
      170,
      -135, // Left eye left corner
      225,
      170,
      -135, // Right eye right corne
      // -150, -150, -125,  // Left Mouth corner
      // 150, -150, -125,  // Right mouth corner
    ]);

    const size = { width: 640, height: 480 };
    const focalLength = size.width;
    const center = [size.width / 2, size.height / 2];

    const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
      focalLength,
      0,
      center[0],
      0,
      focalLength,
      center[1],
      0,
      0,
      1,
    ]);

    // [
    //   positions[30].x,
    //   positions[30].y, // nose tip
    //   positions[30].x,
    //   positions[30].y, // nose tip
    //   positions[36].x,
    //   positions[36].y, // left corner of left eye
    //   positions[45].x,
    //   positions[45].y, // right corner of right eye
    // ].map((v, i) => {
    //   imagePoints.data64F[i] = v;
    // });

    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1);
    const rotationVector = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // cv.Mat.zeros(1, 3, cv.CV_64FC1);
    const translationVector = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1); // cv.Mat.zeros(1, 3, cv.CV_64FC1);

    // Hack! initialize transition and rotation matrixes to improve estimation
    translationVector.data64F[0] = -100;
    translationVector.data64F[1] = 100;
    translationVector.data64F[2] = 1000;
    const distToLeftEyeX = Math.abs(positions[36].x - positions[30].x);
    const distToRightEyeX = Math.abs(positions[45].x - positions[30].x);
    if (distToLeftEyeX < distToRightEyeX) {
      // looking at left
      rotationVector.data64F[0] = -1.0;
      rotationVector.data64F[1] = -0.75;
      rotationVector.data64F[2] = -3.0;
    } else {
      // looking at right
      rotationVector.data64F[0] = 1.0;
      rotationVector.data64F[1] = -0.75;
      rotationVector.data64F[2] = -3.0;
    }

    console.log(
      '1',
      rotationVector.data64F.map((d) => (d / Math.PI) * 180)
    );

    const success = cv.solvePnP(
      modelPoints,
      imagePoints,
      cameraMatrix,
      distCoeffs,
      rotationVector,
      translationVector,
      true
    );
    if (!success) return;

    console.log(
      '2',
      rotationVector.data64F.map((d) => (d / Math.PI) * 180)
    );

    let rotationVectorDegree = rotationVector.data64F.map((d) => (d / Math.PI) * 180);
    console.log('rotationVectorDegree: ', rotationVectorDegree[0]);

    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rotationVector.delete();
    translationVector.delete();

    return rotationVectorDegree;
  };

  const estimatePose2 = (positions) => {
    const numRows = 6;
    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, getPoints4(positions));
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0,
      0,
      0, // Nose tip
      0,
      0,
      0, // Nose tip
      // 0, -330, -65, // Chin
      -225,
      170,
      -135, // Left eye left corner
      225,
      170,
      -135, // Right eye right corne
      // -150, -150, -125,  // Left Mouth corner
      // 150, -150, -125,  // Right mouth corner
    ]);

    const size = { width: 640, height: 480 };
    const focalLength = size.width;
    const center = [size.width / 2, size.height / 2];

    const cameraMatrix = cv.matFromArray(3, 3, cv.CV_64FC1, [
      focalLength,
      0,
      center[0],
      0,
      focalLength,
      center[1],
      0,
      0,
      1,
    ]);

    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1);
    const rotationVector = cv.Mat.zeros(1, 3, cv.CV_64FC1);
    const translationVector = cv.Mat.zeros(1, 3, cv.CV_64FC1);
    cv.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rotationVector, translationVector);
    const rotationVectorMatrix = cv.Mat.zeros(3, 3, cv.CV_64FC1);
    cv.Rodrigues(rotationVector, rotationVectorMatrix);
    const matVector = new cv.MatVector();
    matVector.push_back(rotationVectorMatrix);
    matVector.push_back(translationVector);
    const projectMatrix = cv.Mat.zeros(3, 4, cv.CV_64FC1);
    cv.hconcat(matVector, projectMatrix);
    const noArray0 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const noArray1 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const noArray2 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const noArray3 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const noArray4 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const noArray5 = cv.Mat.zeros(0, 0, cv.CV_64FC1);
    const eulerAngles = cv.Mat.zeros(1, 3, cv.CV_64FC1);
    cv.decomposeProjectionMatrix(
      projectMatrix,
      noArray0,
      noArray1,
      noArray2,
      noArray3,
      noArray4,
      noArray5,
      eulerAngles
    );
    const [yaw, pitch] = eulerAngles.data64F.map((degree) => Math.asin(Math.sin((degree / 180) * Math.PI)));
    console.log('yaw, pitch: ', yaw, pitch);
    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rotationVector.delete();
    translationVector.delete();
    rotationVectorMatrix.delete();
    projectMatrix.delete();
    matVector.delete();
    noArray0.delete();
    noArray1.delete();
    noArray2.delete();
    noArray3.delete();
    noArray4.delete();
    noArray5.delete();
    eulerAngles.delete();
    return { pitch, yaw };
  };

  const onPlay = async () => {
    const result = await faceapi
      .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    if (result) {
      const rotationVectorDegree = estimatePose2(result.landmarks.positions);
    }

    setTimeout(() => onPlay(), 100);
  };

  const capture = () => {
    let canvas = canvasRef.current;
    let video = videoRef.current;
    let img = imgRef.current;

    const width = video.videoWidth;
    const height = video.videoHeight;

    canvas.width = width;
    canvas.height = height;

    canvas.getContext('2d').drawImage(video, 0, 0, width, height);
    img.src = canvas.toDataURL('image/jpeg');
  };

  if (!openCvLoaded) {
    return <h1>Loading...</h1>;
  }

  return (
    <div className='App'>
      <video
        ref={videoRef}
        width={640}
        height={480}
        autoPlay
        muted
        playsInline
        onLoadedMetadata={() => {
          onPlay();
        }}
      />
      <img ref={imgRef} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      <button onClick={capture}>chụp</button>
    </div>
  );
}

export default App;
