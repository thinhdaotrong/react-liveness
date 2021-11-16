import logo from './logo.svg';
import './App.css';
import { useCallback, useEffect, useRef } from 'react';
import { useOpenCv } from 'opencv-react';
import * as faceapi from 'face-api.js';
import opencv from 'opencv';

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

  const estimatePose = async (positions) => {
    // const cv = await opencv;
    const numRows = 4;
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0.0,
      0.0,
      0.0, // Nose tip
      0.0,
      0.0,
      0.0, // HACK! solvePnP doesn't work with 3 points, so copied the
      //   first point to make the input 4 points
      // 0.0, -330.0, -65.0,  // Chin
      -225.0,
      170.0,
      -135.0, // Left eye left corner
      225.0,
      170.0,
      -135.0, // Right eye right corne
      // -150.0, -150.0, -125.0,  // Left Mouth corner
      // 150.0, -150.0, -125.0,  // Right mouth corner
    ]);

    // Camera internals
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

    const imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);

    const ns = positions[30];
    const le = positions[37];
    const re = positions[44];

    // const ns = positions[30];
    // const le = positions[36];
    // const re = positions[45];

    [
      ns.x,
      ns.y, // nose tip
      ns.x,
      ns.y, // nose tip
      le.x,
      le.y, // left corner of left eye
      re.x,
      re.y, // right corner of right eye
    ].map((v, i) => {
      imagePoints.data64F[i] = v;
    });

    // Hack! initialize transition and rotation matrixes to improve estimation
    tvec.data64F[0] = -100;
    tvec.data64F[1] = 100;
    tvec.data64F[2] = 1000;
    const distToLeftEyeX = Math.abs(le.x - ns.x);
    const distToRightEyeX = Math.abs(re.x - ns.x);
    if (distToLeftEyeX < distToRightEyeX) {
      // looking at left
      rvec.data64F[0] = -1.0;
      rvec.data64F[1] = -0.75;
      rvec.data64F[2] = -3.0;
    } else {
      // looking at right
      rvec.data64F[0] = 1.0;
      rvec.data64F[1] = -0.75;
      rvec.data64F[2] = -3.0;
    }

    const success = cv.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true);
    if (!success) return;

    let rvecDegree = rvec.data64F.map((d) => (d / Math.PI) * 180);
    console.log('rvecDegree: ', rvecDegree);

    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();

    return rvecDegree;
  };

  const estimatePose2 = async (positions) => {
    const cv = await opencv;
    const numRows = 6;

    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, getPoints6(positions));
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0,
      0,
      0, // Nose tip
      -330,
      -65, // Chin
      -225,
      170,
      -135, // Left eye left corner
      225,
      170,
      -135, // Right eye right corne
      -150,
      -150,
      -125, // Left Mouth corner
      150,
      -150,
      -125, // Right mouth corner
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

  const estimatePose3 = async (positions) => {
    // const cv = await opencv;
    const numRows = 6;
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0.0,
      0.0,
      0.0, // Nose tip
      0.0,
      -330.0,
      -65.0, // Chin
      -225.0,
      170.0,
      -135.0, // Left eye left corner
      225.0,
      170.0,
      -135.0, // Right eye right corne
      -150.0,
      -150.0,
      -125.0, // Left Mouth corner
      150.0,
      -150.0,
      -125.0, // Right mouth corner
    ]);

    // Camera internals
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

    // const imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
    const imagePoints = cv.matFromArray(6, 2, cv.CV_64FC1, getPoints6(positions));
    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    // const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    // const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const rvec = cv.Mat.zeros(1, 3, cv.CV_64FC1);
    const tvec = cv.Mat.zeros(1, 3, cv.CV_64FC1);

    // Hack! initialize transition and rotation matrixes to improve estimation
    // tvec.data64F[0] = -100;
    // tvec.data64F[1] = 100;
    // tvec.data64F[2] = 1000;
    // const distToLeftEyeX = Math.abs(le.x - ns.x);
    // const distToRightEyeX = Math.abs(re.x - ns.x);
    // if (distToLeftEyeX < distToRightEyeX) {
    //   // looking at left
    //   rvec.data64F[0] = -1.0;
    //   rvec.data64F[1] = -0.75;
    //   rvec.data64F[2] = -3.0;
    // } else {
    //   // looking at right
    //   rvec.data64F[0] = 1.0;
    //   rvec.data64F[1] = -0.75;
    //   rvec.data64F[2] = -3.0;
    // }

    const success = cv.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true);
    if (!success) return;

    console.log(
      'sau',
      rvec.data64F.map((d) => (d / Math.PI) * 180)
    );

    let rvecDegree = rvec.data64F.map((d) => (d / Math.PI) * 180);
    // console.log('rvecDegree: ', rvecDegree);

    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();

    return rvecDegree;
  };

  const estimatePose4 = async (positions) => {
    // const cv = await opencv;
    const numRows = 4;
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0.0,
      0.0,
      0.0, // Nose tip
      0.0,
      0.0,
      0.0, // HACK! solvePnP doesn't work with 3 points, so copied the
      //   first point to make the input 4 points
      // 0.0, -330.0, -65.0,  // Chin
      -225.0,
      170.0,
      -135.0, // Left eye left corner
      225.0,
      170.0,
      -135.0, // Right eye right corne
      // -150.0, -150.0, -125.0,  // Left Mouth corner
      // 150.0, -150.0, -125.0,  // Right mouth corner
    ]);

    // Camera internals
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

    const imagePoints = cv.Mat.zeros(numRows, 2, cv.CV_64FC1);
    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);

    const ns = positions[30];
    const le = positions[36];
    const re = positions[45];

    [
      ns.x,
      ns.y, // nose tip
      ns.x,
      ns.y, // nose tip
      le.x,
      le.y, // left corner of left eye
      re.x,
      re.y, // right corner of right eye
    ].map((v, i) => {
      imagePoints.data64F[i] = v;
    });

    // Hack! initialize transition and rotation matrixes to improve estimation
    // tvec.data64F[0] = -100;
    // tvec.data64F[1] = 100;
    // tvec.data64F[2] = 1000;
    // const distToLeftEyeX = Math.abs(le.x - ns.x);
    // const distToRightEyeX = Math.abs(re.x - ns.x);
    // if (distToLeftEyeX < distToRightEyeX) {
    //   // looking at left
    //   rvec.data64F[0] = -1.0;
    //   rvec.data64F[1] = -0.75;
    //   rvec.data64F[2] = -3.0;
    // } else {
    //   // looking at right
    //   rvec.data64F[0] = 1.0;
    //   rvec.data64F[1] = -0.75;
    //   rvec.data64F[2] = -3.0;
    // }

    const success = cv.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true);
    if (!success) return;

    let rvecDegree = rvec.data64F.map((d) => (d / Math.PI) * 180);
    console.log('rvecDegree: ', rvecDegree);

    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();

    return rvecDegree;
  };

  const estimatePose5 = async (positions) => {
    // const cv = await opencv;
    const numRows = 6;
    const modelPoints = cv.matFromArray(numRows, 3, cv.CV_64FC1, [
      0.0,
      0.0,
      0.0, // Nose tip
      0.0,
      -330.0,
      -65.0, // Chin
      -225.0,
      170.0,
      -135.0, // Left eye left corner
      225.0,
      170.0,
      -135.0, // Right eye right corne
      -150.0,
      -150.0,
      -125.0, // Left Mouth corner
      150.0,
      -150.0,
      -125.0, // Right mouth corner
    ]);

    // Camera internals
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

    const imagePoints = cv.matFromArray(numRows, 2, cv.CV_64FC1, getPoints6(positions));
    const distCoeffs = cv.Mat.zeros(4, 1, cv.CV_64FC1); // Assuming no lens distortion
    const rvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);
    const tvec = new cv.Mat({ width: 1, height: 3 }, cv.CV_64FC1);

    const ns = positions[30];
    const le = positions[36];
    const re = positions[45];

    // Hack! initialize transition and rotation matrixes to improve estimation
    tvec.data64F[0] = -100;
    tvec.data64F[1] = 100;
    tvec.data64F[2] = 1000;
    const distToLeftEyeX = Math.abs(le.x - ns.x);
    const distToRightEyeX = Math.abs(re.x - ns.x);
    if (distToLeftEyeX < distToRightEyeX) {
      // looking at left
      rvec.data64F[0] = -1.0;
      rvec.data64F[1] = -0.75;
      rvec.data64F[2] = -3.0;
    } else {
      // looking at right
      rvec.data64F[0] = 1.0;
      rvec.data64F[1] = -0.75;
      rvec.data64F[2] = -3.0;
    }

    const success = cv.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, true);
    if (!success) return;

    let rvecDegree = rvec.data64F.map((d) => (d / Math.PI) * 180);
    console.log('rvecDegree: ', rvecDegree);

    imagePoints.delete();
    modelPoints.delete();
    cameraMatrix.delete();
    distCoeffs.delete();
    rvec.delete();
    tvec.delete();

    return rvecDegree;
  };

  const onPlay = async () => {
    const result = await faceapi
      .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks();

    if (result) {
      const rotationVectorDegree = estimatePose(result.landmarks.positions);

      const dims = faceapi.matchDimensions(canvasRef.current, videoRef.current, true);
      const resizedResult = faceapi.resizeResults(result, dims);
      faceapi.draw.drawDetections(canvasRef.current, resizedResult);
      faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedResult);
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
    <div className='App' style={{ position: 'relative' }}>
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
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0 }} />
      <button onClick={capture}>chụp</button>
    </div>
  );
}

export default App;
