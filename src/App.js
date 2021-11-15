import logo from './logo.svg';
import './App.css';
import { useCallback, useEffect, useRef } from 'react';
import { useOpenCv } from 'opencv-react';
import * as faceapi from 'face-api.js';

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

  const estimatePose = () => {};

  const onPlay = async () => {
    const result = await faceapi
      .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();

    console.log('result: ', result);

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
