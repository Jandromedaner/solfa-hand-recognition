"use client";
import React, { useState, useEffect, useRef } from "react";
import { Camera, Settings, Check, X, Volume2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Slider } from "../components/ui/slider";
import { VisuallyHidden } from "@radix-ui/react-visually-hidden";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "../components/ui/sheet";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../components/ui/select";

// Correct imports for MediaPipe
import * as Hands from "@mediapipe/hands";
// import { Camera as MPCamera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";

// example
import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

const handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton: HTMLButtonElement;
let webcamRunning: Boolean = false;

const createHandLandmarker = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm",
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "GPU",
    },
    runningMode: runningMode,
    numHands: 2,
  });
  demosSection.classList.remove("invisible");
};
createHandLandmarker();

const SolfaLearningApp = () => {
  const [currentShape, setCurrentShape] = useState(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [settings, setSettings] = useState({
    difficulty: "normal",
    gestureThreshold: 0.8,
    showGuideLines: true,
    audioEnabled: true,
    practiceMode: "sequential",
  });
  const [handLandmarks, setHandLandmarks] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);
  // const cameraRef = useRef(null);
  const handsRef = useRef(null);
  const contextRef = useRef(null);

  useEffect(() => {
    const video = document.getElementById("webcam") as HTMLVideoElement;
    const canvasElement = document.getElementById(
      "output_canvas",
    ) as HTMLCanvasElement;
    const canvasCtx = canvasElement.getContext("2d");

    // initializeCamera();

    // Check if webcam access is supported.
    const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

    // If webcam supported, add event listener to button for when user
    // wants to activate it.
    if (hasGetUserMedia()) {
      enableWebcamButton = document.getElementById("webcamButton");
      enableWebcamButton.addEventListener("click", enableCam);
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }

    // Load MediaPipe Hands asynchronously
    //    loadMediapipeHands()
    //      .then((hands) => {
    //        initializeHandDetection(hands);
    //      })
    //      .catch((error) => {
    //        console.error("Error loading MediaPipe Hands:", error);
    //      });
    //    return () => {
    //      cleanupResources();
    //    };
  }, []);

  const cleanupResources = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    if (cameraRef.current) {
      cameraRef.current.stop();
    }
    if (handsRef.current) {
      handsRef.current.close();
    }
  };

  const loadMediapipeHands = async () => {
    return new Promise((resolve, reject) => {
      // Assuming you're loading the MediaPipe Hands library via a script tag
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/@mediapipe/hands";
      script.async = true;

      script.onload = () => {
        resolve(
          new Hands.Hands({
            locateFile: (file) => {
              return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            },
          }),
        );
      };

      script.onerror = (error) => {
        reject(error);
      };

      document.body.appendChild(script);
    });
  };

  const initializeCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 640,
          height: 480,
          facingMode: "user",
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      mediaStreamRef.current = stream;
    } catch (error) {
      console.error("Error initializing camera:", error);
    }
  };

  const initializeHandDetection = (hands) => {
    if (!videoRef.current || !canvasRef.current) return;

    // Initialize canvas context and set canvas size
    const canvas = canvasRef.current;
    canvas.width = 640;
    canvas.height = 480;
    contextRef.current = canvas.getContext("2d");

    // Initialize MediaPipe Hands with loaded object
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    hands.onResults(handleHandResults);
    handsRef.current = hands;
  };

  // Initialize MediaPipe Camera
  //  const camera = new MPCamera(videoRef.current, {
  //    onFrame: async () => {
  //      if (handsRef.current) {
  //        await handsRef.current.send({ image: videoRef.current });
  //      }
  //    },
  //    width: 640,
  //    height: 480,
  //  });

  //  cameraRef.current = camera;
  //  camera.start();
  //};

  const handleHandResults = (results) => {
    const canvas = canvasRef.current;
    const ctx = contextRef.current;

    if (!canvas || !ctx) return;

    // Clear previous drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw video frame on canvas
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    // Draw hand landmarks if detected
    if (results.multiHandLandmarks?.length > 0) {
      setHandLandmarks(results.multiHandLandmarks[0]);

      if (settings.showGuideLines) {
        // Draw the hand skeleton
        drawConnectors(ctx, results.multiHandLandmarks[0], HAND_CONNECTIONS, {
          color: "#00FF00",
          lineWidth: 5,
        });
        drawLandmarks(ctx, results.multiHandLandmarks[0], {
          color: "#FF0000",
          lineWidth: 2,
        });
      }
    }
  };

  const checkHandShape = (landmarks) => {
    if (!currentShape) return;

    // Calculate finger positions and angles
    const fingerMeasurements = calculateFingerMeasurements(landmarks);
    const matchScore = compareWithTarget(
      fingerMeasurements,
      currentShape.fingerPositions,
    );

    const threshold =
      settings.difficulty === "easy"
        ? 0.6
        : settings.difficulty === "normal"
          ? 0.75
          : 0.9;

    if (matchScore >= threshold) {
      setFeedback(true);
      playAudioFeedback(true);
      setTimeout(selectRandomShape, 1000);
    } else {
      setFeedback(false);
      playAudioFeedback(false);
    }
  };

  const calculateFingerMeasurements = (landmarks) => {
    // Simplified finger measurements calculation
    return {
      thumb: calculateFingerProperties(landmarks, [2, 3, 4]),
      index: calculateFingerProperties(landmarks, [5, 6, 7, 8]),
      middle: calculateFingerProperties(landmarks, [9, 10, 11, 12]),
      ring: calculateFingerProperties(landmarks, [13, 14, 15, 16]),
      pinky: calculateFingerProperties(landmarks, [17, 18, 19, 20]),
    };
  };

  const calculateFingerProperties = (landmarks, indices) => {
    // Calculate curl and direction based on landmark positions
    const start = landmarks[indices[0]];
    const end = landmarks[indices[indices.length - 1]];

    const curl = Math.sqrt(
      Math.pow(end.x - start.x, 2) +
        Math.pow(end.y - start.y, 2) +
        Math.pow(end.z - start.z, 2),
    );

    const direction = end.y > start.y ? "down" : "up";

    return { curl, direction };
  };

  const compareWithTarget = (measured, target) => {
    // Compare measured finger positions with target shape
    let totalScore = 0;
    let measurements = 0;

    for (const [finger, position] of Object.entries(measured)) {
      if (target[finger]) {
        const curlScore = 1 - Math.abs(position.curl - target[finger].curl);
        const directionScore =
          position.direction === target[finger].direction ? 1 : 0;
        totalScore += (curlScore + directionScore) / 2;
        measurements++;
      }
    }

    return totalScore / measurements;
  };

  const playAudioFeedback = (success) => {
    if (!settings.audioEnabled) return;

    const audioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = success ? 800 : 400;
    gainNode.gain.value = 0.1;

    oscillator.start();
    setTimeout(() => {
      oscillator.stop();
      audioContext.close();
    }, 200);

    /// example

    const video = document.getElementById("webcam") as HTMLVideoElement;
    const canvasElement = document.getElementById(
      "output_canvas",
    ) as HTMLCanvasElement;
    const canvasCtx = canvasElement.getContext("2d");

    // Check if webcam access is supported.
    const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

    // If webcam supported, add event listener to button for when user
    // wants to activate it.
    if (hasGetUserMedia()) {
      enableWebcamButton = document.getElementById("webcamButton");
      enableWebcamButton.addEventListener("click", enableCam);
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }

    // Enable the live webcam view and start detection.
    function enableCam(event) {
      if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
      }

      if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
      } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
      }

      // getUsermedia parameters.
      const constraints = {
        video: true,
      };

      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
      });
    }

    let lastVideoTime = -1;
    let results = undefined;
    console.log(video);
    async function predictWebcam() {
      canvasElement.style.width = video.videoWidth;
      canvasElement.style.height = video.videoHeight;
      canvasElement.width = video.videoWidth;
      canvasElement.height = video.videoHeight;

      // Now let's start detecting the stream.
      if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
      }
      let startTimeMs = performance.now();
      if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
      }
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      if (results.landmarks) {
        for (const landmarks of results.landmarks) {
          drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5,
          });
          drawLandmarks(canvasCtx, landmarks, {
            color: "#FF0000",
            lineWidth: 2,
          });
        }
      }
      canvasCtx.restore();

      // Call this function again to keep predicting when the browser is ready.
      if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
      }
    }

    return (
      <div className="container mx-auto p-4">
        <Card className="max-w-2xl mx-auto">
          <CardHeader>
            <CardTitle className="flex justify-between items-center">
              <span>Solfa Hand Shape Learning</span>
              <Sheet>
                <SheetTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <Settings className="h-6 w-6" />
                    <VisuallyHidden>Open Settings</VisuallyHidden>
                  </Button>
                </SheetTrigger>
                <SheetContent>
                  <SheetHeader>
                    <SheetTitle>App Settings</SheetTitle>
                    <VisuallyHidden>
                      Adjust your Solfa Hand Shape Learning app settings
                    </VisuallyHidden>
                    <SheetDescription>
                      Customize your learning experience
                    </SheetDescription>
                  </SheetHeader>
                  <div className="space-y-6 mt-6">
                    <div className="space-y-2">
                      <label
                        htmlFor="difficulty"
                        className="text-sm font-medium"
                      >
                        Difficulty
                      </label>
                      <Select
                        id="difficulty"
                        value={settings.difficulty}
                        onValueChange={(value) =>
                          setSettings((prev) => ({
                            ...prev,
                            difficulty: value,
                          }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select difficulty" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="easy">Easy</SelectItem>
                          <SelectItem value="normal">Normal</SelectItem>
                          <SelectItem value="hard">Hard</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium">
                        Gesture Threshold
                      </label>
                      <Slider
                        value={[settings.gestureThreshold]}
                        min={0.5}
                        max={1}
                        step={0.1}
                        onValueChange={([value]) =>
                          setSettings((prev) => ({
                            ...prev,
                            gestureThreshold: value,
                          }))
                        }
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        Audio Feedback
                      </span>
                      <Button
                        variant={
                          settings.audioEnabled ? "default" : "secondary"
                        }
                        size="icon"
                        onClick={() =>
                          setSettings((prev) => ({
                            ...prev,
                            audioEnabled: !prev.audioEnabled,
                          }))
                        }
                      >
                        <Volume2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </SheetContent>
              </Sheet>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video bg-slate-900 rounded-lg mb-4">
              <video
                ref={videoRef}
                className="w-full h-full rounded-lg"
                autoPlay
                playsInline
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
                width={640}
                height={480}
              />
              {feedback !== null && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  {feedback ? (
                    <Check className="w-16 h-16 text-green-500" />
                  ) : (
                    <X className="w-16 h-16 text-red-500" />
                  )}
                </div>
              )}
            </div>

            <div className="text-center mb-4">
              <h2 className="text-2xl font-bold">
                {currentShape ? currentShape.name : "Loading..."}
              </h2>
            </div>

            <div className="flex justify-center gap-4">
              <Button
                onClick={() => setIsRecognizing(!isRecognizing)}
                className="flex items-center gap-2"
              >
                <Camera className="h-4 w-4" />
                {isRecognizing ? "Stop Recognition" : "Start Recognition"}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };
};

export default SolfaLearningApp;
