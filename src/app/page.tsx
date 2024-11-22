import React, { useState, useEffect, useRef } from "react";
import { Camera, Settings, Check, X } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

const solfaHandShapes = [
  { name: "Do", audio: "do.mp3", difficulty: 1 },
  { name: "Re", audio: "re.mp3", difficulty: 1 },
  { name: "Mi", audio: "mi.mp3", difficulty: 1 },
  { name: "Fa", audio: "fa.mp3", difficulty: 2 },
  { name: "Sol", audio: "sol.mp3", difficulty: 2 },
  { name: "La", audio: "la.mp3", difficulty: 2 },
  { name: "Ti", audio: "ti.mp3", difficulty: 3 },
];

const SolfaLearningApp = () => {
  const [currentShape, setCurrentShape] = useState(null);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [difficulty, setDifficulty] = useState("normal");
  const videoRef = useRef(null);

  useEffect(() => {
    // Select random shape on mount
    selectRandomShape();
  }, []);

  const selectRandomShape = () => {
    const randomIndex = Math.floor(Math.random() * solfaHandShapes.length);
    setCurrentShape(solfaHandShapes[randomIndex]);
    setFeedback(null);
  };

  const startRecognition = async () => {
    setIsRecognizing(true);
    // MediaPipe implementation will go here
    // For now, just simulate recognition
    setTimeout(() => {
      const success = Math.random() > 0.5;
      setFeedback(success);
      if (success) {
        setTimeout(selectRandomShape, 1000);
      }
      setIsRecognizing(false);
    }, 2000);
  };

  return (
    <div className="container mx-auto p-4">
      <Card className="max-w-2xl mx-auto">
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            <span>Solfa Hand Shape Learning</span>
            <Button variant="ghost" size="icon">
              <Settings className="h-6 w-6" />
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Camera Preview */}
          <div className="relative aspect-video bg-slate-900 rounded-lg mb-4">
            <video
              ref={videoRef}
              className="w-full h-full rounded-lg"
              autoPlay
              playsInline
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

          {/* Current Shape Display */}
          <div className="text-center mb-4">
            <h2 className="text-2xl font-bold">
              {currentShape ? currentShape.name : "Loading..."}
            </h2>
          </div>

          {/* Controls */}
          <div className="flex justify-center gap-4">
            <Button
              onClick={startRecognition}
              disabled={isRecognizing}
              className="flex items-center gap-2"
            >
              <Camera className="h-4 w-4" />
              {isRecognizing ? "Recognizing..." : "Start Recognition"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default SolfaLearningApp;
