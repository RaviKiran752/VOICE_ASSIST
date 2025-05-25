import { useState, useRef, useEffect } from 'react';
import {
  Box,
  Button,
  VStack,
  Text,
  useToast,
  Container,
  Heading,
  Badge,
  HStack,
  Spinner,
} from '@chakra-ui/react';
import { FaMicrophone, FaStop, FaPlay, FaPause } from 'react-icons/fa';
import axios from 'axios';

interface AssistantResponse {
  transcription: {
    text: string;
    confidence: number;
  };
  understanding: {
    intent: string;
    entities: Record<string, string>;
    confidence: number;
  };
  response: {
    text: string;
    audio: string;
    audioFormat: string;
  };
}

export const VoiceAssistant = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [response, setResponse] = useState<AssistantResponse | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const audioPlayer = useRef<HTMLAudioElement | null>(null);
  const toast = useToast();

  useEffect(() => {
    // Initialize audio player
    audioPlayer.current = new Audio();
    audioPlayer.current.onended = () => setIsPlaying(false);
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {
        audioChunks.current.push(event.data);
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
        await processAudio(audioBlob);
      };

      mediaRecorder.current.start();
      setIsRecording(true);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Could not access microphone',
        status: 'error',
        duration: 3000,
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && isRecording) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  const processAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.wav');

      const response = await axios.post<AssistantResponse>(
        'http://localhost:3000/process-voice',
        audioBlob,
        {
          headers: {
            'Content-Type': 'audio/wav',
          },
        }
      );

      setResponse(response.data);
      playResponse(response.data.response.audio);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to process audio',
        status: 'error',
        duration: 3000,
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const playResponse = (audioData: string) => {
    if (audioPlayer.current) {
      audioPlayer.current.src = `data:audio/wav;base64,${audioData}`;
      audioPlayer.current.play();
      setIsPlaying(true);
    }
  };

  const togglePlayback = () => {
    if (audioPlayer.current) {
      if (isPlaying) {
        audioPlayer.current.pause();
      } else {
        audioPlayer.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  return (
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Heading textAlign="center">Voice Assistant</Heading>

        <Box
          p={6}
          borderRadius="lg"
          boxShadow="lg"
          bg="white"
          textAlign="center"
        >
          <VStack spacing={4}>
            <Button
              size="lg"
              colorScheme={isRecording ? 'red' : 'blue'}
              onClick={isRecording ? stopRecording : startRecording}
              leftIcon={isRecording ? <FaStop /> : <FaMicrophone />}
              isLoading={isProcessing}
              loadingText="Processing..."
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>

            {response && (
              <VStack spacing={4} align="stretch" w="100%">
                <Box>
                  <Text fontWeight="bold">You said:</Text>
                  <Text>{response.transcription.text}</Text>
                  <Badge colorScheme="green">
                    Confidence: {(response.transcription.confidence * 100).toFixed(1)}%
                  </Badge>
                </Box>

                <Box>
                  <Text fontWeight="bold">Intent:</Text>
                  <HStack>
                    <Badge colorScheme="purple">{response.understanding.intent}</Badge>
                    <Badge colorScheme="blue">
                      Confidence: {(response.understanding.confidence * 100).toFixed(1)}%
                    </Badge>
                  </HStack>
                </Box>

                <Box>
                  <Text fontWeight="bold">Assistant:</Text>
                  <Text>{response.response.text}</Text>
                  <Button
                    size="sm"
                    leftIcon={isPlaying ? <FaPause /> : <FaPlay />}
                    onClick={togglePlayback}
                    mt={2}
                  >
                    {isPlaying ? 'Pause' : 'Play'} Response
                  </Button>
                </Box>
              </VStack>
            )}

            {isProcessing && (
              <VStack spacing={2}>
                <Spinner size="lg" />
                <Text>Processing your request...</Text>
              </VStack>
            )}
          </VStack>
        </Box>
      </VStack>
    </Container>
  );
}; 