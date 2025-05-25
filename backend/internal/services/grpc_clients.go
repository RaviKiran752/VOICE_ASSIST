package services

import (
	"context"
	"os"
	"time"

	pb "github.com/yourusername/voice_assist/backend/internal/proto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ServiceClients struct {
	STTClient pb.STTServiceClient
	NLPClient pb.NLPServiceClient
	TTSClient pb.TTSServiceClient
	STTConn   *grpc.ClientConn
	NLPConn   *grpc.ClientConn
	TTSConn   *grpc.ClientConn
}

func NewServiceClients() (*ServiceClients, error) {
	// Get service URLs from environment variables
	sttURL := os.Getenv("STT_SERVICE_URL")
	if sttURL == "" {
		sttURL = "localhost:50051"
	}
	nlpURL := os.Getenv("NLP_SERVICE_URL")
	if nlpURL == "" {
		nlpURL = "localhost:50052"
	}
	ttsURL := os.Getenv("TTS_SERVICE_URL")
	if ttsURL == "" {
		ttsURL = "localhost:50053"
	}

	// Create gRPC connections
	sttConn, err := grpc.Dial(sttURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	nlpConn, err := grpc.Dial(nlpURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		sttConn.Close()
		return nil, err
	}

	ttsConn, err := grpc.Dial(ttsURL, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		sttConn.Close()
		nlpConn.Close()
		return nil, err
	}

	return &ServiceClients{
		STTClient: pb.NewSTTServiceClient(sttConn),
		NLPClient: pb.NewNLPServiceClient(nlpConn),
		TTSClient: pb.NewTTSServiceClient(ttsConn),
		STTConn:   sttConn,
		NLPConn:   nlpConn,
		TTSConn:   ttsConn,
	}, nil
}

func (c *ServiceClients) Close() {
	if c.STTConn != nil {
		c.STTConn.Close()
	}
	if c.NLPConn != nil {
		c.NLPConn.Close()
	}
	if c.TTSConn != nil {
		c.TTSConn.Close()
	}
}

func (c *ServiceClients) TranscribeAudio(audioData []byte, audioFormat string) (string, float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := c.STTClient.TranscribeAudio(ctx, &pb.TranscribeRequest{
		AudioData:   audioData,
		AudioFormat: audioFormat,
	})
	if err != nil {
		return "", 0, err
	}

	return resp.Text, resp.Confidence, nil
}

func (c *ServiceClients) ProcessText(text string) (string, map[string]string, float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := c.NLPClient.ProcessText(ctx, &pb.ProcessTextRequest{
		Text: text,
	})
	if err != nil {
		return "", nil, 0, err
	}

	return resp.Intent, resp.Entities, resp.Confidence, nil
}

func (c *ServiceClients) SynthesizeSpeech(text string, voiceID string) ([]byte, string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	resp, err := c.TTSClient.SynthesizeSpeech(ctx, &pb.SynthesizeRequest{
		Text:    text,
		VoiceId: voiceID,
	})
	if err != nil {
		return nil, "", err
	}

	return resp.AudioData, resp.AudioFormat, nil
}
