package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/yourusername/voice_assist/backend/internal/services"
)

func main() {
	// Initialize gRPC clients
	clients, err := services.NewServiceClients()
	if err != nil {
		log.Fatalf("Failed to initialize service clients: %v", err)
	}
	defer clients.Close()

	// Create new Fiber app
	app := fiber.New(fiber.Config{
		AppName: "VoiceAssist API v1.0",
	})

	// Middleware
	app.Use(recover.New())
	app.Use(logger.New())
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowHeaders: "Origin, Content-Type, Accept",
	}))

	// Health check endpoint
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"status":  "ok",
			"service": "voice_assist",
		})
	})

	// Process voice input endpoint
	app.Post("/process-voice", func(c *fiber.Ctx) error {
		// Get audio data from request
		audioData := c.Body()
		if len(audioData) == 0 {
			return c.Status(400).JSON(fiber.Map{
				"error": "No audio data provided",
			})
		}

		// Get audio format from header or default to wav
		audioFormat := c.Get("Content-Type", "audio/wav")
		if audioFormat == "audio/wav" {
			audioFormat = "wav"
		}

		// Step 1: Transcribe audio to text
		text, confidence, err := clients.TranscribeAudio(audioData, audioFormat)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error": "Failed to transcribe audio",
			})
		}

		// Step 2: Process text to understand intent
		intent, entities, intentConfidence, err := clients.ProcessText(text)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error": "Failed to process text",
			})
		}

		// Step 3: Generate response based on intent
		response := generateResponse(intent, entities)

		// Step 4: Convert response to speech
		audioResponse, audioFormat, err := clients.SynthesizeSpeech(response, "default")
		if err != nil {
			return c.Status(500).JSON(fiber.Map{
				"error": "Failed to synthesize speech",
			})
		}

		// Return the complete response
		return c.JSON(fiber.Map{
			"transcription": fiber.Map{
				"text":       text,
				"confidence": confidence,
			},
			"understanding": fiber.Map{
				"intent":     intent,
				"entities":   entities,
				"confidence": intentConfidence,
			},
			"response": fiber.Map{
				"text":        response,
				"audio":       audioResponse,
				"audioFormat": audioFormat,
			},
		})
	})

	// Handle graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down gracefully...")
		app.Shutdown()
	}()

	// Start server
	log.Fatal(app.Listen(":3000"))
}

func generateResponse(intent string, entities map[string]string) string {
	switch intent {
	case "greeting":
		return "Hello! How can I help you today?"
	case "farewell":
		return "Goodbye! Have a great day!"
	case "weather":
		if location, ok := entities["GPE"]; ok {
			return "I'm sorry, I don't have access to weather information for " + location + " at the moment."
		}
		return "I'm sorry, I don't have access to weather information at the moment."
	case "time":
		return "I'm sorry, I don't have access to the current time at the moment."
	case "date":
		return "I'm sorry, I don't have access to the current date at the moment."
	case "command":
		return "I'm sorry, I don't have the capability to execute that command yet."
	case "general_question":
		return "I'm sorry, I don't have enough information to answer that question."
	default:
		return "I'm sorry, I didn't understand that. Could you please rephrase?"
	}
}
