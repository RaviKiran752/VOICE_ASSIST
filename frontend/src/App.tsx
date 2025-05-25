import { ChakraProvider, CSSReset } from '@chakra-ui/react';
import { VoiceAssistant } from './components/VoiceAssistant';

function App() {
  return (
    <ChakraProvider>
      <CSSReset />
      <VoiceAssistant />
    </ChakraProvider>
  );
}

export default App;
