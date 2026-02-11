/**
 * AudioWorklet processor for capturing microphone audio.
 *
 * Collects samples into buffers of configurable size (default 4096),
 * then posts them to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.bufferSize = options.processorOptions?.bufferSize || 4096;
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex++] = channelData[i];

      if (this.bufferIndex >= this.bufferSize) {
        // Calculate RMS level for visual feedback
        let sum = 0;
        for (let j = 0; j < this.bufferSize; j++) {
          sum += this.buffer[j] * this.buffer[j];
        }
        const level = Math.sqrt(sum / this.bufferSize);

        // Post to main thread
        this.port.postMessage({
          audioData: this.buffer.slice().buffer,
          level,
        }, [this.buffer.slice().buffer]);

        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor('audio-processor', AudioCaptureProcessor);
