# DM-Codec

This repository contains the source code for DM-Codec. 

As illustrated in **Figure 1**, DM-Codec introduces speech tokenization approaches using discrete acoustic, semantic, and contextual tokens. DM-Codec integrates these multimodal representations for robust speech tokenization, learning comprehensive speech representations.

<img src="assets/intro_figure.png" alt="Figure 1" width="80%" /><br>

The DM-Codec framework is further detailed in **Figure 2**. The framework consists of an encoder that extracts latent representations from the input speech signal. These latent vectors are subsequently quantized using a Residual Vector Quantizer (RVQ). We designed two distinct distillation approaches: (i) distillation from a language model, and (ii) a combined distillation from both a language model (LM) and a speech model (SM). These approaches integrate acoustic,semantic, and contextual representations into the quantized vectors to improve speech representation for downstream tasks.

<img src="assets/tokenizer_figure.png" alt="Figure 2" width="80%" />

## Status

- We have released code and trained model checkpoints.

More instructions and details will be provided soon.

## Model Checkpoints

| Model | Description |
|-------|-------------|
| [DM-Codec_checkpoint_LM_SM](https://drive.google.com/file/d/1pvrPcbUTUAlo2_iGLIIf7IFNJPfzmHbh/view?usp=drive_link) | Utilizes LM and SM-guided representation distillation approach uniting acoustic, semantic, and contextual representations into DM-Codec. |
| [DM-Codec_checkpoint_LM](https://drive.google.com/file/d/14DMrDzIssP-8qzXBG8v65ctF4WPA_Uyv/view?usp=drive_link) | Utilizes LM-guided representation distillation approach incorporating acoustic and contextual representations into DM-Codec. |


## Speech Reconstruction

Below, we provide reconstructed speech samples from DM-Codec and compare them with the reconstructed speech from EnCodec, SpeechTokenizer, and FACodec. Click on the audio files to listen.

| Codec             | Reconstructed Sample 1                                   | Reconstructed Sample 2                                   |
|--------------------|---------------------------------------------------------|---------------------------------------------------------|
| **Original**        | <audio controls><source src="assets/Original_sample_132.flac" type="audio/flac">Your browser does not support the audio element.</audio> | <audio controls><source src="assets/Original_sample_118.flac" type="audio/flac">Your browser does not support the audio element.</audio> |
| **DM-Codec**        | <audio controls><source src="assets/DM-Codec_sample_132.flac" type="audio/flac">Your browser does not support the audio element.</audio> | <audio controls><source src="assets/DM-Codec_sample_118.flac" type="audio/flac">Your browser does not support the audio element.</audio> |
| **EnCodec**         | <audio controls><source src="assets/EnCodec_sample_132.flac" type="audio/flac">Your browser does not support the audio element.</audio> | <audio controls><source src="assets/EnCodec_sample_118.flac" type="audio/flac">Your browser does not support the audio element.</audio> |
| **SpeechTokenizer** | <audio controls><source src="assets/SpeechTokenizer_sample_132.flac" type="audio/flac">Your browser does not support the audio element.</audio> | <audio controls><source src="assets/SpeechTokenizer_sample_118.flac" type="audio/flac">Your browser does not support the audio element.</audio> |
| **FACodec**         | <audio controls><source src="assets/FACodec_sample_132.flac" type="audio/flac">Your browser does not support the audio element.</audio> | <audio controls><source src="assets/FACodec_sample_118.flac" type="audio/flac">Your browser does not support the audio element.</audio> |

