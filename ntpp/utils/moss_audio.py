"""Convert raw waveform manifest entries to NTPP-compatible Moss code tokens."""

from pathlib import Path

import torch
import torchaudio
import soundfile as sf


class MossAudioTokenEncoder:
    """Encode stereo speaker tracks as interleaved Moss RVQ tokens.

    The model implements the paper's RVQ pair objective. Its two channels are
    speakers, so encode each input channel independently and
    flatten each frame as [A_q0, ..., A_qD, B_q0, ..., B_qD].
    """

    def __init__(self, model_name_or_path, num_codebooks=4, device="cpu"):
        from third_party.moss_nano_trainer_unofficial.moss_nano.configuration_moss_audio_tokenizer import (
            MossAudioTokenizerConfig,
        )

        self.model_name_or_path = model_name_or_path
        self.config = MossAudioTokenizerConfig.from_pretrained(model_name_or_path)
        self._model = None
        self.device = torch.device(device)
        self.num_codebooks = int(num_codebooks)
        if not 1 <= self.num_codebooks <= self.config.num_quantizers:
            raise ValueError("Requested depth must be within the codec quantizer count")
        self.sample_rate = int(self.config.sampling_rate)
        self.channels = int(self.config.number_channels)
        self.codebook_size = int(self.config.codebook_size)
        self.downsample_rate = int(self.config.downsample_rate)

    @property
    def model(self):
        # Import the bundled class: HF 4.46's remote-code scanner mistakes the
        # fallback sibling configuration import for an external dependency.
        if self._model is None:
            from third_party.moss_nano_trainer_unofficial.moss_nano.modeling_moss_audio_tokenizer import (
                MossAudioTokenizerModel,
            )
            self._model = MossAudioTokenizerModel.from_pretrained(
                self.model_name_or_path, config=self.config
            ).eval().requires_grad_(False).to(self.device)
        return self._model

    @property
    def token_strings(self):
        return [
            f"<moss_q{q}_{code}>"
            for q in range(self.num_codebooks)
            for code in range(self.codebook_size)
        ]

    def _load_waveform(self, source, require_two=True):
        if isinstance(source, dict):
            wav = torch.as_tensor(source["array"], dtype=torch.float32)
            sample_rate = int(source["sampling_rate"])
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            elif wav.ndim == 2 and wav.shape[0] > wav.shape[1]:
                # SoundFile returns [samples, channels], while torchaudio uses
                # [channels, samples]. Convert the former before channel fitting.
                wav = wav.transpose(0, 1)
        else:
            samples, sample_rate = sf.read(str(Path(source)), dtype="float32", always_2d=True)
            wav = torch.from_numpy(samples.T.copy())

        if sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, self.sample_rate)

        if require_two and wav.shape[0] < 2:
            raise ValueError(
                "NTPP training requires two aligned speaker tracks; this audio has fewer than two channels."
            )
        wav = wav[: 2 if require_two else 1].contiguous()
        if wav.shape[-1] == 0:
            raise ValueError(f"Empty waveform: {source}")
        return wav

    @torch.inference_mode()
    def encode_codes(self, source):
        """Return aligned codes with shape [speaker=2, depth, frame]."""
        speaker_tracks = self._load_waveform(source)
        speaker_codes = []
        for speaker_idx in range(2):
            # Moss requires two input channels; duplicate this speaker track
            # across its codec input channels so speakers are encoded separately.
            mono = speaker_tracks[speaker_idx : speaker_idx + 1]
            codec_input = mono.repeat(self.channels, 1).unsqueeze(0).to(self.device)
            encoded = self.model.encode(codec_input, num_quantizers=self.num_codebooks, return_dict=True)
            codes = encoded.audio_codes
            if codes is None or codes.ndim != 3 or codes.shape[1] != 1:
                raise ValueError("Moss encode() must return [quantizers, 1, frames] codes")
            if codes.shape[0] < self.num_codebooks:
                raise ValueError(
                    f"Moss returned {codes.shape[0]} RVQ levels, but {self.num_codebooks} were requested"
                )
            valid_frames = (
                int(encoded.audio_codes_lengths[0])
                if encoded.audio_codes_lengths is not None
                else codes.shape[-1]
            )
            speaker_codes.append(codes[: self.num_codebooks, 0, :valid_frames].cpu())
        frame_count = min(codes.shape[-1] for codes in speaker_codes)
        return torch.stack(
            [codes[:, :frame_count] for codes in speaker_codes], dim=0
        )

    @torch.inference_mode()
    def encode_single_codes(self, source, channel=0):
        """Return one speaker track as [depth, frame] codes for stage 1 NTP."""
        tracks = self._load_waveform(source, require_two=False)
        if channel != 0:
            raise ValueError("Single-channel stage-1 encoding uses channel=0")
        mono = tracks[channel : channel + 1]
        codec_input = mono.repeat(self.channels, 1).unsqueeze(0).to(self.device)
        encoded = self.model.encode(codec_input, num_quantizers=self.num_codebooks, return_dict=True)
        codes = encoded.audio_codes
        if codes is None or codes.ndim != 3 or codes.shape[1] != 1:
            raise ValueError("Moss encode() must return [quantizers, 1, frames] codes")
        if codes.shape[0] < self.num_codebooks:
            raise ValueError(
                f"Moss returned {codes.shape[0]} RVQ levels, but {self.num_codebooks} were requested"
            )
        valid_frames = (
            int(encoded.audio_codes_lengths[0])
            if encoded.audio_codes_lengths is not None
            else codes.shape[-1]
        )
        return codes[: self.num_codebooks, 0, :valid_frames].cpu()

    @torch.inference_mode()
    def encode_single(self, source, channel=0):
        """Format one track frame-wise as [q0, ..., qD] for stage 1."""
        return self._format_codes(self.encode_single_codes(source, channel).transpose(0, 1).reshape(-1))

    @torch.inference_mode()
    def encode(self, source):
        speaker_codes = self.encode_codes(source)
        # [speaker, depth, frame] -> [frame, speaker, depth], matching Eq. (10).
        interleaved = speaker_codes.permute(2, 0, 1).reshape(-1)
        return self._format_codes(interleaved)

    @torch.inference_mode()
    def decode_codes(self, codes):
        """Decode [depth, frame] Moss codes to a waveform tensor."""
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(
                f"codes must have shape [{self.num_codebooks}, frames], got {tuple(codes.shape)}"
            )
        decoded = self.model.decode(codes.unsqueeze(1), return_dict=True)
        return decoded.audio.squeeze(0).cpu()

    def _format_codes(self, interleaved_codes):
        tokens = []
        for idx, code in enumerate(interleaved_codes.tolist()):
            q = idx % self.num_codebooks
            if code < 0 or code >= self.codebook_size:
                raise ValueError(f"Moss code {code} outside codebook range [0, {self.codebook_size})")
            tokens.append(f"<moss_q{q}_{code}>")
        # Keep special tokens adjacent: whitespace would itself tokenize into
        # ordinary LM tokens and shift the alternating NTPP channel positions.
        return "".join(tokens)
