# Ultrastar Architecture — Core (§1–§7)

> Core spec kept concise. Bulky normative blocks live in **Ultrastar Arch — Appendices A–D**.

---

## 1) Purpose & Scope

* **Goal:** 1:1 USDX compatibility; cross‑platform; local‑only (no cloud).
* **Included:** local FS/USB library + LAN ingress; Solo/Duet/Party; scoring; HUD; persistence; accessibility.
* **Excluded v1:** Cloud catalogs/leaderboards; non‑USDX formats.
* **Determinism:** Fixed clocks/rounding; identical inputs → identical outputs; seeded RNG for tests.

## 2) Supported Platforms & Runtimes

* **Tier‑A:** Android/Google TV (arm64‑v8a). **Tier‑B:** Tizen TV, Desktop, Mobile — gameplay/scoring parity required.
* **Audio engine:** 48 kHz; fixed tick (see §8). UI 60 fps; input→pitch ≤ 60 ms.
* **Mics:** Up to **4** streams with **Capability Gating** (auto‑bench on device). Per‑mic gain/latency calibration.
* **Controllers:** D‑pad/gamepads; deterministic focus rules.

## 3) Folder & File Conventions (Library Layout)

* **Repositories:** `LocalFilesystemRepository` (roots, USB) and `LanIngressRepository` (paired WS ingress). Unified catalog.
* **Admission:** `.txt` parses (§4) **and** playable audio present. Video optional. Unknown tags preserved.
* **Asset resolution:** USDX tags first; non‑configurable community fallback for `cover.*` / `background.*` only.
* **Safety:** Unicode NFC; case‑insensitive match; reject absolute/`..`; ignore system files.
* **Streams:** `ReadableByteStream` with seekable audio/video; defined error types.
* **Ingress:** See Appendix **A** (protocol & MIME). Limits: size caps, rate limits, pairing tokens.

## 4) USDX File Format Compatibility

* **Encoding:** UTF‑8; CR/LF tolerant. **Required tags:** `#TITLE`, `#ARTIST`, `#BPM`, and `#AUDIO` (or legacy `#MP3`).
* **Notes grammar:** `:|*|F|R|G  StartBeat Length Pitch Text`; `- EndBeat`; `E`; `P1/P2`.
* **Paths:** Relative to `.txt` folder; case‑insensitive; community fallbacks as above.
* **Duet:** `P1/P2` define voices; names from `#P1/#P2` (legacy normalized).
* **Errors/limits:** See Appendix **B** (taxonomy & hard limits). Mapping to admission reasons fixed.

## 5) Media Format Support

* **Audio (guaranteed):** MP3 (CBR), OGG, WAV. **Video (guaranteed):** H.264/MP4 up to 1080p30; MPEG‑1 best‑effort. **Images:** JPEG/JPG; PNG via fallback only.
* **Determinism:** Resampling/mixing must not affect scoring math. `#VIDEOGAP` applies to video only. See Appendix **C** for minima.

## 6) Core Domain Model

* **Entities:** `Song`, `SongMeta`, `SongAssets`, `Voice`, `Line`, `Note`, `MedleyWindow`, `Preview` with invariants (beats as ints; orderings; durations).
* **Identity:** `SongId` from §3 (repo + content hash); `SongHash` for de‑dup across roots.
* **Interfaces:** `NoteMatcher`, `ScoringEngine` (wired in §12). Units: engine µs; UI ms; pitch relative to C4.

## 7) Library Index & Persistence

* **Per‑root SQLite sidecar** `.ultrastar_catalog_v1.sqlite` (WAL). RO‑USB → mirrored in app cache.
* **Presence:** Union of mounted repos; unavailable items hidden by default; recent/favorites by `SongHash` with greying.
* **De‑dup & canonical origin:** Prefer Persistent FS → Removable FS → Ingress; tie: latest mtime then RootId.
* **Scan:** Manual only; incremental by `(path,size,mtime)`; bounded hashing concurrency; progress UI.
* **Schema & PRAGMAs:** Move to Appendix **D** (DDL, PRAGMAs, Unicode normalization, RootIdHash, diagnostics).

---

## 8) Timing & Synchronization

### 8.1 Master Clocks & Time Domains

* **Global master:** **System monotonic clock** on the host device (TV/desktop/mobile). Unit: **microseconds since boot**.
* **Audio device clock:** measured and slaved to the global master via latency estimation and **drift slewing** (§8.3). **Scoring uses the global master.**
* **Video timeline:** slaved to **audio playback**, adjusted by `#VIDEOGAP` (§4). Video never influences score timing.
* **Wi‑Fi mics:** Each phone maintains an offset to the host’s **global master** via lightweight LAN time‑sync (see §10.6) using the same pairing trust as ingress (Appendix A).

### 8.2 Engine Tick & Quantization

* **Tick size:** **10 ms** at 48 kHz (**480 samples**). All scheduling is quantized to ticks. (See Core §2 for engine rate.)
* **Rounding:** **nearest, ties‑to‑even** when converting between beats/seconds/ticks.
* **Beat time:** `beatSeconds = (beatIndex / BPM) * 60`. `#GAP` shifts **lyrics** only (§4); `#START` trims audio/video playback; `#VIDEOGAP` shifts video only.

### 8.3 Drift & Latency Model

* **Audio output latency:** measured on init and updated when device changes; stored as `latencyOutputMs`.
* **Slew policy:** Correct audio clock drift by **≤ ±1 ms per second** (slew), re‑measure if accumulated drift exceeds **30 ms**. Never hard‑jump during an active line; apply correction between lines or during pre‑roll.
* **Wi‑Fi mic jitter buffer:** Target **80 ms** (adaptive range **40–120 ms**). Frames carry capture timestamps in global time; buffer aligns and de‑jitters (see §10.5).

### 8.4 Pause/Seek Semantics

* **Gameplay:** **No user‑initiated pause or seek** during singing. If the OS suspends, the session auto‑pauses and resumes with a **50 ms** audio pre‑roll; scoreboard is preserved.
* **Pre‑roll & countdown:** At song start, a 3‑2‑1 countdown is rendered from the **global clock**; audio starts on tick 0.
* **Previews:** Song‑select preview player may seek freely; unrelated to scoring.

### 8.5 Determinism

* Given identical inputs/settings/calibration, the engine produces identical **pitch frames, matches, and scores**. All rounding rules are fixed as above.

---

## 9) Audio Subsystem (Playback & Mixing)

### 9.1 Goals & Scope

* Deterministic playback aligned to §8 (10 ms ticks @ 48 kHz).
* Format unification: all audio rendered as stereo float32, 48 000 Hz.
* Separation of concerns: song backing vs. UI/SFX are isolated in fixed buses.
* No monitoring/sidetone: microphone signals are never routed to speakers. HUD provides feedback.

### 9.2 Internal Graph (fixed order, per 10 ms tick = 480 samples)

1. SongDecoder → Resampler (backing audio from §4/§5 asset)
2. SfxPlayer (polyphonic SFX)
3. SongAudioBus (apply gain/ramp, meters)
4. SfxBus (apply gain/ramp, meters)
5. Sum2 (song + sfx)
6. Headroom (−6 dB fixed scalar)
7. MasterLimiter (brick-wall with look-ahead)
8. OutputSink (device I/O)

No other nodes exist in v1 (no EQ, no reverb, no vocal removal).

### 9.3 Formats & Blocks

* Internal mix format: stereo, interleaved float32, 48 kHz.
* Block size: all producers/consumers operate in 480-sample blocks (10 ms).
* Device mismatch: if the platform cannot supply a multiple of 480, an internal ring-buffer assembles 480-sample blocks for the graph and adapts to the device buffer on output.

### 9.4 Buses (definitions + contract)

#### Common bus contract (applies to both buses)

* Gain range: \[−60 dB, +6 dB], step 0.5 dB.
* Ramping: gain changes are linearly ramped over 50 ms (5 ticks).
* Meters per tick:

  * peak = max(|L|,|R|) over the tick;
  * rms = sqrt(mean(L²+R²)/2) over a 300 ms sliding window.
* Headroom expectation: downstream chain assumes −6 dBFS headroom before the limiter.

```ts
interface AudioBus {
  setGain(dB: number): void;        // clamp to [-60,+6], ramp over 50 ms
  setMuted(muted: boolean): void;   // mutes after ramp
  getMeters(): { peak: number; rms: number };
}
```

#### SongAudioBus (backing track)

* Source: the single audio file referenced by #AUDIO/#MP3 (see §4, §5).
* Trim: applies #START (sec) and #END (ms). Outside this window the bus outputs silence.
* State: Idle → Prerolled → Playing → Draining → Idle. Playback begins only once at least 1 tick of decoded data is queued.
* Underrun: if the decoder cannot supply a full tick, emit silence for that tick and log WARN\_AUDIO\_UNDERRUN.
* Defaults: gain 0 dB.

#### SfxBus (UI/score sounds)

* Assets: WAV PCM16 or OGG; decoded to float32 at load; resampled to 48 kHz.
* Polyphony cap: 8 simultaneous voices (MAX\_SFX\_VOICES=8).
* Scheduling: playSfx(id) triggers on the next tick boundary. Sub-tick requests are rounded to nearest, ties-to-even (consistent with §8.2).
* Overflow policy: if a new voice would exceed the cap, drop the newest and log WARN\_SFX\_DROPPED.
* No ducking: SFX never auto-reduces the song level in v1.
* Defaults: gain −10 dB.
* Cache: decoded SFX MRU cache 8 MB; LRU eviction when exceeded.

### 9.5 Decoding & Resampling

* Decoders: per §5 (MP3/OGG/WAV), producing float32 at source rate.
* Resampler: polyphase windowed-sinc, linear-phase. Implementation may vary, but MUST meet:

  * Passband ripple ≤ 0.05 dB (0–20 kHz @ 48 kHz out).
  * Stopband attenuation ≥ 100 dB.
  * Added algorithmic latency ≤ 10 ms (≤ 1 tick).
* Arbitrary ratios: support 44.1→48 kHz and any 8–192 kHz inputs (§5).

### 9.6 Headroom & Limiting

* Fixed headroom: apply −6 dB scalar before the limiter to prevent incidental clipping on summation.
* MasterLimiter: single-band brick-wall with look-ahead.

  * Look-ahead 2 ms, attack 1 ms, release 100 ms.
  * Threshold −1 dBFS, hard ceiling −0.1 dBFS.
  * Determinism: fixed coefficients/tables; channel-linked; no noise-shaping.
* Telemetry: if limiter sits at ceiling > 100 ms continuously, log WARN\_MASTER\_LIMITING.

### 9.7 Device Selection, Routing & Hot-swap

* Output priority (TV/embedded): 1) HDMI/eARC, 2) USB Audio, 3) Analog/Headphone, 4) Bluetooth A2DP, 5) Internal speaker.
* Desktop: use OS default unless the user selects a device in-app; persist per machine.
* Attach prompts: on attach of a higher-priority device, prompt “Switch to ?”. Default Yes; remember choice until the next attach.
* Channel layout: always render stereo; downmix multichannel sinks (ITU-R BS.775) before §9.6.

### 9.8 Buffers, Latency & Scheduling

* Latency targets (steady-state): mixer→speaker ≤ 40 ms (Tier-A), ≤ 60 ms (Tier-B).
* Hardware buffer: pick nearest multiple of 480 samples; if none, choose ≥ 240 and adapt via internal ring-buffer.
* Queue depth: keep 2–3 hardware bursts queued while meeting the targets above.
* Deadline discipline: render thread must have the next hardware buffer ready ≥ 2 ms before deadline. Missed deadline = underrun handling in §9.9.

### 9.9 Fault Handling & Diagnostics

* Underrun order: drop SFX first; if song underruns, emit silence for that tick; log WARN\_AUDIO\_UNDERRUN.
* Device loss/hot-swap: pause render, rebuild I/O, resume with 50 ms pre-roll (audio only). Global master clock (§8.1) continues—scoring unaffected.
* Diagnostic fields: { timeEpochMs, event, deviceId?, bufferIndex?, details } written to diagnostics per §7.13.

### 9.10 Public Control Surface (host engine API)

```ts
enum AudioBusId { Song, Sfx }

interface AudioEngine {
  setBusGain(bus: AudioBusId, dB: number): void;   // clamps & ramps per §9.4
  setMuted(bus: AudioBusId, muted: boolean): void;
  playSfx(id: string): void;                       // schedules on next tick boundary
  setOutputDevice(deviceId: string | null): void;  // null = system default
  getLatencyMs(): number;                          // mixer→speaker estimate (steady-state)
  on(event: 'underrun'|'deviceChange'|'limiting', cb: (info:any)=>void): void;
}
```

### 9.11 Acceptance Tests (minimum)

* Resampler: swept-sine 20 Hz–20 kHz → THD+N ≤ −90 dB; passband/stopband within §9.5 limits.
* Latency: tone→speaker median ≤ 40 ms on Tier-A across 50 trials; no single trial > 50 ms.
* Stability under load: 10 min playback with CPU stress → 0 song underruns; SFX drop rate ≤ 0.1%.
* Limiter ceiling: +12 dB step input above threshold → output never exceeds −0.1 dBFS; recovery ≤ 150 ms after step removal.
* Hot-swap: attach/detach across all priority classes resumes audio within 500 ms with 50 ms pre-roll, no crash.

## 10) Microphone Input & Device Management

### 10.1 Sources

* **USB class‑compliant mics** (mono/stereo) on host device.
* **Built‑in mics** on host, if present.
* **Wi‑Fi microphones** via phones (companion app) over LAN (§10.4). Pairing shares ingress trust (Appendix A).

### 10.2 Capture Format & Chain

* **Capture:** PCM **16‑bit mono @ 48 kHz** per mic.
* **Per‑mic chain:** DC‑block → high‑pass 80 Hz → noise gate (−50 dBFS, 10 ms attack, 50 ms release) → **pitch detection** (§11). **AGC disabled**.
* **Clipping:** detect at −0.5 dBFS over 5 consecutive samples → `WARN_CLIPPING`.

### 10.3 Enumeration & Routing

* Enumerate devices at launch and on hot‑plug. Each mic maps to **P1/P2** lanes according to game mode (§13); if >2 mics, multiple players may share a lane (round‑robin in Party mode).

### 10.4 Wi‑Fi Microphone Transport

* **Protocol:** UDP over LAN with a simple RTP‑like header.
* **Packet cadence:** one packet per **10 ms** containing **480 samples**.
* **Header:** `{ seq, captureTimeUs (global clock), micId, flags }`.
* **Loss handling:** up to **2%** random loss concealed by linear interpolation; burst loss > **40 ms** mutes until buffer realigns.
* **Security:** paired devices only (same pairing as ingress, Appendix A); packets from unpaired devices are dropped.

### 10.5 Jitter Buffer & Alignment

* Target **80 ms** latency; adaptive **40–120 ms**. Align by `captureTimeUs`. Apply **drift slewing** ≤ ±1 ms/s to maintain lock (see §8.3).

### 10.6 Calibration

* **Beep‑&‑clap** flow per mic: play a test tone on host → phone records → phone plays clap tone locally → host measures round‑trip and stores **per‑device offset**.
* Quick recheck on session start; if skipped, offset = **0 ms**.

### 10.7 Capability Gating

* Support up to **4** concurrent mics. Run the **Mic Capacity Test** (10 s, real pipeline) at N=4→3→2; accept the highest N meeting: median input→pitch ≤ **60 ms** and ≤ **10% CPU per mic**, total audio+DSP ≤ **50% CPU** (see Standards: performance budgets).

### 10.8 Errors & Diagnostics

* Codes: `MIC_DEVICE_LOST`, `MIC_UNSUPPORTED_FORMAT`, `MIC_BUFFER_OVERRUN`, `WIFI_MIC_UNPAIRED`, `WIFI_MIC_LATE_FRAMES`, `WIFI_MIC_RATE_MISMATCH`.
* All mic warnings/errors are written to diagnostics with timestamps and micId.

---

## 11) Pitch Detection

### 11.1 Output contract

* Frame cadence: one frame every **10 ms** (hop = **480 samples** @ 48 kHz; aligned to §8 ticks).
* Timestamp: `timeUs` is the **center** of the analysis window, expressed in the **global master clock** (see §8.1).
* Frame schema:

  ```ts
  type PitchFrame = {
    timeUs:    uint64;   // global time of the window center
    midi:      int32;    // MIDI note number (C4 = 60)
    cents:     float32;  // deviation from midi, range [-50, +50)
    confidence:float32;  // [0,1], higher = more periodic/voiced
  }
  ```
* Stream interface:

  ```ts
  interface PitchDetector {
    start(micId: string): AsyncIterable<PitchFrame>;
    stop(micId: string): void;
  }
  ```

### 11.2 Input & pre-processing

* Input is the **per-mic** PCM stream **48 kHz, 16-bit mono** after §10.2 processing:

  * DC-block → 80 Hz high-pass → noise gate (−50 dBFS, 10 ms attack, 50 ms release).
* The detector operates on **float32** samples in **stereo-summed→mono** form (if a mic is stereo).

### 11.3 Detector algorithm

* Method: **YIN** with the cumulative mean normalized difference function (CMNDF) and parabolic interpolation around the selected lag.
* Window: **30 ms** Hann (1440 samples). Overlap implied by the 10 ms hop.
* Search range: **60–1500 Hz** → lag `τ ∈ [32, 800]` samples at 48 kHz.
* Procedure (per frame):

  1. Take the 30 ms Hann-windowed block centered at `timeUs`.
  2. Compute YIN difference `d(τ)` and `CMNDF(τ)`.
  3. Find the **first local minimum** of `CMNDF(τ)` below threshold (see 11.4). If none found, pick the global minimum.
  4. **Parabolic interpolate** around that `τ̂` to obtain sub-sample lag `τ*`.
  5. Convert to frequency `f0 = 48_000 / τ*` (Hz).

### 11.4 Voicing & confidence

* Confidence: `confidence = clamp01(1 − CMNDF(τ̂))`.
* A frame is voiced iff:

  * `confidence ≥ 0.60`, and
  * frame RMS (pre-gate) ≥ **−45 dBFS**.
* Unvoiced frames: emit `confidence = 0` and carry forward the previous `midi/cents` values (see 11.6). Scoring interprets `confidence = 0` as no hit.

### 11.5 Hz → MIDI/cents mapping

```
m_real = 69 + 12 * log2(f0 / 440)
midi   = round(m_real)
cents  = 1200 * log2( f0 / (440 * 2^((midi - 69)/12)) )
```

* Clamp `cents` into **\[−50, +50)** by folding ±1200-cent jumps as needed.

### 11.6 Temporal smoothing, hysteresis, octave correction

* Median-of-3 over the last **voiced** `midi` estimates (in semitones).
* Hysteresis: ignore changes **< ±20 cents** from the previous output unless **three consecutive** frames agree; then adopt the new value.
* Octave correction: if the jump from the previous voiced `midi` exceeds **±700 cents**, also evaluate `±1200`-cent folds and keep the result with the **smallest absolute cents**. Apply at most **one** fold per frame.

### 11.7 Timestamps & latency

* `timeUs` is `windowStart + windowLength/2` in the global clock.
* Algorithmic latency = **15 ms** (window center). End-to-end budgets are defined in §8 and §10.

### 11.8 Numeric rules

* Internal math in **float32**; use `log2f` and exact constants above.
* All rounding is **nearest, ties-to-even** (for `round` and any time/beat quantization).
* Deterministic results across platforms are required for identical input streams.

### 11.9 CPU & concurrency budgets

* Per-mic CPU: **≤ 10%** of one big core on Tier-A devices under load.
* Vectorization (SSE2/NEON) is required for the YIN inner loop.

### 11.10 Out-of-range & degenerate cases

* If `τ*` falls outside `[32, 800]`, mark **unvoiced** (`confidence = 0`).
* If the window is all zeros/silence, mark **unvoiced**.
* If interpolation is ill-conditioned (peak plateau), use integer `τ̂` without interpolation.

### 11.11 Tests

* Sine sweep 60–1500 Hz: median absolute error **≤ 5 cents**; octave-error rate **≤ 0.5%**.
* Noisy tone (−20 dB SNR): voiced precision **≥ 95%** at the 0.60 threshold.
* Glissando 200→400 Hz: reported latency within **15 ms** and no more than **one** octave fold during the glide.
* Repeatability: identical input buffers produce byte-identical `PitchFrame` sequences on all platforms.

## 12) Note Matching & Scoring (USDX‑faithful)

* Bridges `PitchFrame` streams to the USDX notes grammar from §4 to produce scores identical to USDX.
* Defines:

  * Pitch quantization (how MIDI/cents map to a note’s expected pitch).
  * Timing windows and tolerances around note starts/ends.
  * Treatment of note types: normal (`:`), golden (`*`), freestyle (`F`), rap (`R`), rap‑golden (`G`).
  * Line bonuses, streak/combo, golden multipliers, duet part assignment.
  * Rounding and precision rules so final totals match USDX.
* Interfaces:

  * `NoteMatcher.match(pitch: stream<PitchFrame>, song, playerId) → MatchResult`
  * `ScoringEngine.score(match: MatchResult, difficulty) → ScoreSummary`
* Output data:

  * `MatchResult` holds per‑note hit/miss with timing and pitch deltas.
  * `ScoreSummary` holds per‑line and total scores; used by HUD and results screen.
