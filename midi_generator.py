"""
midi_generator.py
-----------------
Gerador Musical MIDI — Componente do João (J)
Projeto: Adaptive Music Experience (C2)

Responsabilidades:
  - Geração de melodias, ritmos e progressões harmónicas em formato MIDI
  - Geração probabilística por mood (happy, sad, calm, energética)
  - Parâmetros expostos: BPM, densidade, repetição, complexidade
  - Exportação de ficheiros .mid para o backend servir ao frontend

Dependências:
  pip install pretty_midi numpy
"""

import pretty_midi
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuração de Moods
# ---------------------------------------------------------------------------

MOOD_CONFIGS = {
    "happy": {
        "scale": "major",
        "bpm_range": (120, 160),
        "density": 0.8,        # notas por beat (0-1)
        "complexity": 0.7,     # complexidade rítmica (0-1)
        "octave_range": (4, 6),
        "note_duration_range": (0.1, 0.4),
        "velocity_range": (80, 110),
    },
    "sad": {
        "scale": "minor",
        "bpm_range": (60, 90),
        "density": 0.4,
        "complexity": 0.3,
        "octave_range": (3, 5),
        "note_duration_range": (0.3, 0.8),
        "velocity_range": (40, 70),
    },
    "calm": {
        "scale": "major",
        "bpm_range": (70, 100),
        "density": 0.3,
        "complexity": 0.2,
        "octave_range": (4, 5),
        "note_duration_range": (0.4, 1.0),
        "velocity_range": (45, 75),
    },
    "energetic": {
        "scale": "major",
        "bpm_range": (150, 180),
        "density": 0.9,
        "complexity": 0.9,
        "octave_range": (4, 6),
        "note_duration_range": (0.05, 0.25),
        "velocity_range": (90, 127),
    },
}

# Escalas: intervalos em semitons a partir da tónica
SCALES = {
    "major":        [0, 2, 4, 5, 7, 9, 11],
    "minor":        [0, 2, 3, 5, 7, 8, 10],
    "pentatonic":   [0, 2, 4, 7, 9],
    "blues":        [0, 3, 5, 6, 7, 10],
}

# Progressões harmónicas (graus da escala) por mood
CHORD_PROGRESSIONS = {
    "happy":    [[0, 4, 5, 3], [0, 5, 3, 4]],    # I-V-VI-IV, I-VI-IV-V
    "sad":    [[5, 3, 0, 4], [0, 6, 3, 7]],     # VI-IV-I-V (menor)
    "calm":     [[0, 3, 4, 3], [0, 4, 3, 0]],
    "energetic":[[0, 4, 5, 4], [0, 5, 0, 4]],
}

# Notas raiz disponíveis (C=60, D=62, ...)
ROOT_NOTES = {
    "C": 60, "D": 62, "E": 64, "F": 65,
    "G": 67, "A": 69, "B": 71,
}


# ---------------------------------------------------------------------------
# Parâmetros de geração (interface com BO e Bandit)
# ---------------------------------------------------------------------------

@dataclass
class GeneratorParams:
    """
    Parâmetros expostos para o sistema de otimização.
    A Otimização Bayesiana (L) vai ajustar estes valores.
    O Bandit (L) escolhe o mood.
    """
    mood: str = "happy"
    bpm: float = 120.0           # 60 – 180
    density: float = 0.6         # 0.1 – 1.0 (notas por beat)
    complexity: float = 0.5      # 0.0 – 1.0 (variação rítmica)
    repetition: float = 0.3      # 0.0 – 1.0 (0 = sem repetição, 1 = muito repetitivo)
    duration_bars: int = 8       # duração em compassos
    root_note: str = "C"         # tónica
    instrument_program: int = 0  # General MIDI program (0=piano, 25=guitar, 40=violin…)


# ---------------------------------------------------------------------------
# Gerador de escala e acordes
# ---------------------------------------------------------------------------

def get_scale_notes(root_midi: int, scale_name: str, octave_range: tuple) -> list[int]:
    """Devolve todas as notas MIDI da escala no intervalo de oitavas dado."""
    intervals = SCALES.get(scale_name, SCALES["major"])
    notes = []
    for oct in range(octave_range[0], octave_range[1] + 1):
        base = root_midi + (oct - 4) * 12  # ajusta oitava relativa ao MIDI 60
        for interval in intervals:
            note = base + interval
            if 0 <= note <= 127:
                notes.append(note)
    return sorted(notes)


def build_chord(root_midi: int, scale_name: str, degree: int, octave: int = 4) -> list[int]:
    """Constrói um acorde de 3 notas (tríade) a partir do grau da escala."""
    intervals = SCALES.get(scale_name, SCALES["major"])
    n = len(intervals)
    chord_notes = []
    for step in [0, 2, 4]:  # 1ª, 3ª, 5ª
        idx = (degree + step) % n
        extra_octave = (degree + step) // n
        note = root_midi + (octave - 4) * 12 + intervals[idx] + extra_octave * 12
        if 0 <= note <= 127:
            chord_notes.append(note)
    return chord_notes


# ---------------------------------------------------------------------------
# Gerador principal
# ---------------------------------------------------------------------------

class MIDIGenerator:
    """
    Gera ficheiros MIDI adaptativos com base em mood e parâmetros contínuos.

    Uso básico:
        gen = MIDIGenerator()
        path = gen.generate(params, output_path="musica.mid")
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate(self, params: GeneratorParams, output_path: str = "output.mid") -> str:
        """
        Gera um ficheiro MIDI completo e guarda-o em output_path.
        Devolve o caminho do ficheiro criado.
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=params.bpm)
        root_midi = ROOT_NOTES.get(params.root_note, 60)
        mood_cfg = self._resolve_mood_config(params)

        # Faixas: melodia principal + acompanhamento harmónico + bateria
        melody_track = pretty_midi.Instrument(
            program=params.instrument_program, name="Melody"
        )
        harmony_track = pretty_midi.Instrument(
            program=params.instrument_program, name="Harmony", is_drum=False
        )
        drum_track = pretty_midi.Instrument(
            program=0, name="Drums", is_drum=True
        )

        seconds_per_beat = 60.0 / params.bpm
        beats_per_bar = 4
        total_beats = params.duration_bars * beats_per_bar

        self._generate_melody(
            melody_track, params, mood_cfg, root_midi,
            total_beats, seconds_per_beat
        )
        self._generate_harmony(
            harmony_track, params, mood_cfg, root_midi,
            total_beats, seconds_per_beat, beats_per_bar
        )
        self._generate_drums(
            drum_track, params, total_beats, seconds_per_beat
        )

        midi.instruments.extend([melody_track, harmony_track, drum_track])
        midi.write(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Geração de melodia
    # ------------------------------------------------------------------

    def _generate_melody(
        self, track, params, mood_cfg, root_midi,
        total_beats, spb
    ):
        scale_notes = get_scale_notes(
            root_midi, mood_cfg["scale"], mood_cfg["octave_range"]
        )

        # Motivo base: sequência curta reutilizada (controla repetição)
        motif_length = max(4, int(8 * (1 - params.repetition)))
        motif = self.rng.choice(scale_notes, size=motif_length).tolist()

        beat = 0.0
        note_dur_min, note_dur_max = mood_cfg["note_duration_range"]
        vel_min, vel_max = mood_cfg["velocity_range"]

        while beat < total_beats:
            # Decide se toca nota ou pausa com base na densidade
            if self.rng.random() < params.density:
                # Escolhe nota: com probabilidade `repetition` usa o motivo
                if self.rng.random() < params.repetition and motif:
                    pitch = int(self.rng.choice(motif))
                else:
                    pitch = int(self.rng.choice(scale_notes))

                # Duração com variação rítmica controlada por complexity
                base_dur = self.rng.uniform(note_dur_min, note_dur_max)
                if params.complexity > 0.5:
                    # Quantiza a durações mais irregulares
                    base_dur *= self.rng.choice([0.5, 1.0, 1.5, 2.0],
                                                p=[0.3, 0.4, 0.2, 0.1])

                duration = max(0.05, base_dur) * spb
                velocity = int(self.rng.integers(vel_min, vel_max))

                start_time = beat * spb
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=start_time,
                    end=start_time + duration,
                )
                track.notes.append(note)

            # Avanço rítmico: steps menores = mais complexidade
            if params.complexity > 0.6:
                step = self.rng.choice([0.25, 0.5, 0.75, 1.0],
                                       p=[0.4, 0.3, 0.2, 0.1])
            else:
                step = self.rng.choice([0.5, 1.0], p=[0.4, 0.6])

            beat += step

    # ------------------------------------------------------------------
    # Geração de harmonia (acordes)
    # ------------------------------------------------------------------

    def _generate_harmony(
        self, track, params, mood_cfg, root_midi,
        total_beats, spb, beats_per_bar
    ):
        progressions = CHORD_PROGRESSIONS.get(params.mood, [[0, 4, 5, 3]])
        progression = self.rng.choice(progressions)  # escolhe uma progressão
        scale_name = mood_cfg["scale"]
        vel_min, vel_max = mood_cfg["velocity_range"]
        harmony_vel = int((vel_min + vel_max) / 2 * 0.7)  # mais suave que melodia

        bar = 0
        beat = 0.0
        prog_idx = 0

        while beat < total_beats:
            degree = progression[prog_idx % len(progression)]
            chord_notes = build_chord(root_midi, scale_name, degree, octave=3)
            chord_dur = beats_per_bar * spb  # 1 acorde por compasso

            for pitch in chord_notes:
                note = pretty_midi.Note(
                    velocity=harmony_vel,
                    pitch=pitch,
                    start=beat * spb,
                    end=beat * spb + chord_dur * 0.95,
                )
                track.notes.append(note)

            beat += beats_per_bar
            prog_idx += 1
            bar += 1

    # ------------------------------------------------------------------
    # Geração de bateria
    # ------------------------------------------------------------------

    def _generate_drums(self, track, params, total_beats, spb):
        """
        Padrão rítmico simples mas parametrizado.
        GM drum map: kick=36, snare=38, hi-hat=42, open-hat=46
        """
        KICK, SNARE, HIHAT, OPEN_HAT = 36, 38, 42, 46

        beat = 0.0
        step = 0.5  # resolução de semicolcheias

        while beat < total_beats:
            beat_in_bar = beat % 4

            # Kick: beats 1 e 3 (sempre)
            if abs(beat_in_bar % 2) < 0.01:
                self._add_drum_note(track, KICK, beat * spb, spb * 0.4, 100)

            # Snare: beats 2 e 4
            if abs(beat_in_bar - 1) < 0.01 or abs(beat_in_bar - 3) < 0.01:
                self._add_drum_note(track, SNARE, beat * spb, spb * 0.3, 90)

            # Hi-hat: baseado na densidade e complexidade
            if self.rng.random() < params.density * 0.8:
                vel_hat = int(60 + params.complexity * 40)
                drum = OPEN_HAT if (params.complexity > 0.7
                                    and self.rng.random() < 0.2) else HIHAT
                self._add_drum_note(track, drum, beat * spb, spb * 0.2, vel_hat)

            beat += step

    def _add_drum_note(self, track, pitch, start, duration, velocity):
        note = pretty_midi.Note(
            velocity=min(127, velocity),
            pitch=pitch,
            start=start,
            end=start + duration,
        )
        track.notes.append(note)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_mood_config(self, params: GeneratorParams) -> dict:
        """
        Combina a configuração base do mood com os parâmetros contínuos
        passados pelo sistema de BO/Bandit.
        """
        base = MOOD_CONFIGS.get(params.mood, MOOD_CONFIGS["happy"]).copy()
        # Sobrepõe com parâmetros explícitos (vindos de BO)
        base["density"] = params.density
        base["complexity"] = params.complexity
        return base


# ---------------------------------------------------------------------------
# Geração de batch — 5 músicas por mood (20 no total)
# ---------------------------------------------------------------------------

# Tónicas usadas para variar as 5 versões de cada mood
_VARIATION_ROOTS = ["C", "D", "E", "G", "A"]

# Instrumentos GM por mood (programa MIDI)
_MOOD_INSTRUMENTS = {
    "happy":     [0, 4, 24, 40, 56],   # Piano, E.Piano, Guitar, Violin, Trumpet
    "sad":     [0, 11, 40, 42, 70],  # Piano, Vibraphone, Violin, Cello, Bassoon
    "calm":      [0, 8, 24, 46, 52],   # Piano, Celesta, Guitar, Harp, Choir
    "energetic": [29, 30, 33, 56, 80], # Muted Guitar, Overdriven, E.Bass, Trumpet, Square
}

def generate_initial_batch(
    output_dir: str = "midi_files",
    songs_per_mood: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Gera `songs_per_mood` músicas para cada mood (por omissão: 5 × 4 = 20 faixas).
    Cada variação tem BPM, densidade, complexidade, tónica e instrumento ligeiramente
    diferentes para garantir diversidade suficiente para o cold start do Bandit.

    Devolve lista de dicts com metadados (para o backend do L).

    Uso:
        tracks = generate_initial_batch("midi_files")
        # → 20 dicts, 5 por mood, organizados em subpastas por mood
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    gen = MIDIGenerator(seed=seed)

    moods = list(MOOD_CONFIGS.keys())   # ["happy", "sad", "calm", "energetic"]
    tracks = []
    track_id = 0

    for mood in moods:
        cfg   = MOOD_CONFIGS[mood]
        bpm_lo, bpm_hi = cfg["bpm_range"]

        # Subpasta por mood para organização
        mood_dir = os.path.join(output_dir, mood)
        os.makedirs(mood_dir, exist_ok=True)

        print(f"\n  [{mood.upper()}]")

        for v in range(songs_per_mood):
            # Varia os parâmetros contínuos dentro do range do mood
            bpm        = float(rng.uniform(bpm_lo, bpm_hi))
            density    = float(np.clip(rng.normal(cfg["density"],    0.12), 0.1, 1.0))
            complexity = float(np.clip(rng.normal(cfg["complexity"], 0.12), 0.0, 1.0))
            repetition = float(rng.uniform(0.2, 0.5))
            root       = _VARIATION_ROOTS[v % len(_VARIATION_ROOTS)]
            instrument = _MOOD_INSTRUMENTS[mood][v % len(_MOOD_INSTRUMENTS[mood])]
            duration_bars = int(rng.choice([8, 12, 16]))

            params = GeneratorParams(
                mood=mood,
                bpm=bpm,
                density=density,
                complexity=complexity,
                repetition=repetition,
                duration_bars=duration_bars,
                root_note=root,
                instrument_program=instrument,
            )

            filename = f"{mood}_v{v + 1}_{root}_{int(bpm)}bpm.mid"
            path     = os.path.join(mood_dir, filename)
            gen.generate(params, output_path=path)

            entry = {
                "id":            track_id,
                "path":          path,
                "mood":          mood,
                "bpm":           round(bpm, 1),
                "density":       round(density, 2),
                "complexity":    round(complexity, 2),
                "repetition":    round(repetition, 2),
                "duration_bars": duration_bars,
                "root_note":     root,
                "instrument":    instrument,
            }
            tracks.append(entry)
            print(f"    [✓] v{v + 1}  {filename}"
                  f"  BPM={bpm:.0f}  dens={density:.2f}  cmplx={complexity:.2f}")
            track_id += 1

    return tracks


# ---------------------------------------------------------------------------
# Demo / teste rápido
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Teste do Gerador MIDI ===\n")
    
    # 1. Teste unitário por mood
    gen = MIDIGenerator(seed=0)
    for mood in ["happy", "sad", "calm", "energetic"]:
        cfg = MOOD_CONFIGS[mood]
        params = GeneratorParams(
            mood=mood,
            bpm=float(np.mean(cfg["bpm_range"])),
            density=cfg["density"],
            complexity=cfg["complexity"],
        )
        path = gen.generate(params, output_path=f"test_{mood}.mid")
        print(f"[✓] {mood:12s} → {path}")

    print()

    # 2. Teste com parâmetros customizados (como viria da BO)
    custom_params = GeneratorParams(
        mood="energetic",
        bpm=165.0,
        density=0.85,
        complexity=0.9,
        repetition=0.2,
        duration_bars=16,
        root_note="G",
        instrument_program=25,  # Guitar
    )
    path = gen.generate(custom_params, output_path="test_custom_bo.mid")
    print(f"[✓] Parâmetros BO customizados → {path}\n")

    # 3. Batch inicial (cold start) — 5 músicas por mood
    print("A gerar batch inicial (5 por mood)...")
    tracks = generate_initial_batch("midi_files_test", songs_per_mood=5)
    print(f"\n[✓] {len(tracks)} faixas geradas no total.")
    print("\nResumo por mood:")
    for mood in ["happy", "sad", "calm", "energetic"]:
        subset = [t for t in tracks if t["mood"] == mood]
        bpms   = [t["bpm"] for t in subset]
        print(f"  {mood:12s} → {len(subset)} faixas"
              f"  BPM=[{min(bpms):.0f}\u2013{max(bpms):.0f}]")