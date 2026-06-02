# Adaptive Music — Backend

API em FastAPI para recolha de sinais e geração de faixas musicais.

## Arrancar o servidor

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Documentação interativa disponível em: `http://localhost:8000/docs`

---

## Contrato da API (para o grupo)

#### Gerar uma faixa
```
POST /tracks/generate
```
```json
{
  "mood": "Calmo",
  "bpm": 75,        // opcional
  "density": 0.4    // opcional, 0.0–1.0
}
```
Resposta:
```json
{
  "track_id": "ABC123",
  "mood": "Calmo",
  "bpm": 75,
  "density": 0.4,
  "name": "Coastal Echo"
}
```

#### Enviar feedback do utilizador
```
POST /signals/feedback
```
```json
{
  "track_id": "ABC123",
  "mood": "Calmo",
  "bpm": 75,
  "feedback": 1       // 1 = 👍, -1 = 👎
}
```

#### Registar mudança de mood
```
POST /signals/mood
```
```json
{
  "mood": "Energético"
}
```

---

O backend vai chamar o teu gerador com:
```
POST http://<teu-host>/generate
{
  "mood": "Calmo",
  "bpm": 75,
  "density": 0.4
}
```
Esperamos receber de volta pelo menos `track_id`, `bpm`, `mood`. Confirma se o formato está ok.