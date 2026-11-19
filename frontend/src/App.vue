<template>
  <div class="page">
    <div class="container">
      <h1>PKD Recommender</h1>
      <p class="subtitle">Opisz swoją działalność, a system dobierze odpowiednie kody PKD.</p>

      <div class="form">
        <label for="desc">Czym chcesz się zajmować w firmie?</label>
        <textarea
          id="desc"
          v-model="description"
          :disabled="loading"
          :class="{ invalid: validationError }"
          placeholder="np. Prowadzę sklep internetowy z elektroniką, zajmuję się sprzedażą i serwisem laptopów."
          rows="4"
          maxlength="1000"
          @input="validationError = false; error = null"
        />
        <button @click="recommend" :disabled="loading || !description.trim()">
          <span v-if="loading" class="spinner" />
          {{ loading ? 'Analizuję...' : 'Znajdź kody PKD' }}
        </button>
      </div>

      <div v-if="error" class="error">
        <template v-if="validationError">
          <strong>Opis wymaga poprawki</strong><br>
          Spróbuj opisać konkretną działalność gospodarczą, np. <em>„sprzedaż odzieży przez internet"</em> lub <em>„usługi budowlane i wykończeniowe"</em>.
        </template>
        <template v-else-if="rateLimitError">
          Zbyt wiele zapytań. Odczekaj chwilę i spróbuj ponownie.
        </template>
        <template v-else-if="serverMessage">
          {{ serverMessage }} Spróbuj ponownie za chwilę.
        </template>
        <template v-else>
          Coś poszło nie tak. Spróbuj ponownie za chwilę.
        </template>
      </div>

      <div v-if="result" class="results">
        <div v-if="result.low_confidence" class="warning">
          <strong>Niepewne dopasowanie</strong><br>
          Nie znaleźliśmy kodów, które wyraźnie pasują do opisu. Poniższe propozycje traktuj ostrożnie —
          spróbuj opisać działalność bardziej szczegółowo albo zweryfikuj kody w oficjalnej klasyfikacji PKD.
        </div>
        <PkdCard :pkd="result.main" :main="true" />
        <PkdCard v-for="pkd in result.additional" :key="pkd.code" :pkd="pkd" />
      </div>
    </div>
  </div>
</template>

<script>
import PkdCard from './components/PkdCard.vue'

const API_URL = '/api/recommend'

export default {
  name: 'App',
  components: { PkdCard },
  data() {
    return {
      description: '',
      loading: false,
      result: null,
      error: null,
      validationError: false,
      rateLimitError: false,
      serverMessage: null,
    }
  },
  methods: {
    async recommend() {
      this.error = null
      this.result = null
      this.loading = true
      this.rateLimitError = false
      this.serverMessage = null
      try {
        const res = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ main_description: this.description }),
        })
        if (!res.ok) {
          const data = await res.json().catch(() => ({}))
          if (data.detail?.code === 'invalid_description') {
            this.validationError = true
          } else if (res.status === 429) {
            this.rateLimitError = true
          } else if (typeof data.detail?.message === 'string') {
            // Komunikat z backendu, np. timeout albo niedostępny model
            this.serverMessage = data.detail.message
          }
          throw new Error(data.detail?.message || `Błąd serwera: ${res.status}`)
        }
        this.result = await res.json()
      } catch (e) {
        this.error = e.message
      } finally {
        this.loading = false
      }
    },
  },
}
</script>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  padding: 48px 16px;
}

.container {
  width: 100%;
  max-width: 720px;
}

h1 {
  font-size: 1.75rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.subtitle {
  color: #666;
  margin-bottom: 32px;
  font-size: 0.95rem;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

label {
  font-weight: 600;
  font-size: 0.95rem;
}

textarea {
  width: 100%;
  padding: 12px 14px;
  border: 1.5px solid #d0d5dd;
  border-radius: 8px;
  font-size: 1rem;
  font-family: inherit;
  resize: vertical;
  transition: border-color 0.2s;
  background: #fff;
}

textarea:focus {
  outline: none;
  border-color: #4f46e5;
}

textarea.invalid {
  border-color: #f87171;
}

button {
  align-self: flex-start;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 11px 28px;
  background: #4f46e5;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s, opacity 0.2s;
}

button:hover:not(:disabled) { background: #4338ca; }
button:disabled { opacity: 0.6; cursor: not-allowed; }

.spinner {
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255, 255, 255, 0.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes spin { to { transform: rotate(360deg); } }

.error {
  margin-top: 20px;
  padding: 12px 16px;
  background: #fef2f2;
  border: 1px solid #fca5a5;
  border-radius: 8px;
  color: #b91c1c;
  font-size: 0.9rem;
}

.warning {
  padding: 12px 16px;
  background: #fffbeb;
  border: 1px solid #fcd34d;
  border-radius: 8px;
  color: #92400e;
  font-size: 0.9rem;
}

.results {
  margin-top: 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
