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
          placeholder="np. Prowadzę sklep internetowy z elektroniką, zajmuję się sprzedażą i serwisem laptopów."
          rows="4"
        />
        <button @click="recommend" :disabled="loading || !description.trim()">
          <span v-if="loading" class="spinner" />
          {{ loading ? 'Analizuję...' : 'Znajdź kody PKD' }}
        </button>
      </div>

      <div v-if="error" class="error">{{ error }}</div>

      <div v-if="result" class="results">
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
    }
  },
  methods: {
    async recommend() {
      this.error = null
      this.result = null
      this.loading = true
      try {
        const res = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ main_description: this.description }),
        })
        if (!res.ok) {
          const data = await res.json()
          throw new Error(data.detail || `Błąd serwera: ${res.status}`)
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

.results {
  margin-top: 32px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
