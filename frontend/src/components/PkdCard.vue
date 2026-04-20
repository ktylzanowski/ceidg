<template>
  <div class="card">
    <div class="card-header">
      <span class="badge" :class="main ? 'badge-main' : 'badge-additional'">
        {{ main ? 'Kod główny' : 'Dodatkowy' }}
      </span>
      <span class="code">{{ pkd.code }}</span>
    </div>
    <p class="description">{{ pkd.description }}</p>
    <p class="justification">{{ pkd.justification }}</p>
  </div>
</template>

<script>
export default {
  name: 'PkdCard',
  props: {
    pkd: { type: Object, required: true },
    main: { type: Boolean, default: false },
  },
}
</script>

<style scoped>
.card {
  background: #fff;
  border: 1.5px solid #e5e7eb;
  border-radius: 12px;
  padding: 20px 22px;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.badge {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.badge-main {
  background: #ede9fe;
  color: #4f46e5;
}

.badge-additional {
  background: #f0fdf4;
  color: #16a34a;
}

.code {
  font-size: 1.1rem;
  font-weight: 700;
  font-family: monospace;
  color: #1a1a2e;
}

.description {
  font-size: 0.97rem;
  color: #374151;
  margin-bottom: 8px;
}

.justification {
  font-size: 0.87rem;
  color: #6b7280;
  border-left: 3px solid #e5e7eb;
  padding-left: 10px;
}
</style>
