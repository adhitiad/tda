# Aturan dan Konfigurasi Bot Telegram — 5 Grup Chat

## Fokus Utama: Crypto (Tier 1)

Dokumen ini mengatur perilaku bot Telegram di lima grup chat. Prioritas utama adalah crypto; forex dan saham IDX diperluas hanya untuk grup tier Pro dan Enterprise.

---

## 1. Definisi Tier Grup

| Tier  | Deskripsi | Cakupan Asset |
|-------|-----------|---------------|
| **Free**  | Akses dasar, terbatas pada crypto populer atau semua crypto | Crypto saja |
| **Pro**   | Akses lanjutan, crypto + forex atau crypto + IDX | Crypto + Forex atau Crypto + IDX |
| **Enterprise** | Akses lengkap, semua asset | Crypto + Forex + Saham IDX |

---

## 2. Konfigurasi Per Grup

### 2.1 Grup Free — BTC & ETH Only

- **Tier**: Free
- **Asset yang dipantau**: BTC/USDT, ETH/USDT (pasangan spot dan swap)
- **Asset yang diblokir**: Semua altcoin selain BTC dan ETH, forex, saham IDX
- **Routing logic**:
  - Filter incoming signal berdasarkan symbol: hanya `BTC*` dan `ETH*` yang lolos
  - Symbol lain langsung ditolak dengan pesan otomatis: `"Grup ini hanya menerima sinyal BTC dan ETH."`
- **Konten yang boleh diposting**:
  - Sinyal entry/exit BTC dan ETH
  - Analisis teknikal BTC dan ETH (RSI, MACD, Bollinger, ATR)
  - Update harga real-time BTC dan ETH
  - Peringatan risk management untuk BTC dan ETH
- **Konten yang TIDAK boleh diposting**:
  - Altcoin selain BTC dan ETH
  - Pasangan forex
  - Indeks saham
  - Promosi proyek/coin lain
  - Forward pesan dari grup lain (kecuali ditujukan untuk BTC/ETH)
- **Frekuensi pengiriman**: Maks 3 sinyal per jam per asset (BTC/ETH masing-masing dihitung terpisah)
- **Format sinyal**: Lihat Bagian 4 (Format Sinyal Crypto)

### 2.2 Grup Free — All Crypto

- **Tier**: Free
- **Asset yang dipantau**: Semua pasangan crypto (BTC, ETH, altcoin utama, stablecoin excluded)
- **Asset yang diblokir**: Forex, saham IDX
- **Routing logic**:
  - Filter menerima semua symbol crypto yang terdaftar di `config.yaml` → `symbols.crypto`
  - Symbol non-crypto ditolak dengan pesan: `"Grup ini hanya menerima sinyal crypto."`
- **Konten yang boleh diposting**:
  - Sinyal entry/exit untuk semua crypto
  - Analisis teknikal semua crypto
  - Update harga real-time crypto
  - Peringatan risk management crypto
  - Highlight altcoin dengan volume异常 (volume anomaly)
- **Konten yang TIDAK boleh diposting**:
  - Pasangan forex
  - Indeks saham
  - Promosi proyek/coin tanpa data dasar (no fundamental data)
  - Forward pesan dari grup lain
- **Frekuensi pengiriman**: Maks 5 sinyal per jam (aggregate semua crypto)
- **Format sinyal**: Lihat Bagian 4 (Format Sinyal Crypto)

### 2.3 Grup Pro — All Crypto & Forex

- **Tier**: Pro
- **Asset yang dipantau**: Semua crypto + pasangan forex utama (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD, XAU/USD)
- **Routing logic**:
  - Tag asset: `crypto` atau `forex` berdasarkan symbol
  - Crypto: routing ke modul sinyal crypto
  - Forex: routing ke modul sinyal forex (jika tersedia)
  - Symbol yang tidak terdaftar di kedua daftar ditolak
- **Konten yang boleh diposting**:
  - Semua konten Grup Free — All Crypto
  - Sinyal entry/exit forex utama
  - Analisis teknikal forex (support/resistance, trend line, Fibonacci)
  - Update harga real-time forex
  - Analisis korelasi crypto-forex (misal: BTC vs DXY)
  - Peringatan risk management forex
- **Konten yang TIDAK boleh diposting**:
  - Indeks saham
  - Forward pesan dari grup lain
  - Sinyal tanpa timeframe yang jelas
- **Frekuensi pengiriman**: Maks 8 sinyal per jam (crypto + forex aggregate)
  - Sub-limit: maks 3 sinyal forex per jam
  - Sub-limit: maks 5 sinyal crypto per jam
- **Format sinyal**: Lihat Bagian 4 (Format Sinyal Crypto) dan Bagian 5 (Format Sinyal Forex)

### 2.4 Grup Pro — Indo Sham Idx

- **Tier**: Pro
- **Asset yang dipantau**: Indeks saham Indonesia (IDX: IHSG, LQ45, IDX30) + crypto terkait pasar lokal (ADA/IDR, DOT/IDR, dll.)
- **Routing logic**:
  - Symbol IDX: routing ke modul sinyal saham Indonesia
  - Symbol crypto: hanya yang memiliki pasangan IDR atau terdaftar sebagai "local crypto" di config
  - Symbol non-IDX dan non-crypto-IDR ditolak
- **Konten yang boleh diposting**:
  - Sinyal entry/exit indeks IDX (IHSG, LQ45, IDX30)
  - Analisis teknikal indeks IDX
  - Update harga real-time indeks IDX
  - Sinyal crypto pasangan IDR (ADA/IDR, DOT/IDR, dll.)
  - Analisis dampak makroekonomi Indonesia terhadap crypto dan IDX
  - Peringatan risk management IDX dan crypto lokal
- **Konten yang TIDAK boleh diposting**:
  - Crypto non-IDR (kecuali sebagai referensi korelasi)
  - Forex non-IDR
  - Saham individual selain indeks IDX
  - Forward pesan dari grup lain
- **Frekuensi pengiriman**: Maks 6 sinyal per jam (IDX + crypto IDR aggregate)
  - Sub-limit: maks 3 sinyal IDX per jam
  - Sub-limit: maks 3 sinyal crypto IDR per jam
- **Format sinyal**: Lihat Bagian 4 (Format Sinyal Crypto) dan Bagian 6 (Format Sinyal IDX)

### 2.5 Grup Enterprise — Crypto, Forex & Saham Idx

- **Tier**: Enterprise
- **Asset yang dipantau**: Semua crypto + semua forex utama + semua indeks saham (IDX + global: S&P 500, NASDAQ, DOW, FTSE, NIKKEI)
- **Routing logic**:
  - Setiap symbol di-tag: `crypto`, `forex`, atau `stock`
  - Routing ke modul yang sesuai berdasarkan tag
  - Tidak ada pembatasan symbol (selama terdaftar di config)
- **Konten yang boleh diposting**:
  - Semua konten dari grup Pro
  - Sinyal entry/exit saham global (S&P 500, NASDAQ, dll.)
  - Analisis teknikal saham global
  - Update harga real-time saham global
  - Analisis korelasi multi-asset (crypto-forex-stock)
  - Laporan portfolio aggregate (jika diaktifkan)
  - Peringatan risk management multi-asset
- **Konten yang TIDAK boleh diposting**:
  - Forward pesan dari grup lain (kecuali laporan internal)
  - Sinyal tanpa timeframe dan confidence score
  - Konten promosi tanpa analisis dasar
- **Frekuensi pengiriman**: Maks 15 sinyal per jam (aggregate semua asset)
  - Sub-limit: maks 5 sinyal forex per jam
  - Sub-limit: maks 5 sinyal crypto per jam
  - Sub-limit: maks 5 sinyal saham per jam
- **Format sinyal**: Lihat Bagian 4, 5, dan 6

---

## 3. Format Sinyal Crypto (Bagian Inti)

### 3.1 Format Standar Sinyal Entry

```
🔔 SINYAL {TIER} — {SYMBOL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Pasangan: {SYMBOL}
📊 Arah: {LONG/SHORT}
💰 Entry: {PRICE}
🎯 Target 1: {TP1} | Target 2: {TP2} | Target 3: {TP3}
🛑 Stop Loss: {SL}
⏱ Timeframe: {TF}
📈 Indikator: {RSI: val, MACD: signal, BB: position}
🎯 Confidence: {HIGH/MEDIUM/LOW}
⚠ Risk/Reward: {R:R ratio}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ Risk Warning: Pastikan position size sesuai risk tolerance.
```

### 3.2 Format Standar Sinyal Exit

```
✅ EXIT {SYMBOL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Pasangan: {SYMBOL}
📊 Alasan: {TP reached / SL hit / Manual / Reversal signal}
💰 Exit Price: {PRICE}
📊 P&L: {PROFIT/LOSS} ({PCT}%)
⏱ Timeframe: {TF}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.3 Format Update Harga

```
📊 UPDATE {SYMBOL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Harga: {PRICE}
📈 24h Change: {+X% / -X%}
📊 Volume: {VOL}
🔺 High: {HIGH} | 🔻 Low: {LOW}
⏱ Waktu: {TIMESTAMP}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.4 Format Peringatan Risk

```
⚠ PERINGATAN RISK — {SYMBOL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Pasangan: {SYMBOL}
⚠ Risiko: {DESKRIPSI RISIKO}
📊 Volatilitas: {HIGH/MEDIUM/LOW}
📉 Drawdown: {CURRENT}%
💡 Rekomendasi: {REKOMENDASI}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 4. Format Sinyal Forex (Bagian 5)

### 4.1 Format Standar Sinyal Forex

```
🔔 SINYAL FOREX — {SYMBOL}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Pasangan: {SYMBOL}
📊 Arah: {BUY/SELL}
💰 Entry: {PRICE}
🎯 Target 1: {TP1} | Target 2: {TP2}
🛑 Stop Loss: {SL}
⏱ Timeframe: {TF}
📈 Analisis: {Technical summary}
📊 Sentimen: {BULLISH/BEARISH/NEUTRAL}
🎯 Confidence: {HIGH/MEDIUM/LOW}
⚠ Risk/Reward: {R:R ratio}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ Pastikan mempertimbangkan sesi trading yang aktif.
```

---

## 5. Format Sinyal IDX (Bagian 6)

### 5.1 Format Standar Sinyal Indeks IDX

```
🔔 SINYAL IDX — {INDEX_NAME}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 Indeks: {INDEX_NAME} ({TICKER})
📊 Arah: {BULLISH/BEARISH}
💰 Level: {CURRENT_LEVEL}
🎯 Target: {TP_LEVEL}
🛑 Support: {SUPPORT} | Resistance: {RESISTANCE}
⏱ Timeframe: {TF}
📈 Indikator: {RSI: val, MACD: signal}
🎯 Confidence: {HIGH/MEDIUM/LOW}
⚠ Risk/Reward: {R:R ratio}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ Data IDX diperbarui setiap sesi perdagangan.
```

---

## 6. Logika Routing Bot

### 6.1 Diagram Routing

```
Pesan/Signal Masuk
       │
       ▼
  ┌─────────────────┐
  │  Parser Symbol  │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Tentukan Asset │
  │  Type:          │
  │  - crypto       │
  │  - forex        │
  │  - stock        │
  │  - invalid      │
  └────────┬────────┘
           │
     ┌─────┼─────────┐
     ▼     ▼         ▼
  crypto  forex    stock
     │     │         │
     ▼     ▼         ▼
  ┌─────────────────────────┐
  │  Cek Tier Grup Penerima │
  │  Free  → crypto only    │
  │  Pro   → crypto+forex   │
  │        → atau crypto+IDX│
  │  Ent.  → crypto+forex+  │
  │          stock           │
  └────────┬────────────────┘
           │
     ┌─────┼─────────┐
     ▼     ▼         ▼
  DITERIMA  DIBLOKIR  DIBLOKIR
     │
     ▼
  ┌─────────────────┐
  │  Format Sinyal  │
  │  Sesuai Type    │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Cek Rate Limit │
  │  per grup       │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Kirim ke Grup  │
  └─────────────────┘
```

### 6.2 Aturan Routing Detail

| Asset Type | Grup Free (BTC/ETH) | Grup Free (All Crypto) | Grup Pro (Crypto+Forex) | Grup Pro (IDX) | Grup Enterprise |
|------------|---------------------|------------------------|-------------------------|----------------|-----------------|
| BTC/ETH    | ✅                  | ✅                     | ✅                      | ✅ (jika BTC/ETH) | ✅              |
| Altcoin    | ❌                  | ✅                     | ✅                      | ✅ (jika IDR pair) | ✅              |
| Forex      | ❌                  | ❌                     | ✅                      | ❌             | ✅              |
| IDX        | ❌                  | ❌                     | ❌                      | ✅             | ✅              |
| Stock Global | ❌               | ❌                     | ❌                      | ❌             | ✅              |
| Invalid    | ❌ (error msg)      | ❌ (error msg)         | ❌ (error msg)          | ❌ (error msg) | ❌ (error msg)  |

---

## 7. Aturan Anti-Spam dan Batasan Frekuensi

### 7.1 Global Rate Limits

| Tier  | Max Sinyal/Jam | Max Pesan Menit | Burst Limit |
|-------|---------------|-----------------|-------------|
| Free  | 3–5           | 10              | 3 berturut-turut |
| Pro   | 6–8           | 15              | 5 berturut-turut |
| Enterprise | 10–15     | 20              | 8 berturut-turut |

### 7.2 Aturan Anti-Spam Detail

1. **Cooldown per symbol**: Tidak boleh mengirim sinyal untuk symbol yang sama dalam < 15 menit (Free), < 10 menit (Pro), < 5 menit (Enterprise).
2. **Duplicate detection**: Jika sinyal baru identik dengan sinyal terakhir (same symbol, same direction, same entry ±0.1%), bot harus menambahkan `🔄 DUPLICATE` di awal pesan dan menurunkan priority.
3. **Burst protection**: Jika bot mendeteksi >3 sinyal dalam 5 menit, bot harus menghentikan pengiriman selama 2 menit dan mengirim pesan: `"⚠ Rate limit activated. Menunggu sebelum mengirim sinyal berikutnya."`
4. **Off-hours**: Tidak mengirim sinyal baru antara pukul 00:00–05:00 WIB (maintenance window). Sinyal yang dihasilkan di jam ini di-antrikan dan dikirim setelah 05:00 WIB.
5. **Error suppression**: Jika terjadi error pada modul analisis, bot tidak boleh mengirim pesan error mentah ke grup. Error harus dicatat ke log dan pengguna menerima: `"⚠ Sistem sedang dalam perawatan. Sinyal akan segera aktif kembali."`

### 7.3 Pesan Sistem yang Diizinkan

Bot boleh mengirim pesan berikut di luar sinyal:
- Pesan selamat datang untuk anggota baru
- Peringatan rate limit
- Pesan maintenance/off-hours
- Konfirmasi command (misal: `/status`, `/help`)
- Ringkasan harian (daily digest) — hanya untuk Pro dan Enterprise

---

## 8. Penanganan Pesan dan Interaksi Anggota

### 8.1 Command yang Didukung

| Command   | Deskripsi                          | Tier yang Bisa Gunakan |
|-----------|------------------------------------|------------------------|
| `/start`  | Pesan selamat datang + panduan     | Semua                  |
| `/help`   | Daftar command dan penggunaan      | Semua                  |
| `/status` | Status bot dan asset yang dipantau | Semua                  |
| `/signal {symbol}` | Minta sinyal manual untuk symbol tertentu | Free (crypto only), Pro (crypto+forex), Enterprise (semua) |
| `/watchlist` | Tampilkan asset yang dipantau di grup ini | Semua                  |
| `/config` | Lihat konfigurasi grup saat ini    | Pro, Enterprise        |
| `/history` | Riwayat sinyal terakhir (24 jam)  | Pro, Enterprise        |
| `/digest` | Ringkasan sinyal harian            | Pro, Enterprise        |

### 8.2 Aturan Interaksi Anggota

1. **Bot hanya merespons command dan sinyal** — tidak mereaksi emoji, poll, atau konten non-command lainnya.
2. **Pemrosesan command**:
   - Validasi input: jika symbol tidak valid untuk tier grup, kirim pesan error spesifik.
   - Response time: maks 3 detik untuk command, maks 30 detik untuk sinyal analisis.
3. **Anggota baru**:
   - Otomatis kirim pesan welcome dengan ringkasan aturan grup.
   - Tidak mengirim sinyal ke anggota baru selama 5 menit pertama (anti-spam perception).
4. **Laporan kesalahan**:
   - Anggota bisa mengirim `/report {deskripsi}` untuk melaporkan masalah.
   - Bot merespons: `"Laporan diterima. Tim akan meninjau."`
   - Tidak ada detail teknis yang dibagikan ke anggota.

### 8.3 Penanganan Pesan Tidak Valid

| Kondisi | Respons Bot |
|---------|-------------|
| Symbol tidak dikenal | `"❌ Symbol '{symbol}' tidak dikenal. Gunakan /watchlist untuk melihat asset yang tersedia."` |
| Symbol di luar tier grup | `"🚫 Grup ini tidak mendukung sinyal untuk {symbol}. Tier grup: {tier}. Asset yang didukung: {list}."` |
| Command tidak valid | `"❓ Command tidak dikenal. Ketik /help untuk daftar command."` |
| Rate limit terpicu | `"⏳ Rate limit aktif. Silakan tunggu {seconds} detik sebelum mengirim command berikutnya."` |
| Error internal | `"⚠ Terjadi kesalahan internal. Tim telah diberitahu."` |

---

## 9. Konfigurasi Teknis

### 9.1 Struktur Konfigurasi (config.yaml)

```yaml
telegram:
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  groups:
    - group_id: "free_btc_eth"
      name: "Free - BTC & ETH Only"
      tier: free
      allowed_assets:
        - BTC/USDT
        - ETH/USDT
      rate_limit:
        max_signals_per_hour: 3
        max_messages_per_minute: 10
        cooldown_minutes:
          per_symbol: 15
          burst_window_minutes: 5
          burst_max_count: 3
    - group_id: "free_all_crypto"
      name: "Free - All Crypto"
      tier: free
      allowed_assets: "all_crypto"
      rate_limit:
        max_signals_per_hour: 5
        max_messages_per_minute: 10
        cooldown_minutes:
          per_symbol: 15
          burst_window_minutes: 5
          burst_max_count: 3
    - group_id: "pro_crypto_forex"
      name: "Pro - All Crypto & Forex"
      tier: pro
      allowed_assets:
        - crypto: "all"
        - forex: "major"
      rate_limit:
        max_signals_per_hour: 8
        max_messages_per_minute: 15
        sub_limits:
          crypto: 5
          forex: 3
        cooldown_minutes:
          per_symbol: 10
          burst_window_minutes: 5
          burst_max_count: 5
    - group_id: "pro_indo_idx"
      name: "Pro - Indo Sham Idx"
      tier: pro
      allowed_assets:
        - idx: ["IHSG", "LQ45", "IDX30"]
        - crypto_idr: "all"
      rate_limit:
        max_signals_per_hour: 6
        max_messages_per_minute: 15
        sub_limits:
          idx: 3
          crypto_idr: 3
        cooldown_minutes:
          per_symbol: 10
          burst_window_minutes: 5
          burst_max_count: 5
    - group_id: "enterprise_all"
      name: "Enterprise - Crypto, Forex & Saham Idx"
      tier: enterprise
      allowed_assets:
        - crypto: "all"
        - forex: "all"
        - stock: "all"
      rate_limit:
        max_signals_per_hour: 15
        max_messages_per_minute: 20
        sub_limits:
          crypto: 5
          forex: 5
          stock: 5
        cooldown_minutes:
          per_symbol: 5
          burst_window_minutes: 5
          burst_max_count: 8
```

### 9.2 Environment Variables yang Dibutuhkan

```env
TELEGRAM_BOT_TOKEN=<bot_token_from_botfather>
TELEGRAM_FREE_BTC_ETH_GROUP_ID=<group_id>
TELEGRAM_FREE_ALL_CRYPTO_GROUP_ID=<group_id>
TELEGRAM_PRO_CRYPTO_FOREX_GROUP_ID=<group_id>
TELEGRAM_PRO_INDO_IDX_GROUP_ID=<group_id>
TELEGRAM_ENTERPRISE_ALL_GROUP_ID=<group_id>
```

---

## 10. Prioritas Pengembangan (Fase Crypto)

Urutan implementasi aturan di atas, berdasarkan prioritas crypto:

1. **Fase 1**: Implementasi routing dan filter untuk Grup Free — BTC & ETH Only (paling sederhana, jadi jadi fondasi)
2. **Fase 2**: Implementasi routing dan filter untuk Grup Free — All Crypto
3. **Fase 3**: Implementasi format sinyal crypto standar dan anti-spam global
4. **Fase 4**: Implementasi Grup Pro — All Crypto & Forex (perluas ke forex)
5. **Fase 5**: Implementasi Grup Pro — Indo Sham Idx (perluas ke IDX)
6. **Fase 6**: Implementasi Grup Enterprise — Crypto, Forex & Saham Idx (akses lengkap)
7. **Fase 7**: Fitur lanjutan — daily digest, korelasi multi-asset, laporan portfolio

Setiap fase harus melewati review aturan di Bagian 3 (Format Sinyal Crypto) sebelum beralih ke fase berikutnya.

---

## 11. Catatan Penting

- Semua sinyal crypto harus mencantumkan **confidence score** dan **risk/reward ratio**.
- Bot tidak boleh memberikan nasihat finansial eksplisit — semua sinyal harus diformulasikan sebagai **analisis teknikal**, bukan rekomendasi investasi.
- Pesan wajib mencantumkan disclaimer: `"⚠ Ini adalah analisis teknikal, bukan nasihat investasi. Gunakan dengan risiko Anda sendiri."`
- Semua timestamp harus dalam **WIB (UTC+7)**.
- Log semua sinyal yang dikirim ke database untuk audit trail.
- Dokumen ini harus direview setiap 3 bulan atau saat ada perubahan tier grup.
