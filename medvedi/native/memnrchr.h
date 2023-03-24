#include <immintrin.h>


static inline const char *memnrchr(const char *s, char c, size_t n) {
  const __m256i t = _mm256_set1_epi8(c);
  size_t i;
  for (i = n; i >= 32; i -= 32) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i - 32));
    unsigned m = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, t));
    if (m) {
      return s + i - __builtin_clz(m);
    }
  }
  while (i--) {
    if (s[i] != c) {
      return s + i + 1;
    }
  }
  return NULL;
}


static inline const wchar_t *wmemnrchr(const wchar_t *s, wchar_t c, size_t n) {
  const __m256i t = _mm256_set1_epi32(c);
  size_t i;
  for (i = n; i >= 8; i -= 8) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i - 8));
    unsigned m = ~_mm256_movemask_epi8(_mm256_cmpeq_epi32(v, t));
    if (m) {
      return s + i - (__builtin_clz(m) >> 2);
    }
  }
  while (i--) {
    if (s[i] != c) {
      return s + i + 1;
    }
  }
  return NULL;
}