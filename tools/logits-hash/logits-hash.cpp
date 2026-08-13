// llama-logits-hash
//
// Hash the raw f32 logits produced for a prompt to fingerprint model output.
// The digest covers the exact bytes of llama_get_logits_ith() for each token
// in order: f32[n_vocab] per token.
//
// Same binary + same config produce the same digest, which allows comparing
// builds and split modes for bit-exact output. Different configs legitimately
// produce different digests (e.g. -sm layer vs tensor, different -ctk/-ctv).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "xxhash/xxhash.h"
#include "sha1/sha1.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "sha256/sha256.h"
#ifdef __cplusplus
}
#endif

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_hex(const unsigned char * data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%02x", data[i]);
    }
}

struct hash_modes {
    bool xxh64  = true;
    bool sha1   = false;
    bool sha256 = false;
};

struct hash_states {
    hash_modes modes;

    XXH64_state_t * xxh64 = nullptr;
    SHA1_CTX sha1;
    sha256_t sha256;

    hash_states(const hash_modes & modes) : modes(modes) {
        if (modes.xxh64) {
            xxh64 = XXH64_createState();
            XXH64_reset(xxh64, 0);
        }
        if (modes.sha1) {
            SHA1Init(&sha1);
        }
        if (modes.sha256) {
            sha256_init(&sha256);
        }
    }

    ~hash_states() {
        if (xxh64 != nullptr) {
            XXH64_freeState(xxh64);
        }
    }

    void update(const void * data, size_t size) {
        if (modes.xxh64) {
            XXH64_update(xxh64, data, size);
        }
        if (modes.sha1) {
            SHA1Update(&sha1, (const unsigned char *) data, size);
        }
        if (modes.sha256) {
            sha256_update(&sha256, (const unsigned char *) data, size);
        }
    }

    void print_digest(const char * label) {
        if (modes.xxh64) {
            const uint64_t digest = XXH64_digest(xxh64);
            printf("xxh64 %016" PRIx64 " %s\n", digest, label);
            XXH64_reset(xxh64, 0);
        }
        if (modes.sha1) {
            unsigned char digest[20];
            SHA1Final(digest, &sha1);
            printf("sha1  ");
            print_hex(digest, sizeof(digest));
            printf(" %s\n", label);
            SHA1Init(&sha1);
        }
        if (modes.sha256) {
            unsigned char digest[32];
            sha256_final(&sha256, digest);
            printf("sha256 ");
            print_hex(digest, sizeof(digest));
            printf(" %s\n", label);
            sha256_init(&sha256);
        }
    }
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOGITS_HASH)) {
        return 1;
    }

    if (params.prompt.empty()) {
        fprintf(stderr, "%s: no prompt provided - use -p or -f\n", __func__);
        return 1;
    }

    common_init_result_ptr llama_init = common_init_from_params(params);

    if (!llama_init) {
        fprintf(stderr, "%s: failed to initialize llama\n", __func__);
        return 1;
    }

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        return 1;
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n_ctx   = llama_n_ctx(ctx);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

    const size_t n_tokens_total = params.n_logits_tokens > 0
        ? std::min((size_t) params.n_logits_tokens, tokens.size())
        : tokens.size();

    if (n_tokens_total > (size_t) n_ctx) {
        fprintf(stderr, "%s: %zu tokens to hash but n_ctx is %d - increase -c\n", __func__, n_tokens_total, n_ctx);
        return 1;
    }

    tokens.resize(n_tokens_total);

    hash_modes modes;
    modes.xxh64  = !params.logits_hash_sha1 && !params.logits_hash_sha256;
    modes.sha1   = params.logits_hash_sha1;
    modes.sha256 = params.logits_hash_sha256;

    hash_states st_total(modes);
    hash_states st_window(modes);

    const int32_t n_batch = params.n_batch;

    LOG_INF("%s: hashing %zu tokens, vocab %d, n_ctx %d, n_batch %d\n", __func__, n_tokens_total, n_vocab, n_ctx, n_batch);

    // decode in chunks and hash the logits of each token; chunked decoding
    // produces the same per-token logits as a single large batch because the
    // kv cache carries the context forward
    constexpr int32_t window_size = 100;
    size_t n_hashed = 0;

    for (size_t i = 0; i < n_tokens_total; i += n_batch) {
        const size_t chunk_size = std::min((size_t) n_batch, n_tokens_total - i);

        llama_batch batch = llama_batch_init(chunk_size, 0, 1);
        for (size_t j = 0; j < chunk_size; j++) {
            batch.token[j]     = tokens[i + j];
            batch.pos[j]       = i + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        batch.n_tokens = chunk_size;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: llama_decode failed\n", __func__);
            llama_batch_free(batch);
            return 1;
        }

        for (size_t j = 0; j < chunk_size; j++) {
            const float * logits = llama_get_logits_ith(ctx, j);
            if (logits == nullptr) {
                fprintf(stderr, "%s: no logits for token %zu\n", __func__, n_hashed + j);
                llama_batch_free(batch);
                return 1;
            }
            st_total.update(logits, n_vocab * sizeof(float));
            st_window.update(logits, n_vocab * sizeof(float));
            n_hashed++;

            if (n_hashed % window_size == 0) {
                const std::string label = "logits:tokens_" + std::to_string(n_hashed - window_size) + "-" + std::to_string(n_hashed - 1);
                st_window.print_digest(label.c_str());
            }
        }
        llama_batch_free(batch);
    }

    if (n_hashed % window_size != 0) {
        const size_t start = (n_hashed / window_size) * window_size;
        const std::string label = "logits:tokens_" + std::to_string(start) + "-" + std::to_string(n_hashed - 1);
        st_window.print_digest(label.c_str());
    }

    st_total.print_digest("logits:total");

    return 0;
}
