#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

static int g_fails = 0;

static void expect(bool cond, const char * msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fails++;
    }
}

int main() {
    try {
        const auto types = common_speculative_types_from_names({ "draft-dspark" });
        expect(types.size() == 1, "draft-dspark parses to one type");
        if (!types.empty()) {
            expect(types[0] == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, "draft-dspark maps to DRAFT_DSPARK");
            expect(common_speculative_type_to_str(types[0]) == "draft-dspark", "DRAFT_DSPARK stringifies");
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "FAIL: draft-dspark name: %s\n", e.what());
        g_fails++;
    }

    expect(common_speculative_type_from_name("draft-dspark") == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
           "type_from_name(draft-dspark)");

    {
        common_params_speculative spec;
        spec.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        spec.draft.n_max = 8;
        expect(spec.need_n_rs_seq() == 8, "draft-dspark requests n_rs_seq");
    }

    const char * env_draft = std::getenv("LFM25_DRAFT");
    std::string gguf;
    if (env_draft && env_draft[0]) {
        gguf = env_draft;
    } else if (const char * home = std::getenv("HOME")) {
        gguf = std::string(home) + "/models/lfm2.5-8b-a1b/LFM2.5-8B-A1B-DSpark-Q8_0.gguf";
    }

    if (!gguf.empty()) {
        FILE * f = std::fopen(gguf.c_str(), "rb");
        if (f) {
            std::fclose(f);
            const auto detected = common_speculative_types_from_gguf(gguf);
            expect(detected.size() == 1, "DSpark GGUF detects one type");
            if (!detected.empty()) {
                expect(detected[0] == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,
                       "DSpark GGUF auto-detects draft-dspark");
            }
        }
    }

    if (g_fails) {
        fprintf(stderr, "%d check(s) failed\n", g_fails);
        return 1;
    }
    printf("test-dspark-type: ok\n");
    return 0;
}
