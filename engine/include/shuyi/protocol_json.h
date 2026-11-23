// The engine's wire format.
//
// One protocol line per message, `info <json>` for asynchronous analysis and
// `= ...` / `? ...` for command results.  This header owns the JSON side so the
// command parser stays readable and the schema lives in one place.
#pragma once

#include "shuyi/engine.h"

#include <string>

namespace shuyi {

// The analysis as one JSON object, without a trailing newline.  Every number is
// printed with a fixed precision so two runs of the same search produce
// byte-identical lines (they are compared in the conformance tests).
// `aInput` reports whether the loaded model can see the A value (v4 models) or
// is A-blind (v3): a front end that shows a win rate must be able to say which.
std::string analysisJson(const Analysis& analysis, const std::string& backend,
                         const std::string& turn, long long elapsedMs,
                         const std::string& nn, bool aInput);

} // namespace shuyi
