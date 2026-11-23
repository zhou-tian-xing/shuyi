// A tiny JSON reader for the engine's analysis lines.
//
// The engine emits one JSON object per `info` line (protocol/ENGINE_PROTOCOL.md).
// That object is the whole analysis: visit counts, win rate, score, the best
// move, the principal variation and one entry per board cell.  A hand written
// reader keeps the UI free of a dependency and makes the accepted grammar
// explicit: objects, arrays, strings, numbers, true/false/null, nothing else.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace shuyi {

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<JsonValue> items;
    std::vector<std::pair<std::string, JsonValue>> members;

    bool isNull() const { return type == Type::Null; }
    bool isNumber() const { return type == Type::Number; }
    bool isString() const { return type == Type::String; }
    bool isArray() const { return type == Type::Array; }
    bool isObject() const { return type == Type::Object; }

    // Object member, or nullptr when this is not an object or the key is absent.
    const JsonValue* find(const std::string& key) const;

    double asNumber(double fallback = 0.0) const { return isNumber() ? number : fallback; }
    int asInt(int fallback = 0) const;
    bool asBool(bool fallback = false) const;
    const std::string& asString() const { return text; }
};

// Parses one complete JSON document.  On failure returns false and fills
// `error` with the byte offset and what was expected.
bool parseJson(const std::string& input, JsonValue& out, std::string* error);

} // namespace shuyi
