#include "json.h"

#include <cstdlib>
#include <cstring>

namespace shuyi {
namespace {

class Parser {
public:
    Parser(const std::string& input) : input_(input) {}

    bool parse(JsonValue& out) {
        skipSpace();
        if (!parseValue(out)) return false;
        skipSpace();
        if (position_ != input_.size()) return fail("trailing characters");
        return true;
    }

    const std::string& error() const { return error_; }

private:
    bool fail(const char* what) {
        if (error_.empty()) {
            error_ = std::string(what) + " at byte " + std::to_string(position_);
        }
        return false;
    }

    void skipSpace() {
        while (position_ < input_.size()) {
            const char c = input_[position_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') ++position_;
            else break;
        }
    }

    bool consume(char expected) {
        if (position_ >= input_.size() || input_[position_] != expected) return false;
        ++position_;
        return true;
    }

    bool parseValue(JsonValue& out) {
        skipSpace();
        if (position_ >= input_.size()) return fail("unexpected end of input");
        const char c = input_[position_];
        if (c == '{') return parseObject(out);
        if (c == '[') return parseArray(out);
        if (c == '"') {
            out.type = JsonValue::Type::String;
            return parseString(out.text);
        }
        if (c == 't' || c == 'f') {
            if (input_.compare(position_, 4, "true") == 0) {
                position_ += 4;
                out.type = JsonValue::Type::Bool;
                out.boolean = true;
                return true;
            }
            if (input_.compare(position_, 5, "false") == 0) {
                position_ += 5;
                out.type = JsonValue::Type::Bool;
                out.boolean = false;
                return true;
            }
            return fail("bad literal");
        }
        if (c == 'n') {
            if (input_.compare(position_, 4, "null") == 0) {
                position_ += 4;
                out.type = JsonValue::Type::Null;
                return true;
            }
            return fail("bad literal");
        }
        return parseNumber(out);
    }

    bool parseNumber(JsonValue& out) {
        const std::size_t start = position_;
        if (position_ < input_.size() && (input_[position_] == '-' || input_[position_] == '+')) {
            ++position_;
        }
        bool digits = false;
        while (position_ < input_.size()) {
            const char c = input_[position_];
            if ((c >= '0' && c <= '9')) {
                digits = true;
                ++position_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                ++position_;
            } else {
                break;
            }
        }
        if (!digits) return fail("expected a number");
        const std::string token = input_.substr(start, position_ - start);
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end != token.c_str() + token.size()) return fail("bad number");
        out.type = JsonValue::Type::Number;
        out.number = value;
        return true;
    }

    bool parseString(std::string& out) {
        if (!consume('"')) return fail("expected a string");
        out.clear();
        while (position_ < input_.size()) {
            const char c = input_[position_++];
            if (c == '"') return true;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (position_ >= input_.size()) return fail("unterminated escape");
            const char escape = input_[position_++];
            switch (escape) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                if (position_ + 4 > input_.size()) return fail("short \\u escape");
                unsigned code = 0;
                for (int i = 0; i < 4; ++i) {
                    const char digit = input_[position_++];
                    code <<= 4;
                    if (digit >= '0' && digit <= '9') code |= static_cast<unsigned>(digit - '0');
                    else if (digit >= 'a' && digit <= 'f') code |= static_cast<unsigned>(digit - 'a' + 10);
                    else if (digit >= 'A' && digit <= 'F') code |= static_cast<unsigned>(digit - 'A' + 10);
                    else return fail("bad \\u escape");
                }
                // The protocol is ASCII apart from model paths, which arrive as
                // UTF-8 bytes already; a UTF-16 escape is encoded back to UTF-8.
                if (code < 0x80) {
                    out += static_cast<char>(code);
                } else if (code < 0x800) {
                    out += static_cast<char>(0xC0 | (code >> 6));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                } else {
                    out += static_cast<char>(0xE0 | (code >> 12));
                    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                }
                break;
            }
            default: return fail("unknown escape");
            }
        }
        return fail("unterminated string");
    }

    bool parseArray(JsonValue& out) {
        if (!consume('[')) return fail("expected [");
        out.type = JsonValue::Type::Array;
        skipSpace();
        if (consume(']')) return true;
        while (true) {
            JsonValue item;
            if (!parseValue(item)) return false;
            out.items.push_back(std::move(item));
            skipSpace();
            if (consume(',')) continue;
            if (consume(']')) return true;
            return fail("expected , or ]");
        }
    }

    bool parseObject(JsonValue& out) {
        if (!consume('{')) return fail("expected {");
        out.type = JsonValue::Type::Object;
        skipSpace();
        if (consume('}')) return true;
        while (true) {
            skipSpace();
            std::string key;
            if (!parseString(key)) return false;
            skipSpace();
            if (!consume(':')) return fail("expected :");
            JsonValue value;
            if (!parseValue(value)) return false;
            out.members.emplace_back(std::move(key), std::move(value));
            skipSpace();
            if (consume(',')) continue;
            if (consume('}')) return true;
            return fail("expected , or }");
        }
    }

    const std::string& input_;
    std::size_t position_ = 0;
    std::string error_;
};

} // namespace

const JsonValue* JsonValue::find(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& member : members) {
        if (member.first == key) return &member.second;
    }
    return nullptr;
}

int JsonValue::asInt(int fallback) const {
    if (!isNumber()) return fallback;
    return static_cast<int>(number < 0 ? number - 0.5 : number + 0.5);
}

bool JsonValue::asBool(bool fallback) const {
    if (type == Type::Bool) return boolean;
    if (isNumber()) return number != 0.0;
    return fallback;
}

bool parseJson(const std::string& input, JsonValue& out, std::string* error) {
    Parser parser(input);
    if (!parser.parse(out)) {
        if (error) *error = parser.error();
        return false;
    }
    return true;
}

} // namespace shuyi
